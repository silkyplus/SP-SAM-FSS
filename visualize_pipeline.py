"""
SP-SAM Pipeline 可视化
======================

模拟训练/推理流程，在各阶段进行可视化

流程：
1. 加载Support和Query
2. 提取DINO特征 → 可视化
3. 计算Prototype → 可视化
4. 计算Similarity Map → 可视化
5. 采样Point Prompts → 可视化
6. SAM2生成Mask → 可视化
7. (可选) Memory精炼 → 可视化

使用方法：
    python visualize_pipeline.py --data_root ISIC2018_256 --k_shot 3
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

# 尝试导入PCA
try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except:
    HAS_PCA = False


class PipelineVisualizer:
    """Pipeline各阶段可视化器"""
    
    def __init__(self, output_dir='vis_pipeline'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
    
    def _save_fig(self, name):
        """保存当前图像"""
        path = self.output_dir / f'{self.step:02d}_{name}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   💾 保存: {path}")
        self.step += 1
    
    def vis_input(self, query_img, query_mask, support_imgs, support_masks):
        """Step 0: 可视化输入"""
        k = len(support_imgs)
        fig, axes = plt.subplots(2, k + 1, figsize=(4*(k+1), 8))
        
        # Query
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(query_mask, cmap='gray')
        axes[1, 0].set_title('Query GT Mask')
        axes[1, 0].axis('off')
        
        # Support
        for i in range(k):
            axes[0, i+1].imshow(support_imgs[i])
            axes[0, i+1].set_title(f'Support {i+1}')
            axes[0, i+1].axis('off')
            
            axes[1, i+1].imshow(support_masks[i], cmap='gray')
            axes[1, i+1].set_title(f'Support {i+1} Mask')
            axes[1, i+1].axis('off')
        
        plt.suptitle('Step 0: Input Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('input')
    
    def vis_dino_features(self, query_feat, support_feats, query_img, support_imgs):
        """Step 1: 可视化DINO特征"""
        k = len(support_feats)
        fig, axes = plt.subplots(3, k + 1, figsize=(4*(k+1), 12))
        
        # Query特征
        C, H, W = query_feat.shape
        
        # Row 1: 原图
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        for i in range(k):
            axes[0, i+1].imshow(support_imgs[i])
            axes[0, i+1].set_title(f'Support {i+1}')
            axes[0, i+1].axis('off')
        
        # Row 2: 特征范数
        query_norm = torch.norm(query_feat, dim=0).cpu().numpy()
        im = axes[1, 0].imshow(query_norm, cmap='viridis')
        axes[1, 0].set_title(f'Query Feature Norm\n[{C}×{H}×{W}]')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        for i, sf in enumerate(support_feats):
            sf_norm = torch.norm(sf, dim=0).cpu().numpy()
            im = axes[1, i+1].imshow(sf_norm, cmap='viridis')
            axes[1, i+1].set_title(f'Support {i+1} Norm')
            axes[1, i+1].axis('off')
            plt.colorbar(im, ax=axes[1, i+1], fraction=0.046)
        
        # Row 3: PCA可视化
        if HAS_PCA:
            # Query PCA
            feat_flat = query_feat.reshape(C, -1).T.cpu().numpy()
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat).reshape(H, W, 3)
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
            axes[2, 0].imshow(feat_pca)
            axes[2, 0].set_title(f'Query PCA\nVar: {pca.explained_variance_ratio_.sum()*100:.1f}%')
            axes[2, 0].axis('off')
            
            for i, sf in enumerate(support_feats):
                Cs, Hs, Ws = sf.shape
                sf_flat = sf.reshape(Cs, -1).T.cpu().numpy()
                sf_pca = pca.transform(sf_flat).reshape(Hs, Ws, 3)
                sf_pca = (sf_pca - sf_pca.min()) / (sf_pca.max() - sf_pca.min() + 1e-8)
                axes[2, i+1].imshow(sf_pca)
                axes[2, i+1].set_title(f'Support {i+1} PCA')
                axes[2, i+1].axis('off')
        else:
            for j in range(k + 1):
                axes[2, j].axis('off')
        
        plt.suptitle('Step 1: DINO Feature Extraction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('dino_features')
    
    def vis_prototype(self, fg_prototypes, bg_prototypes, support_masks_down):
        """Step 2: 可视化Prototype"""
        k = len(fg_prototypes)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 各Support的FG Prototype
        for i, proto in enumerate(fg_prototypes):
            axes[0, 0].plot(proto[:100].cpu().numpy(), alpha=0.7, label=f'Support {i+1}')
        
        avg_fg = torch.stack(fg_prototypes).mean(dim=0)
        axes[0, 0].plot(avg_fg[:100].cpu().numpy(), 'k-', linewidth=2, label='Average')
        axes[0, 0].set_xlabel('Channel (first 100)')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Foreground Prototypes')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 各Support的BG Prototype
        for i, proto in enumerate(bg_prototypes):
            axes[0, 1].plot(proto[:100].cpu().numpy(), alpha=0.7, label=f'Support {i+1}')
        
        avg_bg = torch.stack(bg_prototypes).mean(dim=0)
        axes[0, 1].plot(avg_bg[:100].cpu().numpy(), 'k-', linewidth=2, label='Average')
        axes[0, 1].set_xlabel('Channel (first 100)')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Background Prototypes')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # FG vs BG对比
        axes[0, 2].plot(avg_fg[:100].cpu().numpy(), 'g-', linewidth=2, label='FG')
        axes[0, 2].plot(avg_bg[:100].cpu().numpy(), 'r-', linewidth=2, label='BG')
        axes[0, 2].fill_between(range(100), 
                                avg_fg[:100].cpu().numpy(), 
                                avg_bg[:100].cpu().numpy(), 
                                alpha=0.3, color='yellow')
        axes[0, 2].set_xlabel('Channel')
        axes[0, 2].set_title('FG vs BG Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 差异分布
        diff = (avg_fg - avg_bg).cpu().numpy()
        axes[1, 0].hist(diff, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('FG - BG')
        axes[1, 0].set_title(f'Prototype Difference\nMean: {diff.mean():.4f}, Std: {diff.std():.4f}')
        
        # 余弦相似度
        cos_sim = F.cosine_similarity(avg_fg.unsqueeze(0), avg_bg.unsqueeze(0)).item()
        axes[1, 1].bar(['FG-BG Similarity'], [cos_sim], color='orange', edgecolor='black')
        axes[1, 1].set_ylim([-1, 1])
        axes[1, 1].axhline(0, color='gray', linestyle='--')
        axes[1, 1].set_title(f'Cosine Similarity: {cos_sim:.4f}\n(越小越好区分)')
        
        # 显示下采样的mask
        axes[1, 2].set_title('Downsampled Support Masks')
        for i, mask in enumerate(support_masks_down[:4]):
            ax_sub = axes[1, 2].inset_axes([i*0.25, 0, 0.24, 1])
            ax_sub.imshow(mask.cpu().numpy(), cmap='gray')
            ax_sub.set_title(f'S{i+1}', fontsize=8)
            ax_sub.axis('off')
        axes[1, 2].axis('off')
        
        plt.suptitle('Step 2: Prototype Computation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('prototype')
    
    def vis_similarity(self, fg_sim, bg_sim, contrast_sim, query_img):
        """Step 3: 可视化Similarity Map"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        fg_np = fg_sim.cpu().numpy()
        bg_np = bg_sim.cpu().numpy()
        contrast_np = contrast_sim.cpu().numpy()
        
        # 原图
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        # FG Similarity
        im1 = axes[0, 1].imshow(fg_np, cmap='hot')
        axes[0, 1].set_title(f'FG Similarity\nMin: {fg_np.min():.3f}, Max: {fg_np.max():.3f}')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # BG Similarity
        im2 = axes[0, 2].imshow(bg_np, cmap='hot')
        axes[0, 2].set_title(f'BG Similarity\nMin: {bg_np.min():.3f}, Max: {bg_np.max():.3f}')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Contrast Similarity (FG - BG)
        vmax = max(abs(contrast_np.min()), abs(contrast_np.max()))
        im3 = axes[1, 0].imshow(contrast_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, 0].set_title(f'Contrast (FG - BG)\nMean: {contrast_np.mean():.3f}')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Thresholded
        threshold = contrast_np.mean() + 0.5 * contrast_np.std()
        binary = (contrast_np > threshold).astype(float)
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title(f'Thresholded (>{threshold:.3f})')
        axes[1, 1].axis('off')
        
        # 分布直方图
        axes[1, 2].hist(fg_np.flatten(), bins=50, alpha=0.5, label='FG', color='green')
        axes[1, 2].hist(bg_np.flatten(), bins=50, alpha=0.5, label='BG', color='red')
        axes[1, 2].set_xlabel('Similarity')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Similarity Distribution')
        axes[1, 2].legend()
        
        plt.suptitle('Step 3: Similarity Map Computation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('similarity')
    
    def vis_points(self, similarity_map, point_coords, point_labels, query_img):
        """Step 4: 可视化Point Prompts"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sim_np = similarity_map.cpu().numpy()
        H, W = sim_np.shape
        
        # 相似度图 + 点
        axes[0].imshow(sim_np, cmap='hot')
        for (x, y), label in zip(point_coords, point_labels):
            color = 'lime' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            axes[0].scatter(x, y, c=color, marker=marker, s=150, edgecolors='white', linewidths=2)
        axes[0].set_title(f'Similarity + Points\nPos: {sum(point_labels)}, Neg: {len(point_labels)-sum(point_labels)}')
        axes[0].axis('off')
        
        # 原图 + 点（需要缩放坐标）
        img_h, img_w = np.array(query_img).shape[:2]
        scale_x, scale_y = img_w / W, img_h / H
        
        axes[1].imshow(query_img)
        for (x, y), label in zip(point_coords, point_labels):
            px, py = x * scale_x, y * scale_y
            color = 'lime' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            axes[1].scatter(px, py, c=color, marker=marker, s=200, edgecolors='white', linewidths=3)
        axes[1].set_title('Points on Original Image')
        axes[1].axis('off')
        
        # 点坐标详情
        pos_coords = [(x, y) for (x, y), l in zip(point_coords, point_labels) if l == 1]
        neg_coords = [(x, y) for (x, y), l in zip(point_coords, point_labels) if l == 0]
        
        text = "Positive Points (Feature Scale):\n"
        for i, (x, y) in enumerate(pos_coords[:5]):
            text += f"  P{i+1}: ({x:.0f}, {y:.0f}) sim={sim_np[int(y), int(x)]:.3f}\n"
        text += "\nNegative Points:\n"
        for i, (x, y) in enumerate(neg_coords[:5]):
            text += f"  N{i+1}: ({x:.0f}, {y:.0f}) sim={sim_np[int(y), int(x)]:.3f}\n"
        
        axes[2].text(0.1, 0.9, text, transform=axes[2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].set_title('Point Details')
        axes[2].axis('off')
        
        plt.suptitle('Step 4: Point Prompt Sampling', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('points')
    
    def vis_sam_output(self, masks, scores, query_img, gt_mask=None):
        """Step 5: 可视化SAM2输出"""
        n_masks = len(masks)
        fig, axes = plt.subplots(2, max(4, n_masks + 1), figsize=(4 * max(4, n_masks + 1), 8))
        
        # Row 1: 各个Mask
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        for i in range(n_masks):
            axes[0, i + 1].imshow(masks[i], cmap='gray')
            axes[0, i + 1].set_title(f'Mask {i+1}\nScore: {scores[i]:.3f}')
            axes[0, i + 1].axis('off')
        
        # 隐藏多余的
        for i in range(n_masks + 1, axes.shape[1]):
            axes[0, i].axis('off')
        
        # Row 2: 最佳Mask对比
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        axes[1, 0].imshow(query_img)
        axes[1, 0].set_title('Query')
        axes[1, 0].axis('off')
        
        if gt_mask is not None:
            axes[1, 1].imshow(gt_mask, cmap='gray')
            axes[1, 1].set_title('GT Mask')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].axis('off')
        
        axes[1, 2].imshow(best_mask, cmap='gray')
        iou_text = ''
        if gt_mask is not None:
            inter = (best_mask > 0) & (gt_mask > 0)
            union = (best_mask > 0) | (gt_mask > 0)
            iou = inter.sum() / (union.sum() + 1e-8)
            iou_text = f'\nIoU: {iou*100:.1f}%'
        axes[1, 2].set_title(f'Best Mask (#{best_idx + 1}){iou_text}')
        axes[1, 2].axis('off')
        
        # Overlay
        overlay = np.array(query_img).copy()
        mask_bool = best_mask > 0
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 255 * 0.5, 0, 255)
        axes[1, 3].imshow(overlay.astype(np.uint8))
        axes[1, 3].set_title('Overlay')
        axes[1, 3].axis('off')
        
        for i in range(4, axes.shape[1]):
            axes[1, i].axis('off')
        
        plt.suptitle('Step 5: SAM2 Mask Generation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('sam_output')
    
    def vis_final_comparison(self, query_img, gt_mask, rough_mask, final_mask=None):
        """Step 6: 最终结果对比"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Row 1: 各阶段mask
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title('GT Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(rough_mask, cmap='gray')
        rough_iou = self._compute_iou(rough_mask, gt_mask)
        axes[0, 2].set_title(f'Rough Mask\nIoU: {rough_iou*100:.1f}%')
        axes[0, 2].axis('off')
        
        if final_mask is not None:
            axes[0, 3].imshow(final_mask, cmap='gray')
            final_iou = self._compute_iou(final_mask, gt_mask)
            axes[0, 3].set_title(f'Final Mask (Memory)\nIoU: {final_iou*100:.1f}%')
        else:
            axes[0, 3].text(0.5, 0.5, 'No Memory\nRefinement', ha='center', va='center')
        axes[0, 3].axis('off')
        
        # Row 2: 对比可视化
        # GT overlay
        overlay_gt = np.array(query_img).copy()
        overlay_gt[gt_mask > 0, 1] = np.clip(overlay_gt[gt_mask > 0, 1] * 0.5 + 255 * 0.5, 0, 255)
        axes[1, 0].imshow(overlay_gt.astype(np.uint8))
        axes[1, 0].set_title('GT Overlay (Green)')
        axes[1, 0].axis('off')
        
        # Rough overlay
        overlay_rough = np.array(query_img).copy()
        overlay_rough[rough_mask > 0, 0] = np.clip(overlay_rough[rough_mask > 0, 0] * 0.5 + 255 * 0.5, 0, 255)
        axes[1, 1].imshow(overlay_rough.astype(np.uint8))
        axes[1, 1].set_title('Rough Overlay (Red)')
        axes[1, 1].axis('off')
        
        # 对比图: GT(G) vs Rough(R)
        comparison = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        comparison[gt_mask > 0, 1] = 255  # GT: Green
        comparison[rough_mask > 0, 0] = 255  # Pred: Red
        comparison[(gt_mask > 0) & (rough_mask > 0)] = [255, 255, 0]  # Overlap: Yellow
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('GT(G) vs Pred(R)\nOverlap(Y)')
        axes[1, 2].axis('off')
        
        # 指标
        dice = self._compute_dice(rough_mask, gt_mask)
        precision = self._compute_precision(rough_mask, gt_mask)
        recall = self._compute_recall(rough_mask, gt_mask)
        
        metrics_text = f"""
        Metrics:
        ─────────────
        IoU:       {rough_iou*100:.2f}%
        Dice:      {dice*100:.2f}%
        Precision: {precision*100:.2f}%
        Recall:    {recall*100:.2f}%
        """
        axes[1, 3].text(0.1, 0.7, metrics_text, transform=axes[1, 3].transAxes,
                       fontsize=12, fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 3].axis('off')
        
        plt.suptitle('Step 6: Final Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_fig('final_comparison')
    
    def _compute_iou(self, pred, gt):
        inter = ((pred > 0) & (gt > 0)).sum()
        union = ((pred > 0) | (gt > 0)).sum()
        return inter / (union + 1e-8)
    
    def _compute_dice(self, pred, gt):
        inter = ((pred > 0) & (gt > 0)).sum()
        return 2 * inter / ((pred > 0).sum() + (gt > 0).sum() + 1e-8)
    
    def _compute_precision(self, pred, gt):
        tp = ((pred > 0) & (gt > 0)).sum()
        return tp / ((pred > 0).sum() + 1e-8)
    
    def _compute_recall(self, pred, gt):
        tp = ((pred > 0) & (gt > 0)).sum()
        return tp / ((gt > 0).sum() + 1e-8)


def run_pipeline_visualization(data_root, k_shot=3, sample_idx=0, output_dir='vis_pipeline',
                               dino_model='dinov3_vitb16', sam2_model='large', device='cuda'):
    """
    运行完整的Pipeline可视化
    """
    print("="*60)
    print("SP-SAM Pipeline Visualization")
    print("="*60)
    
    # ==================== 1. 加载模型 ====================
    print("\n🚀 Step 0: 加载模型...")
    
    from sp_sam_complete import SPSAMModel, CMRSModule
    from src.model_manager import ModelManager
    
    model_manager = ModelManager(device=device)
    
    if 'dinov3' in dino_model.lower():
        dino, dino_transform = model_manager.load_dinov3_model(dinov3_model_name=dino_model)
    else:
        dino, dino_transform = model_manager.load_dinov2_model(dinov2_model_name=dino_model)
    
    sam2, sam2_predictor, _ = model_manager.load_sam2_model(sam2_model_type=sam2_model)
    
    # 创建CMRS模块（用于中间特征提取）
    cmrs = CMRSModule(dino, dino_transform, device)
    
    # 创建SP-SAM（用于完整预测）
    sp_sam = SPSAMModel(
        sam2_model=sam2,
        sam2_predictor=sam2_predictor,
        dino_model=dino,
        dino_transform=dino_transform,
        device=device,
        sam2_model_type=sam2_model
    )
    
    # ==================== 2. 加载数据 ====================
    print("\n📁 加载数据...")
    
    from isic2018_dataset import ISIC2018Dataset
    dataset = ISIC2018Dataset(data_root, support_ratio=0.2)
    episode = dataset.get_episode(k_shot=k_shot)
    
    support_imgs = [s['img'] for s in episode['support']]
    support_masks = [s['mask'] for s in episode['support']]
    
    query_sample = episode['query'][sample_idx]
    query_img = query_sample['img']
    query_mask = query_sample['mask']
    
    print(f"   Query: {query_sample['sample_id']}")
    print(f"   Support: {k_shot} samples")
    
    # ==================== 3. 创建可视化器 ====================
    visualizer = PipelineVisualizer(output_dir)
    
    # ==================== Step 0: 可视化输入 ====================
    print("\n🎨 Step 0: 可视化输入...")
    visualizer.vis_input(query_img, query_mask, support_imgs, support_masks)
    
    # ==================== Step 1: DINO特征提取 ====================
    print("\n🎨 Step 1: DINO特征提取...")
    
    # Query特征
    query_feat = cmrs.get_features(query_img)
    
    # Support特征
    support_feats = []
    support_masks_down = []
    for sup_img, sup_mask in zip(support_imgs, support_masks):
        sup_feat = cmrs.get_features(sup_img)
        support_feats.append(sup_feat)
        
        H_f, W_f = sup_feat.shape[1], sup_feat.shape[2]
        mask_down = cmrs.downsample_mask(sup_mask, (H_f, W_f))
        support_masks_down.append(mask_down)
    
    visualizer.vis_dino_features(query_feat, support_feats, query_img, support_imgs)
    
    # ==================== Step 2: Prototype计算 ====================
    print("\n🎨 Step 2: Prototype计算...")
    
    fg_prototypes = []
    bg_prototypes = []
    
    for sup_feat, sup_mask in zip(support_feats, support_masks_down):
        fg_proto = cmrs.compute_prototype(sup_feat, sup_mask)
        fg_prototypes.append(fg_proto)
        
        if hasattr(cmrs, 'compute_bg_prototype'):
            bg_proto = cmrs.compute_bg_prototype(sup_feat, sup_mask)
        else:
            bg_mask = 1.0 - sup_mask
            bg_sum = bg_mask.sum() + 1e-8
            bg_proto = (sup_feat * bg_mask.unsqueeze(0)).sum(dim=(1, 2)) / bg_sum
        bg_prototypes.append(bg_proto)
    
    visualizer.vis_prototype(fg_prototypes, bg_prototypes, support_masks_down)
    
    # ==================== Step 3: Similarity Map ====================
    print("\n🎨 Step 3: Similarity Map计算...")
    
    C, H, W = query_feat.shape
    avg_fg = torch.stack(fg_prototypes).mean(dim=0)
    avg_bg = torch.stack(bg_prototypes).mean(dim=0)
    
    avg_fg_norm = F.normalize(avg_fg, p=2, dim=0)
    avg_bg_norm = F.normalize(avg_bg, p=2, dim=0)
    
    query_flat = query_feat.reshape(C, -1).T
    query_flat = F.normalize(query_flat, p=2, dim=1)
    
    fg_sim = torch.mv(query_flat, avg_fg_norm).reshape(H, W)
    bg_sim = torch.mv(query_flat, avg_bg_norm).reshape(H, W)
    contrast_sim = fg_sim - bg_sim
    
    visualizer.vis_similarity(fg_sim, bg_sim, contrast_sim, query_img)
    
    # ==================== Step 4: Point Prompt采样 ====================
    print("\n🎨 Step 4: Point Prompt采样...")
    
    point_coords, point_labels = cmrs.get_prompts_from_similarity(fg_sim, top_k=10, neg_k=5)
    
    visualizer.vis_points(fg_sim, point_coords, point_labels, query_img)
    
    # ==================== Step 5: SAM2预测 ====================
    print("\n🎨 Step 5: SAM2 Mask生成...")
    
    # 坐标缩放到原图尺度
    img_h, img_w = np.array(query_img).shape[:2]
    scale_x, scale_y = img_w / W, img_h / H
    point_coords_img = point_coords.copy().astype(np.float32)
    point_coords_img[:, 0] *= scale_x
    point_coords_img[:, 1] *= scale_y
    point_coords_img = point_coords_img.astype(np.int32)
    
    sam2_predictor.set_image(np.array(query_img))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=point_coords_img,
        point_labels=point_labels,
        multimask_output=True
    )
    
    visualizer.vis_sam_output(masks, scores, query_img, query_mask)
    
    # ==================== Step 6: 最终结果 ====================
    print("\n🎨 Step 6: 最终结果对比...")
    
    rough_mask = masks[np.argmax(scores)]
    visualizer.vis_final_comparison(query_img, query_mask, rough_mask, final_mask=None)
    
    print(f"\n✅ 所有可视化已保存到: {output_dir}/")
    print(f"   共生成 {visualizer.step} 张图像")


def main():
    parser = argparse.ArgumentParser(description='SP-SAM Pipeline可视化')
    parser.add_argument('--data_root', type=str, required=True, help='数据集路径')
    parser.add_argument('--k_shot', type=int, default=3, help='Support数量')
    parser.add_argument('--sample_idx', type=int, default=0, help='Query样本索引')
    parser.add_argument('--output_dir', type=str, default='vis_pipeline', help='输出目录')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16', help='DINO模型')
    parser.add_argument('--sam2_model', type=str, default='large', help='SAM2模型')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    run_pipeline_visualization(
        data_root=args.data_root,
        k_shot=args.k_shot,
        sample_idx=args.sample_idx,
        output_dir=args.output_dir,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model,
        device=args.device
    )


if __name__ == '__main__':
    main()

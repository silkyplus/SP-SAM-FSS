"""
SP-SAM 分阶段特征可视化
========================

分为两个阶段：
1. CMRS阶段：DINO特征 → Prototype → Similarity → Points → Rough Mask
2. Memory阶段：Support编码 → Memory Attention → 精炼Mask

重点分析两个阶段之间的影响关系

使用方法：
    python visualize_stages.py --data_root ISIC2018_256 --k_shot 3
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

try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except:
    HAS_PCA = False


class StageVisualizer:
    """分阶段可视化器"""
    
    def __init__(self, output_dir='vis_stages'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save(self, fig, name):
        path = self.output_dir / f'{name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   💾 {path}")
    
    # ================================================================
    # CMRS 阶段可视化
    # ================================================================
    
    def vis_cmrs_stage(self, 
                       query_img, query_mask,
                       support_imgs, support_masks,
                       query_feat, support_feats, support_masks_down,
                       fg_prototypes, bg_prototypes,
                       fg_sim, bg_sim, contrast_sim,
                       point_coords, point_labels,
                       rough_mask, all_sam_masks=None, sam_scores=None):
        """
        CMRS阶段完整可视化
        
        生成多张图：
        - cmrs_1_features.png: DINO特征提取
        - cmrs_2_prototype.png: Prototype计算与激活
        - cmrs_3_similarity.png: 相似度图
        - cmrs_4_points.png: Point采样
        - cmrs_5_rough_mask.png: SAM2输出
        - cmrs_summary.png: 阶段总结
        """
        print("\n" + "="*50)
        print("📌 CMRS Stage Visualization")
        print("="*50)
        
        # 1. DINO特征
        self._vis_cmrs_features(query_img, query_feat, support_imgs, support_feats)
        
        # 2. Prototype
        self._vis_cmrs_prototype(query_feat, support_feats, support_masks_down,
                                 fg_prototypes, bg_prototypes, support_imgs)
        
        # 3. Similarity
        self._vis_cmrs_similarity(query_img, fg_sim, bg_sim, contrast_sim, query_mask)
        
        # 4. Points
        self._vis_cmrs_points(query_img, contrast_sim, point_coords, point_labels, query_mask)
        
        # 5. Rough Mask
        self._vis_cmrs_rough_mask(query_img, query_mask, rough_mask, 
                                  all_sam_masks, sam_scores)
        
        # 6. Summary
        self._vis_cmrs_summary(query_img, query_mask, query_feat,
                              fg_sim, contrast_sim, point_coords, point_labels, rough_mask)
    
    def _vis_cmrs_features(self, query_img, query_feat, support_imgs, support_feats):
        """CMRS-1: DINO特征可视化"""
        k = len(support_feats)
        fig, axes = plt.subplots(2, k + 1, figsize=(4*(k+1), 8))
        
        C, H, W = query_feat.shape
        
        # Query
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        query_norm = torch.norm(query_feat, dim=0).cpu().numpy()
        im = axes[1, 0].imshow(query_norm, cmap='viridis')
        axes[1, 0].set_title(f'Query Feature\n[{C}×{H}×{W}]')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # Support
        for i in range(k):
            axes[0, i+1].imshow(support_imgs[i])
            axes[0, i+1].set_title(f'Support {i+1}')
            axes[0, i+1].axis('off')
            
            sup_norm = torch.norm(support_feats[i], dim=0).cpu().numpy()
            im = axes[1, i+1].imshow(sup_norm, cmap='viridis')
            axes[1, i+1].set_title(f'Support {i+1} Feature')
            axes[1, i+1].axis('off')
            plt.colorbar(im, ax=axes[1, i+1], fraction=0.046)
        
        fig.suptitle('CMRS Stage 1: DINO Feature Extraction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'cmrs_1_features')
    
    def _vis_cmrs_prototype(self, query_feat, support_feats, support_masks_down,
                           fg_prototypes, bg_prototypes, support_imgs):
        """CMRS-2: Prototype计算与各Support激活图"""
        k = len(fg_prototypes)
        fig, axes = plt.subplots(3, k + 2, figsize=(4*(k+2), 12))
        
        C, H, W = query_feat.shape
        query_flat = query_feat.reshape(C, -1).T
        query_flat_norm = F.normalize(query_flat, p=2, dim=1)
        
        # Row 1: 各Support的Prototype激活Query
        axes[0, 0].set_title('Support → Query\nActivation', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        activation_maps = []
        for i, fg_proto in enumerate(fg_prototypes):
            proto_norm = F.normalize(fg_proto, p=2, dim=0)
            activation = torch.mv(query_flat_norm, proto_norm).reshape(H, W)
            activation_maps.append(activation)
            
            im = axes[0, i+1].imshow(activation.cpu().numpy(), cmap='hot')
            axes[0, i+1].set_title(f'S{i+1} Proto\n→ Query')
            axes[0, i+1].axis('off')
        
        # 平均激活
        avg_activation = torch.stack(activation_maps).mean(dim=0)
        im = axes[0, k+1].imshow(avg_activation.cpu().numpy(), cmap='hot')
        axes[0, k+1].set_title('Average\nActivation')
        axes[0, k+1].axis('off')
        plt.colorbar(im, ax=axes[0, k+1], fraction=0.046)
        
        # Row 2: 激活一致性
        axes[1, 0].set_title('Consistency\nAnalysis', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        if k > 1:
            # 激活图标准差（不一致区域）
            std_activation = torch.stack(activation_maps).std(dim=0)
            im = axes[1, 1].imshow(std_activation.cpu().numpy(), cmap='Blues')
            axes[1, 1].set_title('Std (Inconsistent)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
            
            # 相关性矩阵
            corr = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    a1 = activation_maps[i].flatten().cpu().numpy()
                    a2 = activation_maps[j].flatten().cpu().numpy()
                    corr[i, j] = np.corrcoef(a1, a2)[0, 1]
            
            im = axes[1, 2].imshow(corr, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1, 2].set_xticks(range(k))
            axes[1, 2].set_yticks(range(k))
            axes[1, 2].set_xticklabels([f'S{i+1}' for i in range(k)])
            axes[1, 2].set_yticklabels([f'S{i+1}' for i in range(k)])
            axes[1, 2].set_title(f'Correlation\nMean={corr.mean():.2f}')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        # FG-BG分离度
        fg_bg_sims = []
        for fg, bg in zip(fg_prototypes, bg_prototypes):
            sim = F.cosine_similarity(fg.unsqueeze(0), bg.unsqueeze(0)).item()
            fg_bg_sims.append(sim)
        
        axes[1, 3].bar(range(k), fg_bg_sims, color='coral')
        axes[1, 3].set_xticks(range(k))
        axes[1, 3].set_xticklabels([f'S{i+1}' for i in range(k)])
        axes[1, 3].set_ylim([0, 1])
        axes[1, 3].set_title(f'FG-BG Sim\n(Lower=Better)')
        axes[1, 3].axhline(0.5, ls='--', c='gray')
        
        for j in range(4, k+2):
            axes[1, j].axis('off')
        
        # Row 3: Prototype来源（Support + Mask）
        axes[2, 0].set_title('Prototype\nSource', fontsize=10, fontweight='bold')
        axes[2, 0].axis('off')
        
        for i in range(k):
            mask = support_masks_down[i].cpu().numpy()
            mask_resized = cv2.resize(mask, (support_imgs[i].size[0], support_imgs[i].size[1]))
            
            axes[2, i+1].imshow(support_imgs[i])
            axes[2, i+1].imshow(mask_resized, alpha=0.5, cmap='Greens')
            fg_ratio = mask.sum() / mask.size * 100
            axes[2, i+1].set_title(f'S{i+1} FG={fg_ratio:.1f}%')
            axes[2, i+1].axis('off')
        
        axes[2, k+1].axis('off')
        
        fig.suptitle('CMRS Stage 2: Prototype Computation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'cmrs_2_prototype')
    
    def _vis_cmrs_similarity(self, query_img, fg_sim, bg_sim, contrast_sim, query_mask):
        """CMRS-3: Similarity Map"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        fg_np = fg_sim.cpu().numpy()
        bg_np = bg_sim.cpu().numpy()
        contrast_np = contrast_sim.cpu().numpy()
        
        # Row 1: 各相似度图
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(fg_np, cmap='hot')
        axes[0, 1].set_title(f'FG Similarity\n[{fg_np.min():.2f}, {fg_np.max():.2f}]')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        im2 = axes[0, 2].imshow(bg_np, cmap='hot')
        axes[0, 2].set_title(f'BG Similarity\n[{bg_np.min():.2f}, {bg_np.max():.2f}]')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        vmax = max(abs(contrast_np.min()), abs(contrast_np.max()))
        im3 = axes[0, 3].imshow(contrast_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, 3].set_title(f'Contrast (FG-BG)\nMean={contrast_np.mean():.3f}')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
        
        # Row 2: 与GT对比
        axes[1, 0].imshow(query_mask, cmap='gray')
        axes[1, 0].set_title('GT Mask')
        axes[1, 0].axis('off')
        
        # 阈值化
        thresh = contrast_np.mean() + 0.5 * contrast_np.std()
        binary = (contrast_np > thresh).astype(float)
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title(f'Thresholded\n(>{thresh:.2f})')
        axes[1, 1].axis('off')
        
        # GT区域内的相似度统计
        gt_resized = cv2.resize(query_mask.astype(float), (contrast_np.shape[1], contrast_np.shape[0]))
        gt_binary = gt_resized > 0.5
        
        fg_in_gt = contrast_np[gt_binary].mean() if gt_binary.sum() > 0 else 0
        fg_out_gt = contrast_np[~gt_binary].mean() if (~gt_binary).sum() > 0 else 0
        
        axes[1, 2].bar(['In GT', 'Out GT'], [fg_in_gt, fg_out_gt], 
                      color=['green', 'red'], edgecolor='black')
        axes[1, 2].set_ylabel('Mean Contrast Similarity')
        axes[1, 2].set_title(f'Similarity in/out GT\nGap={fg_in_gt-fg_out_gt:.3f}')
        axes[1, 2].axhline(0, ls='--', c='gray')
        
        # 分布
        axes[1, 3].hist(contrast_np[gt_binary].flatten(), bins=30, alpha=0.6, 
                       label='In GT', color='green')
        axes[1, 3].hist(contrast_np[~gt_binary].flatten(), bins=30, alpha=0.6, 
                       label='Out GT', color='red')
        axes[1, 3].set_xlabel('Contrast Similarity')
        axes[1, 3].set_title('Distribution')
        axes[1, 3].legend()
        
        fig.suptitle('CMRS Stage 3: Similarity Map', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'cmrs_3_similarity')
    
    def _vis_cmrs_points(self, query_img, similarity_map, point_coords, point_labels, query_mask):
        """CMRS-4: Point Prompt采样"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        sim_np = similarity_map.cpu().numpy()
        H, W = sim_np.shape
        img_h, img_w = np.array(query_img).shape[:2]
        scale_x, scale_y = img_w / W, img_h / H
        
        # 相似度图 + 点（特征尺度）
        axes[0].imshow(sim_np, cmap='RdBu_r')
        for (x, y), label in zip(point_coords, point_labels):
            color = 'lime' if label == 1 else 'red'
            marker = '*' if label == 1 else 'x'
            axes[0].scatter(x, y, c=color, marker=marker, s=200, edgecolors='white', linewidths=2)
        axes[0].set_title(f'Points on Similarity\nPos={sum(point_labels)}, Neg={len(point_labels)-sum(point_labels)}')
        axes[0].axis('off')
        
        # 原图 + 点
        axes[1].imshow(query_img)
        for (x, y), label in zip(point_coords, point_labels):
            px, py = x * scale_x, y * scale_y
            color = 'lime' if label == 1 else 'red'
            marker = '*' if label == 1 else 'x'
            axes[1].scatter(px, py, c=color, marker=marker, s=300, edgecolors='white', linewidths=2)
        axes[1].set_title('Points on Image')
        axes[1].axis('off')
        
        # GT + 点
        axes[2].imshow(query_mask, cmap='gray')
        gt_resized = cv2.resize(query_mask.astype(float), (W, H))
        
        correct_pos = 0
        correct_neg = 0
        for (x, y), label in zip(point_coords, point_labels):
            in_gt = gt_resized[int(y), int(x)] > 0.5
            if label == 1 and in_gt:
                correct_pos += 1
            elif label == 0 and not in_gt:
                correct_neg += 1
            
            px, py = x * scale_x, y * scale_y
            color = 'lime' if label == 1 else 'red'
            marker = '*' if label == 1 else 'x'
            axes[2].scatter(px, py, c=color, marker=marker, s=300, edgecolors='white', linewidths=2)
        
        n_pos = sum(point_labels)
        n_neg = len(point_labels) - n_pos
        axes[2].set_title(f'Points vs GT\nPos Acc: {correct_pos}/{n_pos}, Neg Acc: {correct_neg}/{n_neg}')
        axes[2].axis('off')
        
        # 采样详情
        pos_sims = [sim_np[int(y), int(x)] for (x, y), l in zip(point_coords, point_labels) if l == 1]
        neg_sims = [sim_np[int(y), int(x)] for (x, y), l in zip(point_coords, point_labels) if l == 0]
        
        axes[3].boxplot([pos_sims, neg_sims], labels=['Positive', 'Negative'])
        axes[3].set_ylabel('Similarity at Point')
        axes[3].set_title(f'Point Similarity\nPos: {np.mean(pos_sims):.3f}, Neg: {np.mean(neg_sims):.3f}')
        axes[3].axhline(0, ls='--', c='gray')
        
        fig.suptitle('CMRS Stage 4: Point Prompt Sampling', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'cmrs_4_points')
    
    def _vis_cmrs_rough_mask(self, query_img, query_mask, rough_mask, all_masks=None, scores=None):
        """CMRS-5: SAM2输出的Rough Mask"""
        n_masks = len(all_masks) if all_masks is not None else 1
        fig, axes = plt.subplots(2, max(4, n_masks + 1), figsize=(4*max(4, n_masks+1), 8))
        
        # Row 1: 各个mask
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        if all_masks is not None and scores is not None:
            for i, (mask, score) in enumerate(zip(all_masks, scores)):
                axes[0, i+1].imshow(mask, cmap='gray')
                iou = self._iou(mask, query_mask)
                axes[0, i+1].set_title(f'Mask {i+1}\nScore={score:.2f}, IoU={iou*100:.1f}%')
                axes[0, i+1].axis('off')
        
        for j in range(n_masks + 1, axes.shape[1]):
            axes[0, j].axis('off')
        
        # Row 2: Best mask对比
        axes[1, 0].imshow(query_mask, cmap='gray')
        axes[1, 0].set_title('GT Mask')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(rough_mask, cmap='gray')
        iou = self._iou(rough_mask, query_mask)
        dice = self._dice(rough_mask, query_mask)
        axes[1, 1].set_title(f'Rough Mask\nIoU={iou*100:.1f}%, Dice={dice*100:.1f}%')
        axes[1, 1].axis('off')
        
        # 对比图
        comp = np.zeros((*query_mask.shape, 3), dtype=np.uint8)
        comp[query_mask > 0, 1] = 255
        comp[rough_mask > 0, 0] = 255
        comp[(query_mask > 0) & (rough_mask > 0)] = [255, 255, 0]
        axes[1, 2].imshow(comp)
        axes[1, 2].set_title('GT(G) vs Pred(R)\nOverlap(Y)')
        axes[1, 2].axis('off')
        
        # Overlay
        overlay = np.array(query_img).copy()
        overlay[rough_mask > 0, 0] = np.clip(overlay[rough_mask > 0, 0] * 0.5 + 255 * 0.5, 0, 255)
        axes[1, 3].imshow(overlay.astype(np.uint8))
        axes[1, 3].set_title('Rough Mask Overlay')
        axes[1, 3].axis('off')
        
        for j in range(4, axes.shape[1]):
            axes[1, j].axis('off')
        
        fig.suptitle('CMRS Stage 5: SAM2 Rough Mask', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'cmrs_5_rough_mask')
    
    def _vis_cmrs_summary(self, query_img, query_mask, query_feat,
                         fg_sim, contrast_sim, point_coords, point_labels, rough_mask):
        """CMRS阶段总结"""
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 6, figure=fig)
        
        # Query
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(query_img)
        ax1.set_title('1. Query')
        ax1.axis('off')
        
        # Feature
        ax2 = fig.add_subplot(gs[0, 1])
        feat_norm = torch.norm(query_feat, dim=0).cpu().numpy()
        ax2.imshow(feat_norm, cmap='viridis')
        ax2.set_title('2. DINO Feature')
        ax2.axis('off')
        
        # Similarity
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(contrast_sim.cpu().numpy(), cmap='RdBu_r')
        ax3.set_title('3. Similarity')
        ax3.axis('off')
        
        # Points
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(fg_sim.cpu().numpy(), cmap='hot')
        for (x, y), l in zip(point_coords, point_labels):
            c = 'lime' if l == 1 else 'red'
            ax4.scatter(x, y, c=c, s=100, edgecolors='white')
        ax4.set_title('4. Points')
        ax4.axis('off')
        
        # Rough Mask
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(rough_mask, cmap='gray')
        ax5.set_title('5. Rough Mask')
        ax5.axis('off')
        
        # Metrics
        ax6 = fig.add_subplot(gs[0, 5])
        iou = self._iou(rough_mask, query_mask)
        dice = self._dice(rough_mask, query_mask)
        ax6.text(0.5, 0.6, f'IoU: {iou*100:.1f}%', fontsize=20, ha='center', fontweight='bold')
        ax6.text(0.5, 0.4, f'Dice: {dice*100:.1f}%', fontsize=20, ha='center', fontweight='bold')
        ax6.set_title('CMRS Result')
        ax6.axis('off')
        
        fig.suptitle('CMRS Stage Summary: Query → Feature → Similarity → Points → Rough Mask', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'cmrs_summary')
    
    # ================================================================
    # Memory 阶段可视化
    # ================================================================
    
    def vis_memory_stage(self,
                         query_img, query_mask,
                         support_imgs, support_masks,
                         rough_mask,
                         memory_features,  # 编码后的memory特征
                         query_feat_before,  # Memory Attention前的特征
                         query_feat_after,   # Memory Attention后的特征
                         final_mask,
                         mask_logits=None):
        """
        Memory阶段完整可视化
        
        生成：
        - memory_1_encoding.png: Support编码到Memory
        - memory_2_attention.png: Memory Attention前后对比
        - memory_3_refinement.png: Rough→Final对比
        - memory_summary.png: 阶段总结
        """
        print("\n" + "="*50)
        print("📌 Memory Stage Visualization")
        print("="*50)
        
        # 1. Support编码
        self._vis_memory_encoding(support_imgs, support_masks, memory_features)
        
        # 2. Memory Attention
        self._vis_memory_attention(query_img, query_feat_before, query_feat_after, query_mask)
        
        # 3. Mask精炼
        self._vis_memory_refinement(query_img, query_mask, rough_mask, final_mask, mask_logits)
        
        # 4. Summary
        self._vis_memory_summary(query_img, query_mask, rough_mask, final_mask,
                                query_feat_before, query_feat_after)
    
    def _vis_memory_encoding(self, support_imgs, support_masks, memory_features):
        """Memory-1: Support编码到Memory"""
        k = len(support_imgs)
        fig, axes = plt.subplots(3, k, figsize=(4*k, 12))
        
        if k == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(k):
            # Support图像
            axes[0, i].imshow(support_imgs[i])
            axes[0, i].set_title(f'Support {i+1}')
            axes[0, i].axis('off')
            
            # Support Mask
            axes[1, i].imshow(support_masks[i], cmap='gray')
            fg_ratio = (support_masks[i] > 0).sum() / support_masks[i].size * 100
            axes[1, i].set_title(f'Mask (FG={fg_ratio:.1f}%)')
            axes[1, i].axis('off')
            
            # Memory特征
            if memory_features is not None and i < len(memory_features):
                mem_feat = memory_features[i]
                if len(mem_feat.shape) == 3:
                    mem_norm = torch.norm(mem_feat, dim=0).cpu().numpy()
                else:
                    mem_norm = mem_feat.cpu().numpy()
                    if len(mem_norm.shape) == 1:
                        # 如果是1D，reshape成2D
                        side = int(np.sqrt(len(mem_norm)))
                        if side * side == len(mem_norm):
                            mem_norm = mem_norm.reshape(side, side)
                        else:
                            mem_norm = mem_norm[:side*side].reshape(side, side)
                
                im = axes[2, i].imshow(mem_norm, cmap='plasma')
                axes[2, i].set_title(f'Memory Feature {i+1}')
                axes[2, i].axis('off')
                plt.colorbar(im, ax=axes[2, i], fraction=0.046)
            else:
                axes[2, i].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[2, i].axis('off')
        
        fig.suptitle('Memory Stage 1: Support Encoding', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'memory_1_encoding')
    
    def _vis_memory_attention(self, query_img, feat_before, feat_after, query_mask):
        """Memory-2: Memory Attention前后特征对比"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Row 1: 特征范数
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        if feat_before is not None:
            if len(feat_before.shape) == 3:
                before_norm = torch.norm(feat_before, dim=0).cpu().numpy()
            else:
                before_norm = feat_before.cpu().numpy()
            im1 = axes[0, 1].imshow(before_norm, cmap='viridis')
            axes[0, 1].set_title('Before Memory Attention')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        else:
            axes[0, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[0, 1].axis('off')
        
        if feat_after is not None:
            if len(feat_after.shape) == 3:
                after_norm = torch.norm(feat_after, dim=0).cpu().numpy()
            else:
                after_norm = feat_after.cpu().numpy()
            im2 = axes[0, 2].imshow(after_norm, cmap='viridis')
            axes[0, 2].set_title('After Memory Attention')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        else:
            axes[0, 2].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[0, 2].axis('off')
        
        # 差异图
        if feat_before is not None and feat_after is not None:
            diff = after_norm - before_norm
            vmax = max(abs(diff.min()), abs(diff.max()))
            im3 = axes[0, 3].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[0, 3].set_title(f'Difference\n(After - Before)')
            axes[0, 3].axis('off')
            plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
        else:
            axes[0, 3].axis('off')
        
        # Row 2: 与GT的关系
        axes[1, 0].imshow(query_mask, cmap='gray')
        axes[1, 0].set_title('GT Mask')
        axes[1, 0].axis('off')
        
        if feat_before is not None and feat_after is not None:
            # GT区域内外的特征变化
            H, W = before_norm.shape
            gt_resized = cv2.resize(query_mask.astype(float), (W, H)) > 0.5
            
            before_in = before_norm[gt_resized].mean()
            before_out = before_norm[~gt_resized].mean()
            after_in = after_norm[gt_resized].mean()
            after_out = after_norm[~gt_resized].mean()
            
            x = np.arange(2)
            width = 0.35
            axes[1, 1].bar(x - width/2, [before_in, before_out], width, label='Before', color='blue', alpha=0.7)
            axes[1, 1].bar(x + width/2, [after_in, after_out], width, label='After', color='orange', alpha=0.7)
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(['In GT', 'Out GT'])
            axes[1, 1].set_ylabel('Mean Feature Norm')
            axes[1, 1].set_title('Feature Change by Region')
            axes[1, 1].legend()
            
            # 变化量
            change_in = after_in - before_in
            change_out = after_out - before_out
            axes[1, 2].bar(['In GT', 'Out GT'], [change_in, change_out], 
                          color=['green', 'red'], edgecolor='black')
            axes[1, 2].axhline(0, ls='--', c='gray')
            axes[1, 2].set_ylabel('Change (After - Before)')
            axes[1, 2].set_title(f'Memory Effect\nIn GT: {change_in:+.3f}, Out: {change_out:+.3f}')
        else:
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')
        
        axes[1, 3].axis('off')
        
        fig.suptitle('Memory Stage 2: Memory Attention Effect', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'memory_2_attention')
    
    def _vis_memory_refinement(self, query_img, query_mask, rough_mask, final_mask, logits=None):
        """Memory-3: Rough→Final Mask精炼"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Row 1: 各Mask
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(query_mask, cmap='gray')
        axes[0, 1].set_title('GT Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(rough_mask, cmap='gray')
        rough_iou = self._iou(rough_mask, query_mask)
        axes[0, 2].set_title(f'Rough Mask\nIoU={rough_iou*100:.1f}%')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(final_mask, cmap='gray')
        final_iou = self._iou(final_mask, query_mask)
        axes[0, 3].set_title(f'Final Mask\nIoU={final_iou*100:.1f}%')
        axes[0, 3].axis('off')
        
        # Row 2: 对比分析
        # Rough vs GT
        comp1 = np.zeros((*query_mask.shape, 3), dtype=np.uint8)
        comp1[query_mask > 0, 1] = 255
        comp1[rough_mask > 0, 0] = 255
        comp1[(query_mask > 0) & (rough_mask > 0)] = [255, 255, 0]
        axes[1, 0].imshow(comp1)
        axes[1, 0].set_title('Rough vs GT')
        axes[1, 0].axis('off')
        
        # Final vs GT
        comp2 = np.zeros((*query_mask.shape, 3), dtype=np.uint8)
        comp2[query_mask > 0, 1] = 255
        comp2[final_mask > 0, 0] = 255
        comp2[(query_mask > 0) & (final_mask > 0)] = [255, 255, 0]
        axes[1, 1].imshow(comp2)
        axes[1, 1].set_title('Final vs GT')
        axes[1, 1].axis('off')
        
        # Rough vs Final
        comp3 = np.zeros((*query_mask.shape, 3), dtype=np.uint8)
        comp3[rough_mask > 0, 0] = 255
        comp3[final_mask > 0, 2] = 255
        comp3[(rough_mask > 0) & (final_mask > 0)] = [255, 0, 255]
        axes[1, 2].imshow(comp3)
        axes[1, 2].set_title('Rough(R) vs Final(B)')
        axes[1, 2].axis('off')
        
        # 指标对比
        rough_dice = self._dice(rough_mask, query_mask)
        final_dice = self._dice(final_mask, query_mask)
        
        x = np.arange(2)
        width = 0.35
        axes[1, 3].bar(x - width/2, [rough_iou*100, rough_dice*100], width, 
                      label='Rough', color='coral')
        axes[1, 3].bar(x + width/2, [final_iou*100, final_dice*100], width, 
                      label='Final', color='steelblue')
        axes[1, 3].set_xticks(x)
        axes[1, 3].set_xticklabels(['IoU', 'Dice'])
        axes[1, 3].set_ylabel('%')
        axes[1, 3].set_title(f'Improvement\nIoU: {(final_iou-rough_iou)*100:+.1f}%')
        axes[1, 3].legend()
        axes[1, 3].set_ylim([0, 100])
        
        fig.suptitle('Memory Stage 3: Mask Refinement', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'memory_3_refinement')
    
    def _vis_memory_summary(self, query_img, query_mask, rough_mask, final_mask,
                           feat_before, feat_after):
        """Memory阶段总结"""
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 6, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(rough_mask, cmap='gray')
        rough_iou = self._iou(rough_mask, query_mask)
        ax1.set_title(f'1. Rough Mask\nIoU={rough_iou*100:.1f}%')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if feat_before is not None and len(feat_before.shape) == 3:
            ax2.imshow(torch.norm(feat_before, dim=0).cpu().numpy(), cmap='viridis')
        ax2.set_title('2. Before Memory')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if feat_after is not None and len(feat_after.shape) == 3:
            ax3.imshow(torch.norm(feat_after, dim=0).cpu().numpy(), cmap='viridis')
        ax3.set_title('3. After Memory')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(final_mask, cmap='gray')
        final_iou = self._iou(final_mask, query_mask)
        ax4.set_title(f'4. Final Mask\nIoU={final_iou*100:.1f}%')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(query_mask, cmap='gray')
        ax5.set_title('GT Mask')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[0, 5])
        improvement = (final_iou - rough_iou) * 100
        color = 'green' if improvement > 0 else 'red'
        ax6.text(0.5, 0.6, f'Δ IoU: {improvement:+.1f}%', fontsize=20, 
                ha='center', fontweight='bold', color=color)
        ax6.text(0.5, 0.4, f'Final: {final_iou*100:.1f}%', fontsize=16, ha='center')
        ax6.set_title('Memory Effect')
        ax6.axis('off')
        
        fig.suptitle('Memory Stage Summary: Rough → Memory Attention → Final', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'memory_summary')
    
    # ================================================================
    # 两阶段对比
    # ================================================================
    
    def vis_stage_comparison(self, query_img, query_mask, rough_mask, final_mask):
        """两阶段最终对比"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        rough_iou = self._iou(rough_mask, query_mask)
        final_iou = self._iou(final_mask, query_mask)
        rough_dice = self._dice(rough_mask, query_mask)
        final_dice = self._dice(final_mask, query_mask)
        
        # Row 1
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(query_mask, cmap='gray')
        axes[0, 1].set_title('GT')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(rough_mask, cmap='gray')
        axes[0, 2].set_title(f'CMRS Output\nIoU={rough_iou*100:.1f}%')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(final_mask, cmap='gray')
        axes[0, 3].set_title(f'Memory Output\nIoU={final_iou*100:.1f}%')
        axes[0, 3].axis('off')
        
        # Row 2
        # Overlay
        overlay1 = np.array(query_img).copy()
        overlay1[rough_mask > 0, 0] = np.clip(overlay1[rough_mask > 0, 0] * 0.5 + 255 * 0.5, 0, 255)
        axes[1, 0].imshow(overlay1.astype(np.uint8))
        axes[1, 0].set_title('CMRS Overlay')
        axes[1, 0].axis('off')
        
        overlay2 = np.array(query_img).copy()
        overlay2[final_mask > 0, 2] = np.clip(overlay2[final_mask > 0, 2] * 0.5 + 255 * 0.5, 0, 255)
        axes[1, 1].imshow(overlay2.astype(np.uint8))
        axes[1, 1].set_title('Memory Overlay')
        axes[1, 1].axis('off')
        
        # 指标
        metrics = ['IoU', 'Dice']
        cmrs_vals = [rough_iou * 100, rough_dice * 100]
        mem_vals = [final_iou * 100, final_dice * 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 2].bar(x - width/2, cmrs_vals, width, label='CMRS', color='coral')
        axes[1, 2].bar(x + width/2, mem_vals, width, label='Memory', color='steelblue')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].set_ylabel('%')
        axes[1, 2].legend()
        axes[1, 2].set_ylim([0, 100])
        axes[1, 2].set_title('Metrics Comparison')
        
        # 总结
        improvement = final_iou - rough_iou
        axes[1, 3].text(0.5, 0.7, 'Stage Comparison', fontsize=14, ha='center', fontweight='bold')
        axes[1, 3].text(0.5, 0.5, f'CMRS: {rough_iou*100:.1f}%', fontsize=12, ha='center')
        axes[1, 3].text(0.5, 0.35, f'Memory: {final_iou*100:.1f}%', fontsize=12, ha='center')
        color = 'green' if improvement > 0 else 'red'
        axes[1, 3].text(0.5, 0.15, f'Δ: {improvement*100:+.1f}%', fontsize=16, 
                       ha='center', fontweight='bold', color=color)
        axes[1, 3].axis('off')
        
        fig.suptitle('CMRS vs Memory Stage Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'stage_comparison')
    
    # ================================================================
    # 工具函数
    # ================================================================
    
    def _iou(self, pred, gt):
        inter = ((pred > 0) & (gt > 0)).sum()
        union = ((pred > 0) | (gt > 0)).sum()
        return inter / (union + 1e-8)
    
    def _dice(self, pred, gt):
        inter = ((pred > 0) & (gt > 0)).sum()
        return 2 * inter / ((pred > 0).sum() + (gt > 0).sum() + 1e-8)


def main():
    """主函数：运行分阶段可视化"""
    parser = argparse.ArgumentParser(description='SP-SAM分阶段可视化')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--k_shot', type=int, default=3)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='vis_stages')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16')
    parser.add_argument('--sam2_model', type=str, default='large')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SP-SAM Stage-by-Stage Visualization")
    print("="*60)
    
    # 加载模型
    print("\n🚀 加载模型...")
    from sp_sam_complete import SPSAMModel, CMRSModule
    from src.model_manager import ModelManager
    
    model_manager = ModelManager(device=args.device)
    
    if 'dinov3' in args.dino_model:
        dino, dino_transform = model_manager.load_dinov3_model(dinov3_model_name=args.dino_model)
    else:
        dino, dino_transform = model_manager.load_dinov2_model(dinov2_model_name=args.dino_model)
    
    sam2, sam2_predictor, _ = model_manager.load_sam2_model(sam2_model_type=args.sam2_model)
    
    cmrs = CMRSModule(dino, dino_transform, args.device)
    sp_sam = SPSAMModel(sam2, sam2_predictor, dino, dino_transform, args.device, args.sam2_model)
    
    # 加载数据
    print("\n📁 加载数据...")
    from isic2018_dataset import ISIC2018Dataset
    dataset = ISIC2018Dataset(args.data_root, support_ratio=0.2)
    episode = dataset.get_episode(k_shot=args.k_shot)
    
    support_imgs = [s['img'] for s in episode['support']]
    support_masks = [s['mask'] for s in episode['support']]
    query = episode['query'][args.sample_idx]
    query_img, query_mask = query['img'], query['mask']
    
    print(f"   Query: {query['sample_id']}")
    
    # 创建可视化器
    visualizer = StageVisualizer(args.output_dir)
    
    # ================================================================
    # CMRS阶段
    # ================================================================
    print("\n" + "="*60)
    print("🔷 CMRS Stage")
    print("="*60)
    
    # 提取特征
    query_feat = cmrs.get_features(query_img)
    support_feats = [cmrs.get_features(img) for img in support_imgs]
    support_masks_down = [cmrs.downsample_mask(m, (f.shape[1], f.shape[2])) 
                         for m, f in zip(support_masks, support_feats)]
    
    # 计算Prototype
    fg_protos = [cmrs.compute_prototype(f, m) for f, m in zip(support_feats, support_masks_down)]
    bg_protos = [cmrs.compute_bg_prototype(f, m) for f, m in zip(support_feats, support_masks_down)]
    
    # 计算Similarity
    C, H, W = query_feat.shape
    avg_fg = F.normalize(torch.stack(fg_protos).mean(0), p=2, dim=0)
    avg_bg = F.normalize(torch.stack(bg_protos).mean(0), p=2, dim=0)
    query_flat = F.normalize(query_feat.reshape(C, -1).T, p=2, dim=1)
    
    fg_sim = torch.mv(query_flat, avg_fg).reshape(H, W)
    bg_sim = torch.mv(query_flat, avg_bg).reshape(H, W)
    contrast_sim = fg_sim - bg_sim
    
    # Point采样
    point_coords, point_labels = cmrs.get_prompts_from_similarity(fg_sim, top_k=10, neg_k=5)
    
    # SAM2预测
    img_h, img_w = np.array(query_img).shape[:2]
    scale_x, scale_y = img_w / W, img_h / H
    point_coords_img = point_coords.astype(float)
    point_coords_img[:, 0] *= scale_x
    point_coords_img[:, 1] *= scale_y
    
    sam2_predictor.set_image(np.array(query_img))
    all_masks, scores, logits = sam2_predictor.predict(
        point_coords=point_coords_img.astype(np.int32),
        point_labels=point_labels,
        multimask_output=True
    )
    rough_mask = all_masks[np.argmax(scores)]
    
    # CMRS可视化
    visualizer.vis_cmrs_stage(
        query_img, query_mask,
        support_imgs, support_masks,
        query_feat, support_feats, support_masks_down,
        fg_protos, bg_protos,
        fg_sim, bg_sim, contrast_sim,
        point_coords, point_labels,
        rough_mask, all_masks, scores
    )
    
    # ================================================================
    # Memory阶段
    # ================================================================
    print("\n" + "="*60)
    print("🔷 Memory Stage")
    print("="*60)
    
    # 使用SP-SAM的Memory模块
    sp_sam.set_support(support_imgs, support_masks)
    
    # 获取Memory特征（如果可以访问）
    memory_features = None
    if hasattr(sp_sam.memory_module, 'prev_out'):
        memory_features = sp_sam.memory_module.prev_out.get('maskmem_features', None)
    
    # 获取Memory前后的特征（需要修改模型才能获取，这里简化处理）
    query_feat_before = query_feat  # 简化：用CMRS特征代替
    query_feat_after = query_feat   # 实际应该是Memory Attention后的特征
    
    # Memory精炼
    results = sp_sam.predict(
        query_img, support_imgs, support_masks,
        use_cmrs=True, use_memory_refinement=True
    )
    final_mask = results['final_mask']
    
    # Memory可视化
    visualizer.vis_memory_stage(
        query_img, query_mask,
        support_imgs, support_masks,
        rough_mask,
        memory_features,
        query_feat_before, query_feat_after,
        final_mask
    )
    
    # ================================================================
    # 两阶段对比
    # ================================================================
    print("\n" + "="*60)
    print("🔷 Stage Comparison")
    print("="*60)
    
    visualizer.vis_stage_comparison(query_img, query_mask, rough_mask, final_mask)
    
    print(f"\n✅ 所有可视化已保存到: {args.output_dir}/")


if __name__ == '__main__':
    main()

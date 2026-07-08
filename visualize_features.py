"""
SP-SAM 特征可视化工具
======================

可视化模型各阶段的特征图：
1. DINOv2/v3 原始特征图
2. Support Prototype（原型向量）
3. Similarity Map（相似度热力图）
4. Point Prompts（采样的正负点）
5. SAM2 中间特征
6. Rough Mask vs Final Mask

使用方法：
    python visualize_features.py --data_root ISIC2018_256 --sample_id ISIC_0000000 --output_dir vis_output
    
    # 或者可视化多个样本
    python visualize_features.py --data_root ISIC2018_256 --num_samples 5 --output_dir vis_output
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import cv2

# 尝试导入sklearn用于PCA
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn未安装，部分可视化功能受限")


class FeatureVisualizer:
    """SP-SAM特征可视化器"""
    
    def __init__(self, sp_sam_model, device='cuda'):
        """
        Args:
            sp_sam_model: SPSAMModel实例
            device: 计算设备
        """
        self.sp_sam = sp_sam_model
        self.device = device
        self.cmrs = sp_sam_model.cmrs
        
        # 存储中间特征
        self.cached_features = {}
    
    def extract_all_features(self, query_img: Image.Image,
                            support_imgs: List[Image.Image],
                            support_masks: List[np.ndarray]) -> Dict:
        """
        提取所有阶段的特征
        
        Returns:
            features_dict: 包含各阶段特征的字典
        """
        features = {}
        
        # 1. 提取Query的DINO特征（使用get_features，它会根据配置选择方法）
        print("   📊 提取Query DINO特征...")
        if hasattr(self.cmrs, 'get_features'):
            query_dino_feat = self.cmrs.get_features(query_img)
        else:
            query_dino_feat = self.cmrs.extract_dino_features(query_img)
        features['query_dino'] = query_dino_feat.cpu()
        
        # 2. 提取Support的DINO特征
        print("   📊 提取Support DINO特征...")
        support_dino_feats = []
        support_masks_downsampled = []
        
        for i, (sup_img, sup_mask) in enumerate(zip(support_imgs, support_masks)):
            if hasattr(self.cmrs, 'get_features'):
                sup_feat = self.cmrs.get_features(sup_img)
            else:
                sup_feat = self.cmrs.extract_dino_features(sup_img)
            support_dino_feats.append(sup_feat.cpu())
            
            # 下采样mask
            H_feat, W_feat = sup_feat.shape[1], sup_feat.shape[2]
            mask_down = self.cmrs.downsample_mask(sup_mask, (H_feat, W_feat))
            support_masks_downsampled.append(mask_down.cpu())
        
        features['support_dino'] = support_dino_feats
        features['support_masks_down'] = support_masks_downsampled
        
        # 3. 计算Prototype
        print("   📊 计算Prototype...")
        prototypes = []
        bg_prototypes = []
        
        for sup_feat, sup_mask in zip(support_dino_feats, support_masks_downsampled):
            sup_feat = sup_feat.to(self.device)
            sup_mask = sup_mask.to(self.device)
            
            # 前景原型
            fg_proto = self.cmrs.compute_prototype(sup_feat, sup_mask)
            prototypes.append(fg_proto.cpu())
            
            # 背景原型（使用CMRS的方法，如果存在的话）
            if hasattr(self.cmrs, 'compute_bg_prototype'):
                bg_proto = self.cmrs.compute_bg_prototype(sup_feat, sup_mask)
            else:
                # 手动计算
                bg_mask = 1.0 - sup_mask
                bg_sum = bg_mask.sum() + 1e-8
                bg_proto = (sup_feat * bg_mask.unsqueeze(0)).sum(dim=(1, 2)) / bg_sum
            bg_prototypes.append(bg_proto.cpu())
        
        features['fg_prototypes'] = prototypes
        features['bg_prototypes'] = bg_prototypes
        features['avg_fg_prototype'] = torch.stack(prototypes).mean(dim=0)
        features['avg_bg_prototype'] = torch.stack(bg_prototypes).mean(dim=0)
        
        # 4. 计算Similarity Map
        print("   📊 计算Similarity Map...")
        query_feat = query_dino_feat.to(self.device)
        C, H, W = query_feat.shape
        
        # 归一化prototype
        avg_fg_proto = F.normalize(features['avg_fg_prototype'].to(self.device), p=2, dim=0)
        avg_bg_proto = F.normalize(features['avg_bg_prototype'].to(self.device), p=2, dim=0)
        
        # Query特征归一化
        query_flat = query_feat.reshape(C, -1).T
        query_flat = F.normalize(query_flat, p=2, dim=1)
        
        # 前景相似度
        fg_similarity = torch.mv(query_flat, avg_fg_proto).reshape(H, W)
        features['fg_similarity'] = fg_similarity.cpu()
        
        # 背景相似度
        bg_similarity = torch.mv(query_flat, avg_bg_proto).reshape(H, W)
        features['bg_similarity'] = bg_similarity.cpu()
        
        # 对比相似度（fg - bg）
        contrast_similarity = fg_similarity - bg_similarity
        features['contrast_similarity'] = contrast_similarity.cpu()
        
        # 5. 采样Point Prompts
        print("   📊 采样Point Prompts...")
        point_coords, point_labels = self.cmrs.get_prompts_from_similarity(
            fg_similarity, top_k=10, neg_k=5
        )
        features['point_coords_feat'] = point_coords  # 特征图尺度
        features['point_labels'] = point_labels
        
        # 转换到原图尺度
        H_img, W_img = query_img.size[::-1]
        scale_y = H_img / H
        scale_x = W_img / W
        point_coords_img = point_coords.copy().astype(np.float32)
        point_coords_img[:, 0] *= scale_x
        point_coords_img[:, 1] *= scale_y
        features['point_coords_img'] = point_coords_img.astype(np.int32)
        
        # 6. 生成Rough Mask（使用SAM2）
        print("   📊 生成Rough Mask...")
        self.sp_sam.sam2_predictor.set_image(np.array(query_img))
        try:
            masks, scores, logits = self.sp_sam.sam2_predictor.predict(
                point_coords=features['point_coords_img'],
                point_labels=point_labels,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            features['rough_mask'] = masks[best_idx]
            features['all_masks'] = masks
            features['mask_scores'] = scores
            features['mask_logits'] = logits
        except Exception as e:
            print(f"      ⚠️ SAM2预测失败: {e}")
            features['rough_mask'] = None
        
        # 7. 存储原始输入
        features['query_img'] = query_img
        features['support_imgs'] = support_imgs
        features['support_masks'] = support_masks
        
        self.cached_features = features
        return features
    
    def visualize_dino_features(self, features: Dict, save_path: str = None,
                               method: str = 'pca'):
        """
        可视化DINO特征图
        
        Args:
            features: 特征字典
            save_path: 保存路径
            method: 'pca' 或 'channel' 或 'norm'
        """
        query_feat = features['query_dino']  # [C, H, W]
        C, H, W = query_feat.shape
        
        n_support = len(features['support_dino'])
        
        fig = plt.figure(figsize=(16, 4 * (1 + n_support)))
        gs = GridSpec(1 + n_support, 4, figure=fig)
        
        # Query特征可视化
        query_img = features['query_img']
        
        # 原图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(query_img)
        ax1.set_title('Query Image')
        ax1.axis('off')
        
        # 特征范数图
        ax2 = fig.add_subplot(gs[0, 1])
        feat_norm = torch.norm(query_feat, dim=0).numpy()
        im2 = ax2.imshow(feat_norm, cmap='viridis')
        ax2.set_title('Feature Norm')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # PCA降维可视化
        if HAS_SKLEARN and method == 'pca':
            ax3 = fig.add_subplot(gs[0, 2])
            feat_flat = query_feat.reshape(C, -1).T.numpy()  # [H*W, C]
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat)
            feat_pca = feat_pca.reshape(H, W, 3)
            # 归一化到0-1
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
            ax3.imshow(feat_pca)
            ax3.set_title(f'PCA (3 components)\nVar: {pca.explained_variance_ratio_.sum()*100:.1f}%')
            ax3.axis('off')
        else:
            ax3 = fig.add_subplot(gs[0, 2])
            # 显示前3个通道
            feat_rgb = query_feat[:3].permute(1, 2, 0).numpy()
            feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min() + 1e-8)
            ax3.imshow(feat_rgb)
            ax3.set_title('First 3 Channels')
            ax3.axis('off')
        
        # 通道统计
        ax4 = fig.add_subplot(gs[0, 3])
        channel_means = query_feat.mean(dim=(1, 2)).numpy()
        ax4.bar(range(min(50, C)), channel_means[:50], alpha=0.7)
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('Mean Activation')
        ax4.set_title(f'Channel Statistics (first 50/{C})')
        
        # Support特征可视化
        for i, (sup_feat, sup_img, sup_mask) in enumerate(zip(
            features['support_dino'], features['support_imgs'], features['support_masks']
        )):
            row = i + 1
            C_s, H_s, W_s = sup_feat.shape
            
            ax_s1 = fig.add_subplot(gs[row, 0])
            ax_s1.imshow(sup_img)
            ax_s1.set_title(f'Support {i+1} Image')
            ax_s1.axis('off')
            
            ax_s2 = fig.add_subplot(gs[row, 1])
            ax_s2.imshow(sup_mask, cmap='gray')
            ax_s2.set_title(f'Support {i+1} Mask')
            ax_s2.axis('off')
            
            ax_s3 = fig.add_subplot(gs[row, 2])
            sup_norm = torch.norm(sup_feat, dim=0).numpy()
            im_s3 = ax_s3.imshow(sup_norm, cmap='viridis')
            ax_s3.set_title(f'Support {i+1} Feature Norm')
            ax_s3.axis('off')
            plt.colorbar(im_s3, ax=ax_s3, fraction=0.046)
            
            # 前景区域特征
            ax_s4 = fig.add_subplot(gs[row, 3])
            mask_down = features['support_masks_down'][i].numpy()
            masked_norm = sup_norm * mask_down
            im_s4 = ax_s4.imshow(masked_norm, cmap='hot')
            ax_s4.set_title(f'Support {i+1} FG Features')
            ax_s4.axis('off')
            plt.colorbar(im_s4, ax=ax_s4, fraction=0.046)
        
        plt.suptitle('DINO Feature Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {save_path}")
        plt.close()
    
    def visualize_prototype(self, features: Dict, save_path: str = None):
        """
        可视化Prototype向量
        """
        fg_protos = features['fg_prototypes']
        bg_protos = features['bg_prototypes']
        avg_fg = features['avg_fg_prototype']
        avg_bg = features['avg_bg_prototype']
        
        n_support = len(fg_protos)
        C = avg_fg.shape[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 各Support的前景Prototype
        ax1 = axes[0, 0]
        for i, proto in enumerate(fg_protos):
            ax1.plot(proto[:100].numpy(), alpha=0.7, label=f'Support {i+1}')
        ax1.plot(avg_fg[:100].numpy(), 'k-', linewidth=2, label='Average')
        ax1.set_xlabel('Channel (first 100)')
        ax1.set_ylabel('Value')
        ax1.set_title('Foreground Prototypes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 各Support的背景Prototype
        ax2 = axes[0, 1]
        for i, proto in enumerate(bg_protos):
            ax2.plot(proto[:100].numpy(), alpha=0.7, label=f'Support {i+1}')
        ax2.plot(avg_bg[:100].numpy(), 'k-', linewidth=2, label='Average')
        ax2.set_xlabel('Channel (first 100)')
        ax2.set_ylabel('Value')
        ax2.set_title('Background Prototypes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # FG vs BG对比
        ax3 = axes[1, 0]
        ax3.plot(avg_fg[:100].numpy(), 'g-', linewidth=2, label='FG Prototype')
        ax3.plot(avg_bg[:100].numpy(), 'r-', linewidth=2, label='BG Prototype')
        ax3.fill_between(range(100), avg_fg[:100].numpy(), avg_bg[:100].numpy(), 
                        alpha=0.3, color='yellow')
        ax3.set_xlabel('Channel (first 100)')
        ax3.set_ylabel('Value')
        ax3.set_title('FG vs BG Prototype')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Prototype差异分布
        ax4 = axes[1, 1]
        diff = (avg_fg - avg_bg).numpy()
        ax4.hist(diff, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('FG - BG Difference')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Prototype Difference Distribution\nMean: {diff.mean():.4f}, Std: {diff.std():.4f}')
        ax4.grid(True, alpha=0.3)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(avg_fg.unsqueeze(0), avg_bg.unsqueeze(0)).item()
        plt.suptitle(f'Prototype Visualization\nFG-BG Cosine Similarity: {cos_sim:.4f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {save_path}")
        plt.close()
    
    def visualize_similarity_map(self, features: Dict, save_path: str = None):
        """
        可视化Similarity Map
        """
        query_img = features['query_img']
        fg_sim = features['fg_similarity'].numpy()
        bg_sim = features['bg_similarity'].numpy()
        contrast_sim = features['contrast_similarity'].numpy()
        
        # 获取点提示
        point_coords = features['point_coords_feat']  # 特征图尺度
        point_labels = features['point_labels']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原图
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        # 前景相似度
        im1 = axes[0, 1].imshow(fg_sim, cmap='hot')
        axes[0, 1].set_title(f'FG Similarity\nMin: {fg_sim.min():.3f}, Max: {fg_sim.max():.3f}')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # 背景相似度
        im2 = axes[0, 2].imshow(bg_sim, cmap='hot')
        axes[0, 2].set_title(f'BG Similarity\nMin: {bg_sim.min():.3f}, Max: {bg_sim.max():.3f}')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # 对比相似度（FG - BG）
        im3 = axes[1, 0].imshow(contrast_sim, cmap='RdBu_r', 
                                vmin=-abs(contrast_sim).max(), vmax=abs(contrast_sim).max())
        axes[1, 0].set_title(f'Contrast (FG - BG)\nMean: {contrast_sim.mean():.3f}')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # 阈值分割结果
        threshold = contrast_sim.mean() + 0.5 * contrast_sim.std()
        binary_mask = (contrast_sim > threshold).astype(np.float32)
        axes[1, 1].imshow(binary_mask, cmap='gray')
        axes[1, 1].set_title(f'Thresholded Mask\nThreshold: {threshold:.3f}')
        axes[1, 1].axis('off')
        
        # 相似度图 + Point Prompts
        axes[1, 2].imshow(fg_sim, cmap='hot')
        # 绘制采样点
        for (x, y), label in zip(point_coords, point_labels):
            color = 'lime' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            axes[1, 2].scatter(x, y, c=color, marker=marker, s=100, edgecolors='white', linewidths=2)
        axes[1, 2].set_title('Similarity + Point Prompts\n(Green=Pos, Red=Neg)')
        axes[1, 2].axis('off')
        
        plt.suptitle('Similarity Map Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {save_path}")
        plt.close()
    
    def visualize_point_prompts(self, features: Dict, save_path: str = None):
        """
        可视化Point Prompts在原图上的位置
        """
        query_img = features['query_img']
        point_coords_img = features['point_coords_img']  # 原图尺度
        point_labels = features['point_labels']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图 + 点
        axes[0].imshow(query_img)
        for (x, y), label in zip(point_coords_img, point_labels):
            color = 'lime' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            axes[0].scatter(x, y, c=color, marker=marker, s=200, edgecolors='white', linewidths=3)
        axes[0].set_title('Point Prompts on Image\n(Green=Positive, Red=Negative)')
        axes[0].axis('off')
        
        # 正样本点分布热力图
        H, W = np.array(query_img).shape[:2]
        pos_heatmap = np.zeros((H, W))
        for (x, y), label in zip(point_coords_img, point_labels):
            if label == 1:
                # 高斯模糊
                cv2.circle(pos_heatmap, (int(x), int(y)), 30, 1, -1)
        pos_heatmap = cv2.GaussianBlur(pos_heatmap, (51, 51), 0)
        
        axes[1].imshow(query_img)
        axes[1].imshow(pos_heatmap, cmap='Greens', alpha=0.6)
        axes[1].set_title('Positive Points Heatmap')
        axes[1].axis('off')
        
        # 负样本点分布热力图
        neg_heatmap = np.zeros((H, W))
        for (x, y), label in zip(point_coords_img, point_labels):
            if label == 0:
                cv2.circle(neg_heatmap, (int(x), int(y)), 30, 1, -1)
        neg_heatmap = cv2.GaussianBlur(neg_heatmap, (51, 51), 0)
        
        axes[2].imshow(query_img)
        axes[2].imshow(neg_heatmap, cmap='Reds', alpha=0.6)
        axes[2].set_title('Negative Points Heatmap')
        axes[2].axis('off')
        
        n_pos = sum(point_labels)
        n_neg = len(point_labels) - n_pos
        plt.suptitle(f'Point Prompts Visualization\nPositive: {n_pos}, Negative: {n_neg}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {save_path}")
        plt.close()
    
    def visualize_masks(self, features: Dict, gt_mask: np.ndarray = None, 
                       save_path: str = None):
        """
        可视化各阶段的Mask
        """
        query_img = features['query_img']
        rough_mask = features.get('rough_mask')
        all_masks = features.get('all_masks')
        mask_scores = features.get('mask_scores')
        
        n_cols = 4 if all_masks is not None else 3
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        
        # 第一行：原图、GT、Rough、Overlay
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        
        if gt_mask is not None:
            axes[0, 1].imshow(gt_mask, cmap='gray')
            axes[0, 1].set_title('Ground Truth')
        else:
            axes[0, 1].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=14)
        axes[0, 1].axis('off')
        
        if rough_mask is not None:
            axes[0, 2].imshow(rough_mask, cmap='gray')
            iou_text = ''
            if gt_mask is not None:
                inter = (rough_mask > 0) & (gt_mask > 0)
                union = (rough_mask > 0) | (gt_mask > 0)
                iou = inter.sum() / (union.sum() + 1e-8)
                iou_text = f'\nIoU: {iou*100:.1f}%'
            axes[0, 2].set_title(f'Rough Mask{iou_text}')
        else:
            axes[0, 2].text(0.5, 0.5, 'Failed', ha='center', va='center', fontsize=14)
        axes[0, 2].axis('off')
        
        # Overlay
        if rough_mask is not None:
            overlay = np.array(query_img).copy()
            mask_bool = rough_mask > 0
            overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 255 * 0.5, 0, 255)
            axes[0, 3].imshow(overlay.astype(np.uint8))
            axes[0, 3].set_title('Rough Mask Overlay')
        axes[0, 3].axis('off')
        
        # 第二行：SAM2多mask输出
        if all_masks is not None and mask_scores is not None:
            for i in range(min(3, len(all_masks))):
                axes[1, i].imshow(all_masks[i], cmap='gray')
                axes[1, i].set_title(f'Mask {i+1}\nScore: {mask_scores[i]:.3f}')
                axes[1, i].axis('off')
            
            # 最后显示所有mask叠加
            combined = np.zeros((*all_masks[0].shape, 3))
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # RGB
            for i, mask in enumerate(all_masks[:3]):
                for c in range(3):
                    combined[:, :, c] += mask * colors[i][c] * 0.5
            combined = np.clip(combined, 0, 1)
            axes[1, 3].imshow(combined)
            axes[1, 3].set_title('All Masks Combined\n(R=1, G=2, B=3)')
            axes[1, 3].axis('off')
        else:
            for i in range(4):
                axes[1, i].axis('off')
        
        plt.suptitle('Mask Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 保存: {save_path}")
        plt.close()
    
    def visualize_full_pipeline(self, query_img: Image.Image,
                               support_imgs: List[Image.Image],
                               support_masks: List[np.ndarray],
                               gt_mask: np.ndarray = None,
                               output_dir: str = 'vis_output',
                               prefix: str = 'sample'):
        """
        完整Pipeline可视化
        
        生成多个可视化文件
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 开始特征可视化: {prefix}")
        
        # 1. 提取所有特征
        features = self.extract_all_features(query_img, support_imgs, support_masks)
        
        # 2. 可视化DINO特征
        print("   🎨 可视化DINO特征...")
        self.visualize_dino_features(
            features, 
            save_path=str(output_dir / f'{prefix}_1_dino_features.png')
        )
        
        # 3. 可视化Prototype
        print("   🎨 可视化Prototype...")
        self.visualize_prototype(
            features,
            save_path=str(output_dir / f'{prefix}_2_prototype.png')
        )
        
        # 4. 可视化Similarity Map
        print("   🎨 可视化Similarity Map...")
        self.visualize_similarity_map(
            features,
            save_path=str(output_dir / f'{prefix}_3_similarity.png')
        )
        
        # 5. 可视化Point Prompts
        print("   🎨 可视化Point Prompts...")
        self.visualize_point_prompts(
            features,
            save_path=str(output_dir / f'{prefix}_4_points.png')
        )
        
        # 6. 可视化Masks
        print("   🎨 可视化Masks...")
        self.visualize_masks(
            features,
            gt_mask=gt_mask,
            save_path=str(output_dir / f'{prefix}_5_masks.png')
        )
        
        # 7. 创建综合大图
        print("   🎨 创建综合可视化...")
        self._create_summary_figure(features, gt_mask, 
                                   save_path=str(output_dir / f'{prefix}_0_summary.png'))
        
        print(f"   ✅ 所有可视化已保存到: {output_dir}")
        
        return features
    
    def _create_summary_figure(self, features: Dict, gt_mask: np.ndarray = None,
                              save_path: str = None):
        """创建综合可视化大图"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.2)
        
        query_img = features['query_img']
        
        # Row 1: Query + Support
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(query_img)
        ax1.set_title('Query', fontsize=10)
        ax1.axis('off')
        
        for i, (sup_img, sup_mask) in enumerate(zip(
            features['support_imgs'][:4], features['support_masks'][:4]
        )):
            ax = fig.add_subplot(gs[0, i+1])
            ax.imshow(sup_img)
            ax.imshow(sup_mask, alpha=0.4, cmap='Greens')
            ax.set_title(f'Support {i+1}', fontsize=10)
            ax.axis('off')
        
        # Row 2: DINO Features
        ax2_1 = fig.add_subplot(gs[1, 0])
        query_feat = features['query_dino']
        feat_norm = torch.norm(query_feat, dim=0).numpy()
        ax2_1.imshow(feat_norm, cmap='viridis')
        ax2_1.set_title('Query Feature Norm', fontsize=10)
        ax2_1.axis('off')
        
        ax2_2 = fig.add_subplot(gs[1, 1])
        ax2_2.imshow(features['fg_similarity'].numpy(), cmap='hot')
        ax2_2.set_title('FG Similarity', fontsize=10)
        ax2_2.axis('off')
        
        ax2_3 = fig.add_subplot(gs[1, 2])
        ax2_3.imshow(features['bg_similarity'].numpy(), cmap='hot')
        ax2_3.set_title('BG Similarity', fontsize=10)
        ax2_3.axis('off')
        
        ax2_4 = fig.add_subplot(gs[1, 3])
        contrast = features['contrast_similarity'].numpy()
        ax2_4.imshow(contrast, cmap='RdBu_r', 
                    vmin=-abs(contrast).max(), vmax=abs(contrast).max())
        ax2_4.set_title('Contrast (FG-BG)', fontsize=10)
        ax2_4.axis('off')
        
        ax2_5 = fig.add_subplot(gs[1, 4])
        # Similarity + Points
        ax2_5.imshow(features['fg_similarity'].numpy(), cmap='hot')
        point_coords = features['point_coords_feat']
        point_labels = features['point_labels']
        for (x, y), label in zip(point_coords, point_labels):
            color = 'lime' if label == 1 else 'red'
            ax2_5.scatter(x, y, c=color, s=50, edgecolors='white', linewidths=1)
        ax2_5.set_title('Similarity + Points', fontsize=10)
        ax2_5.axis('off')
        
        # Row 3: Prototype Analysis
        ax3_1 = fig.add_subplot(gs[2, :2])
        avg_fg = features['avg_fg_prototype']
        avg_bg = features['avg_bg_prototype']
        ax3_1.plot(avg_fg[:100].numpy(), 'g-', linewidth=2, label='FG', alpha=0.8)
        ax3_1.plot(avg_bg[:100].numpy(), 'r-', linewidth=2, label='BG', alpha=0.8)
        ax3_1.legend()
        ax3_1.set_xlabel('Channel')
        ax3_1.set_title('FG vs BG Prototype', fontsize=10)
        ax3_1.grid(True, alpha=0.3)
        
        # SAM2 multi-mask
        all_masks = features.get('all_masks')
        mask_scores = features.get('mask_scores')
        if all_masks is not None:
            for i in range(min(3, len(all_masks))):
                ax = fig.add_subplot(gs[2, 2+i])
                ax.imshow(all_masks[i], cmap='gray')
                ax.set_title(f'Mask {i+1}: {mask_scores[i]:.2f}', fontsize=10)
                ax.axis('off')
        
        # Row 4: Final Results
        ax4_1 = fig.add_subplot(gs[3, 0])
        ax4_1.imshow(query_img)
        ax4_1.set_title('Query', fontsize=10)
        ax4_1.axis('off')
        
        ax4_2 = fig.add_subplot(gs[3, 1])
        if gt_mask is not None:
            ax4_2.imshow(gt_mask, cmap='gray')
            ax4_2.set_title('GT Mask', fontsize=10)
        ax4_2.axis('off')
        
        ax4_3 = fig.add_subplot(gs[3, 2])
        rough_mask = features.get('rough_mask')
        if rough_mask is not None:
            ax4_3.imshow(rough_mask, cmap='gray')
            ax4_3.set_title('Rough Mask', fontsize=10)
        ax4_3.axis('off')
        
        ax4_4 = fig.add_subplot(gs[3, 3])
        if rough_mask is not None and gt_mask is not None:
            # Comparison overlay
            overlay = np.zeros((*gt_mask.shape, 3))
            overlay[gt_mask > 0, 1] = 1  # GT: Green
            overlay[rough_mask > 0, 0] = 1  # Pred: Red
            # Overlap: Yellow
            overlap = (gt_mask > 0) & (rough_mask > 0)
            overlay[overlap] = [1, 1, 0]
            ax4_4.imshow(overlay)
            ax4_4.set_title('GT(G) vs Pred(R)', fontsize=10)
        ax4_4.axis('off')
        
        ax4_5 = fig.add_subplot(gs[3, 4])
        if rough_mask is not None and gt_mask is not None:
            inter = (rough_mask > 0) & (gt_mask > 0)
            union = (rough_mask > 0) | (gt_mask > 0)
            iou = inter.sum() / (union.sum() + 1e-8)
            dice = 2 * inter.sum() / ((rough_mask > 0).sum() + (gt_mask > 0).sum() + 1e-8)
            
            ax4_5.text(0.5, 0.6, f'IoU: {iou*100:.2f}%', fontsize=20, ha='center', fontweight='bold')
            ax4_5.text(0.5, 0.4, f'Dice: {dice*100:.2f}%', fontsize=20, ha='center', fontweight='bold')
        ax4_5.axis('off')
        
        plt.suptitle('SP-SAM Pipeline Visualization', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='SP-SAM特征可视化')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='isic2018',
                       choices=['isic2018', 'fss1000'],
                       help='数据集类型')
    parser.add_argument('--sample_id', type=str, default=None,
                       help='指定样本ID（如 ISIC_0000000）')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='可视化样本数量')
    parser.add_argument('--k_shot', type=int, default=3,
                       help='Support数量')
    parser.add_argument('--output_dir', type=str, default='vis_output',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型类型')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型类型')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 加载模型
    print("🚀 加载模型...")
    from sp_sam_complete import SPSAMModel
    from src.model_manager import ModelManager
    
    model_manager = ModelManager(device=args.device)
    
    if 'dinov3' in args.dino_model.lower():
        dino_model, dino_transform = model_manager.load_dinov3_model(
            dinov3_model_name=args.dino_model
        )
    else:
        dino_model, dino_transform = model_manager.load_dinov2_model(
            dinov2_model_name=args.dino_model
        )
    
    sam2_model, sam2_predictor, _ = model_manager.load_sam2_model(
        sam2_model_type=args.sam2_model
    )
    
    sp_sam = SPSAMModel(
        sam2_model=sam2_model,
        sam2_predictor=sam2_predictor,
        dino_model=dino_model,
        dino_transform=dino_transform,
        device=args.device,
        sam2_model_type=args.sam2_model
    )
    
    # 创建可视化器
    visualizer = FeatureVisualizer(sp_sam, device=args.device)
    
    # 加载数据
    if args.dataset == 'isic2018':
        from isic2018_dataset import ISIC2018Dataset
        dataset = ISIC2018Dataset(args.data_root, support_ratio=0.2)
    else:
        from fss1000_dataset import FSS1000Dataset
        dataset = FSS1000Dataset(args.data_root)
    
    # 获取样本
    if args.dataset == 'isic2018':
        episode = dataset.get_episode(k_shot=args.k_shot)
        support_imgs = [s['img'] for s in episode['support']]
        support_masks = [s['mask'] for s in episode['support']]
        
        # 选择query
        if args.sample_id:
            query_sample = dataset.get_sample_by_id(args.sample_id)
            query_samples = [query_sample]
        else:
            query_samples = episode['query'][:args.num_samples]
        
        # 可视化每个query
        for i, query in enumerate(query_samples):
            prefix = query['sample_id'] if 'sample_id' in query else f'sample_{i}'
            visualizer.visualize_full_pipeline(
                query_img=query['img'],
                support_imgs=support_imgs,
                support_masks=support_masks,
                gt_mask=query['mask'],
                output_dir=args.output_dir,
                prefix=prefix
            )
    else:
        # FSS1000
        if args.sample_id:
            cls_name = args.sample_id
        else:
            cls_name = dataset.classes[0]
        
        episode = dataset.get_episode(cls_name, k_shot=args.k_shot)
        support_imgs = [s['img'] for s in episode['support']]
        support_masks = [s['mask'] for s in episode['support']]
        
        for i, query in enumerate(episode['query'][:args.num_samples]):
            prefix = f"{cls_name}_{query['img_name']}"
            visualizer.visualize_full_pipeline(
                query_img=query['img'],
                support_imgs=support_imgs,
                support_masks=support_masks,
                gt_mask=query['mask'],
                output_dir=args.output_dir,
                prefix=prefix
            )
    
    print(f"\n✅ 可视化完成! 输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()

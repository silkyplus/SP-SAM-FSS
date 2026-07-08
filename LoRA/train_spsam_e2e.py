"""
SP-SAM 完整端到端LoRA训练框架
================================

真正的端到端训练（与sp_sam_complete.py一致）：
1. CMRS阶段：DINO LoRA提取特征 → 相似度图 → 点prompts → SAM2 Predictor → rough_mask
2. Memory阶段：SAM2 LoRA (Image Encoder + Memory Encoder/Attention + Mask Decoder)
3. Loss计算：使用SAM2输出的final_mask与GT计算损失
4. 梯度反传：更新DINO LoRA + SAM2 LoRA + Adapter

使用方法:
    python LoRA/train_spsam_e2e.py --data_root jiguangdatasets --epochs 100
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# 兼容不同PyTorch版本的混合精度
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False

# 路径设置
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LORA_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from full_model_lora import SPSAMFullLoRA, LoRALinear


# ============================================================
# 损失函数
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.float().contiguous().view(-1)
        target = target.float().contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        
        # Dice Loss
        dice = self.dice_loss(pred, target)
        
        # BCE Loss（手动计算避免autocast问题）
        pred_clamp = torch.clamp(pred, 1e-7, 1 - 1e-7)
        bce = -(target * torch.log(pred_clamp) + (1 - target) * torch.log(1 - pred_clamp)).mean()
        
        return self.dice_weight * dice + self.bce_weight * bce


# ============================================================
# 数据集
# ============================================================

class SimpleDataset:
    """简单的数据集类"""
    
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        
        if mode == 'train':
            self.image_dir = os.path.join(data_root, 'train_images')
            self.mask_dir = os.path.join(data_root, 'train_masks')
        else:
            self.image_dir = os.path.join(data_root, 'test_images')
            self.mask_dir = os.path.join(data_root, 'test_masks')
        
        self.samples = []
        img_extensions = ['.png', '.PNG', '.bmp', '.BMP', '.jpg', '.JPG', '.jpeg', '.JPEG']
        
        if os.path.exists(self.image_dir) and os.path.exists(self.mask_dir):
            for f in sorted(os.listdir(self.image_dir)):
                ext = os.path.splitext(f)[1]
                if ext not in img_extensions:
                    continue
                
                img_path = os.path.join(self.image_dir, f)
                base_name = os.path.splitext(f)[0]
                
                mask_path = None
                for mask_ext in ['.png', '.PNG', '.bmp', '.BMP']:
                    candidate = os.path.join(self.mask_dir, base_name + mask_ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                
                if mask_path:
                    self.samples.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'name': base_name
                    })
        
        print(f"📁 {mode}数据集: {len(self.samples)} 张有效样本")
    
    def __len__(self):
        return len(self.samples)
    
    def get_sample(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert('RGB')
        mask_img = Image.open(sample['mask_path'])
        mask_array = np.array(mask_img.convert('L'))
        mask = (mask_array > 0).astype(np.float32)
        return image, mask
    
    def generate_episode(self, k_shot=1):
        n = len(self.samples)
        if n < k_shot + 1:
            raise ValueError(f"数据集太小: {n} < {k_shot + 1}")
        
        indices = np.random.choice(n, k_shot + 1, replace=False)
        
        support_images = []
        support_masks = []
        for i in range(k_shot):
            img, mask = self.get_sample(indices[i])
            support_images.append(img)
            support_masks.append(mask)
        
        query_image, query_mask = self.get_sample(indices[k_shot])
        
        return {
            'support_images': support_images,
            'support_masks': support_masks,
            'query_image': query_image,
            'query_mask': query_mask,
        }


# ============================================================
# SP-SAM Memory模块（可训练版本）
# ============================================================

class SAM2MemoryModuleTrainable(nn.Module):
    """
    SAM2 Memory模块 - 可训练版本
    
    在训练模式下保持梯度流动，使SAM2的LoRA参数可以被更新
    """
    
    def __init__(self, sam2_model, device='cuda'):
        super().__init__()
        self.model = sam2_model
        self.device = device
        self.image_size = getattr(sam2_model, 'image_size', 1024)
        self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
        
        # Memory缓存
        self.memory_features = None
        self.memory_pos_enc = None
    
    def _prepare_image(self, img):
        """准备图像tensor"""
        target_size = self.image_size
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return img_tensor.unsqueeze(0).to(self.device)
    
    def _prepare_mask(self, mask, target_size=None):
        """准备mask tensor"""
        if target_size is None:
            target_size = self.image_size
        mask_resized = cv2.resize(mask.astype(np.float32), (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).float()
        return mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def encode_support(self, support_img, support_mask):
        """
        编码support样本到memory
        
        训练模式下保持梯度
        """
        img_tensor = self._prepare_image(support_img)
        
        # Image Encoder
        backbone_out = self.model.forward_image(img_tensor)
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        pix_feat = feature_maps[-1]
        
        # Memory Encoder
        mask_tensor = self._prepare_mask(support_mask)
        high_res_masks = mask_tensor.float() * 20.0 - 10.0
        
        maskmem_out = self.model.memory_encoder(
            pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True
        )
        
        # 存储memory（保持梯度）
        if self.memory_features is None:
            self.memory_features = [maskmem_out["vision_features"]]
            self.memory_pos_enc = maskmem_out["vision_pos_enc"]
        else:
            self.memory_features.append(maskmem_out["vision_features"])
    
    def clear_memory(self):
        """清除memory"""
        self.memory_features = None
        self.memory_pos_enc = None
    
    def predict_query(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
        """
        预测query图像的分割（使用Memory Attention）
        
        返回logits（未经sigmoid）以便计算损失
        """
        if self.memory_features is None:
            raise ValueError("请先调用encode_support()")
        
        original_size = query_img.size[::-1]  # (H, W)
        img_tensor = self._prepare_image(query_img)
        B = img_tensor.shape[0]
        
        # Image Encoder
        backbone_out = self.model.forward_image(img_tensor)
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        
        high_res_features = feature_maps[:-1]
        pix_feat = feature_maps[-1]
        
        # Memory Attention
        to_cat_memory = [m.flatten(2).permute(2, 0, 1) for m in self.memory_features]
        to_cat_memory_pos = len(self.memory_features) * [
            self.memory_pos_enc[0].flatten(2).permute(2, 0, 1)
        ]
        
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos = torch.cat(to_cat_memory_pos, dim=0)
        
        pix_feat_with_mem = self.model.memory_attention(
            curr=pix_feat.flatten(2).permute(2, 0, 1),
            curr_pos=vision_pos_embeds[-1].flatten(2).permute(2, 0, 1),
            memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(*pix_feat.shape)
        
        # 准备prompts
        if point_coords is not None and point_labels is not None:
            scale_x = self.image_size / original_size[1]
            scale_y = self.image_size / original_size[0]
            
            scaled_coords = point_coords.copy().astype(np.float32)
            scaled_coords[:, 0] *= scale_x
            scaled_coords[:, 1] *= scale_y
            
            sam_point_coords = torch.from_numpy(scaled_coords).unsqueeze(0).float().to(self.device)
            sam_point_labels = torch.from_numpy(point_labels.astype(np.int32)).unsqueeze(0).to(self.device)
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=self.device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=self.device)
        
        # Mask Prompt
        sam_mask_prompt = None
        if rough_mask is not None:
            mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
            high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
            sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
                                            mode='bilinear', align_corners=False)
        
        # Prompt Encoder
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
        )
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()
        
        # Mask Decoder
        low_res_masks, ious, _, _ = self.model.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False, high_res_features=high_res_features
        )
        
        # 选择最佳mask
        best_idx = torch.argmax(ious[0])
        best_mask_logits = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
        
        # 上采样到目标尺寸
        high_res_logits = F.interpolate(best_mask_logits, size=(self.image_size, self.image_size),
                                        mode='bilinear', align_corners=False)
        
        return high_res_logits, ious[0, best_idx]


# ============================================================
# CMRS模块（可训练版本）- 使用SAM2 Predictor生成rough mask
# ============================================================

class CMRSModuleTrainable(nn.Module):
    """
    CMRS模块 - 可训练版本
    
    与sp_sam_complete.py一致：使用SAM2 Predictor生成rough mask
    """
    
    def __init__(self, spsam_model, sam2_predictor, device='cuda'):
        super().__init__()
        self.model = spsam_model
        self.sam2_predictor = sam2_predictor  # SAM2 Predictor用于生成rough mask
        self.device = device
    
    def compute_prototype(self, features, mask):
        """计算原型"""
        C, H, W = features.shape
        
        mask_resized = F.interpolate(
            torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='nearest'
        ).squeeze().to(self.device)
        
        mask_sum = mask_resized.sum() + 1e-8
        prototype = (features * mask_resized.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
        
        return prototype, mask_resized
    
    def compute_similarity(self, query_features, prototype):
        """计算相似度图"""
        C, H, W = query_features.shape
        
        query_flat = query_features.reshape(C, -1).T
        query_norm = F.normalize(query_flat, p=2, dim=1)
        proto_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
        
        similarity = torch.mm(query_norm, proto_norm.T).squeeze()
        similarity_map = similarity.reshape(H, W)
        
        return similarity_map
    
    def get_prompts_from_similarity(self, similarity_map, top_k=10, neg_k=5):
        """
        从相似度图生成点prompts
        与sp_sam_complete.py的实现一致
        """
        sim_np = similarity_map.detach().float().cpu().numpy().astype(np.float32)
        H, W = sim_np.shape
        
        # 计算统计量
        mean_sim = sim_np.mean()
        std_sim = sim_np.std()
        max_sim = sim_np.max()
        
        # ========== 正样本点采样 ==========
        # 使用自适应阈值
        fg_threshold = max(mean_sim + 1.0 * std_sim, max_sim * 0.5)
        fg_mask = sim_np > fg_threshold
        fg_coords = np.argwhere(fg_mask)
        
        # 如果点太少，降低阈值
        if len(fg_coords) < top_k:
            fg_threshold = max(mean_sim + 0.5 * std_sim, max_sim * 0.4)
            fg_mask = sim_np > fg_threshold
            fg_coords = np.argwhere(fg_mask)
        
        if len(fg_coords) < top_k:
            fg_threshold = mean_sim + 0.3 * std_sim
            fg_mask = sim_np > fg_threshold
            fg_coords = np.argwhere(fg_mask)
        
        # 采样正样本点
        if len(fg_coords) == 0:
            # 如果没有前景点，取相似度最高的点
            flat_indices = np.argsort(sim_np.flatten())[-top_k:]
            pos_points = np.array([np.unravel_index(idx, sim_np.shape) 
                                  for idx in flat_indices])
        else:
            # 加权空间采样
            pos_points = self._sample_points_weighted_spatial(fg_coords, sim_np, top_k)
        
        # ========== 负样本点采样 ==========
        bg_threshold = mean_sim - 0.5 * std_sim
        bg_mask = sim_np < bg_threshold
        bg_coords = np.argwhere(bg_mask)
        
        if len(bg_coords) >= neg_k:
            neg_points = self._sample_negative_points_diverse(bg_coords, sim_np, neg_k, (H, W))
        else:
            neg_points = self._sample_from_borders(H, W, neg_k)
        
        # 合并所有点
        all_points = np.vstack([pos_points, neg_points])
        pos_labels = np.ones(len(pos_points), dtype=np.int32)
        neg_labels = np.zeros(len(neg_points), dtype=np.int32)
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        # 坐标格式转换：(y, x) -> (x, y)
        point_coords = all_points[:, ::-1]
        
        return point_coords, all_labels
    
    def _sample_points_weighted_spatial(self, coords, sim_np, num_points):
        """加权空间采样 - 与sp_sam_complete.py一致"""
        sim_values = np.array([sim_np[y, x] for y, x in coords])
        
        selected = []
        selected_indices = []
        
        # 选择相似度最高的点作为第一个
        first_idx = np.argmax(sim_values)
        selected.append(coords[first_idx])
        selected_indices.append(first_idx)
        
        # 后续点：平衡相似度和空间分散性
        for _ in range(num_points - 1):
            if len(selected_indices) >= len(coords):
                break
            
            best_score = -np.inf
            best_idx = -1
            
            for i, (coord, sim_val) in enumerate(zip(coords, sim_values)):
                if i in selected_indices:
                    continue
                
                # 计算到已选点的最小距离
                min_dist = np.inf
                for sel_coord in selected:
                    dist = np.sqrt((coord[0] - sel_coord[0])**2 + 
                                  (coord[1] - sel_coord[1])**2)
                    min_dist = min(min_dist, dist)
                
                # 综合得分：相似度 * 0.7 + 空间分散性 * 0.3
                score = sim_val * 0.7 + (min_dist / max(sim_np.shape)) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(coords[best_idx])
                selected_indices.append(best_idx)
        
        return np.array(selected)
    
    def _sample_negative_points_diverse(self, bg_coords, sim_np, num_points, shape):
        """分散采样负样本点 - 与sp_sam_complete.py一致"""
        H, W = shape
        grid_h, grid_w = 3, 3
        cell_h, cell_w = H // grid_h, W // grid_w
        
        neg_points = []
        
        for gh in range(grid_h):
            for gw in range(grid_w):
                if len(neg_points) >= num_points:
                    break
                
                y_start, y_end = gh * cell_h, (gh + 1) * cell_h
                x_start, x_end = gw * cell_w, (gw + 1) * cell_w
                
                cell_mask = (
                    (bg_coords[:, 0] >= y_start) & (bg_coords[:, 0] < y_end) &
                    (bg_coords[:, 1] >= x_start) & (bg_coords[:, 1] < x_end)
                )
                cell_coords = bg_coords[cell_mask]
                
                if len(cell_coords) > 0:
                    cell_sims = np.array([sim_np[y, x] for y, x in cell_coords])
                    min_idx = np.argmin(cell_sims)
                    neg_points.append(cell_coords[min_idx])
        
        # 如果不够，随机补充
        if len(neg_points) < num_points:
            remaining = num_points - len(neg_points)
            if len(bg_coords) > 0:
                indices = np.random.choice(len(bg_coords), 
                                          min(remaining, len(bg_coords)), 
                                          replace=False)
                for idx in indices:
                    neg_points.append(bg_coords[idx])
        
        return np.array(neg_points) if neg_points else self._sample_from_borders(shape[0], shape[1], num_points)
    
    def _sample_from_borders(self, H, W, num_points):
        """从边界采样负样本点"""
        border_points = []
        # 上下边界
        for x in range(0, W, W // (num_points // 2 + 1)):
            border_points.append([0, x])
            border_points.append([H - 1, x])
        # 左右边界
        for y in range(0, H, H // (num_points // 2 + 1)):
            border_points.append([y, 0])
            border_points.append([y, W - 1])
        
        if len(border_points) >= num_points:
            indices = np.random.choice(len(border_points), num_points, replace=False)
            return np.array([border_points[i] for i in indices])
        return np.array(border_points[:num_points]) if border_points else np.array([[0, 0]] * num_points)
    
    def generate_rough_mask_and_prompts(self, query_img, support_imgs, support_masks):
        """
        生成rough mask和点prompts
        
        与sp_sam_complete.py一致的实现：
        1. DINO提取特征
        2. 计算相似度图
        3. 生成点提示
        4. SAM2 Predictor预测rough_mask
        
        返回:
            rough_mask: numpy array (H, W)
            similarity_map: torch tensor
            point_coords: numpy array (N, 2)
            point_labels: numpy array (N,)
        """
        # 1. 提取query特征
        query_features = self.model.extract_features(query_img)
        C, H_feat, W_feat = query_features.shape
        
        # 2. 计算每个support的prototype
        prototypes = []
        for sup_img, sup_mask in zip(support_imgs, support_masks):
            sup_features = self.model.extract_features(sup_img)
            proto, _ = self.compute_prototype(sup_features, sup_mask)
            prototypes.append(proto)
        
        # 3. 平均prototype并归一化
        avg_prototype = torch.stack(prototypes, dim=0).mean(dim=0)
        avg_prototype = F.normalize(avg_prototype, p=2, dim=0)
        
        # 4. 计算相似度图
        query_flat = query_features.reshape(C, -1).T
        query_flat = F.normalize(query_flat, p=2, dim=1)
        similarity_scores = torch.mv(query_flat, avg_prototype)
        similarity_map = similarity_scores.reshape(H_feat, W_feat)
        
        # 5. 采样点prompts（与sp_sam_complete.py一致）
        point_coords_feat, point_labels = self.get_prompts_from_similarity(
            similarity_map, top_k=10, neg_k=5
        )
        
        # 6. 坐标缩放：特征图尺度 → 原图尺度
        H_img, W_img = query_img.size[::-1]
        scale_y, scale_x = H_img / H_feat, W_img / W_feat
        
        point_coords_img = point_coords_feat.copy().astype(np.float32)
        point_coords_img[:, 0] *= scale_x  # x坐标
        point_coords_img[:, 1] *= scale_y  # y坐标
        point_coords_img = point_coords_img.astype(np.int32)
        
        # 7. SAM2 Predictor预测rough_mask（与sp_sam_complete.py一致）
        self.sam2_predictor.set_image(np.array(query_img))
        
        try:
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords_img,
                point_labels=point_labels,
                multimask_output=True
            )
            rough_mask = masks[np.argmax(scores)].astype(np.uint8)
        except Exception as e:
            print(f"   ⚠️ SAM2 Predictor预测失败: {e}")
            # 失败时回退到相似度图阈值化
            sim_np = similarity_map.detach().float().cpu().numpy()
            pred_prob = 1 / (1 + np.exp(-sim_np * 5))  # sigmoid
            pred_prob_full = cv2.resize(pred_prob, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            rough_mask = (pred_prob_full > 0.5).astype(np.uint8)
        
        return rough_mask, similarity_map, point_coords_img, point_labels


# ============================================================
# 端到端训练器
# ============================================================

class SPSAMEndToEndTrainer:
    """
    SP-SAM 端到端训练器
    
    完整的cmrs_memory训练流程（与sp_sam_complete.py一致）
    """
    
    def __init__(
        self,
        dino_model,
        dino_transform,
        sam2_model,
        sam2_predictor,
        device='cuda',
        dino_lora_rank=4,
        sam2_lora_rank=4,
        lr=1e-4,
        epochs=100,
        output_dir='outputs/spsam_e2e',
    ):
        self.device = device
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存SAM2 Predictor引用
        self.sam2_predictor = sam2_predictor
        
        # 创建全模型LoRA
        self.model = SPSAMFullLoRA(
            dino_model=dino_model,
            dino_transform=dino_transform,
            sam2_model=sam2_model,
            sam2_predictor=sam2_predictor,
            device=device,
            dino_lora_rank=dino_lora_rank,
            sam2_lora_rank=sam2_lora_rank,
            sam2_finetune_image_encoder=True,
            sam2_finetune_mask_decoder=True,
            sam2_finetune_memory_encoder=True,
            sam2_finetune_memory_attention=True,
            use_prototype_adapter=True,
        )
        
        # 创建CMRS和Memory模块（传入sam2_predictor）
        self.cmrs = CMRSModuleTrainable(self.model, sam2_predictor, device)
        self.memory_module = SAM2MemoryModuleTrainable(
            self.model.sam2_lora.sam2_model, device
        )
        
        # 损失函数
        self.criterion = CombinedLoss()
        
        # 优化器 - 所有可训练参数
        param_groups = self.model.get_all_trainable_parameters()
        all_params = []
        for group in param_groups:
            all_params.extend(group['params'])
        
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        # 混合精度
        if USE_NEW_AMP:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = GradScaler()
        
        self.best_iou = 0.0
        
        print("\n✅ SP-SAM端到端训练器初始化完成")
        print("   📌 CMRS使用SAM2 Predictor生成rough mask（与sp_sam_complete.py一致）")
    
    def train_episode(self, episode, debug=False):
        """训练一个episode（端到端）"""
        self.optimizer.zero_grad()
        
        # 清除memory
        self.memory_module.clear_memory()
        
        # 兼容不同PyTorch版本
        if USE_NEW_AMP:
            amp_context = autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_context = autocast()
        
        with amp_context:
            # ========== Stage 1: CMRS (使用SAM2 Predictor) ==========
            rough_mask, similarity_map, point_coords, point_labels = \
                self.cmrs.generate_rough_mask_and_prompts(
                    episode['query_image'],
                    episode['support_images'],
                    episode['support_masks']
                )
            
            if debug:
                print(f"  [CMRS] Rough mask sum: {rough_mask.sum()}")
                print(f"  [CMRS] Points: {len(point_coords)} ({(point_labels==1).sum()} pos)")
            
            # ========== Stage 2: Memory Encoding ==========
            for sup_img, sup_mask in zip(episode['support_images'], episode['support_masks']):
                self.memory_module.encode_support(sup_img, sup_mask)
            
            if debug:
                print(f"  [Memory] Encoded {len(episode['support_images'])} support samples")
            
            # ========== Stage 3: Query Prediction with Memory ==========
            pred_logits, iou_pred = self.memory_module.predict_query(
                episode['query_image'],
                rough_mask=rough_mask,
                point_coords=point_coords,
                point_labels=point_labels
            )
            
            # 准备GT
            H_pred, W_pred = pred_logits.shape[-2:]
            gt_mask = F.interpolate(
                torch.from_numpy(episode['query_mask']).float().unsqueeze(0).unsqueeze(0).to(self.device),
                size=(H_pred, W_pred),
                mode='nearest'
            )
            
            if debug:
                print(f"  [Memory] Pred logits shape: {pred_logits.shape}")
                print(f"  [Memory] GT mask shape: {gt_mask.shape}")
        
        # 计算损失（在autocast外，使用float32）
        pred_prob = torch.sigmoid(pred_logits.float().squeeze())
        gt_mask_f32 = gt_mask.float().squeeze()
        
        loss = self.criterion(pred_prob, gt_mask_f32)
        
        if debug:
            print(f"  [Loss] {loss.item():.4f}")
        
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.model.get_all_trainable_parameters() for p in group['params']],
            max_norm=1.0
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 计算IoU
        with torch.no_grad():
            pred_binary = (pred_prob > 0.5).float()
            intersection = (pred_binary * gt_mask_f32).sum()
            union = pred_binary.sum() + gt_mask_f32.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
        
        return loss.item(), iou
    
    def train(self, train_dataset, val_dataset=None, episodes_per_epoch=100, k_shot=1):
        """训练循环"""
        print("\n" + "="*60)
        print("🚀 开始SP-SAM端到端LoRA训练")
        print("   📌 CMRS阶段：DINO → 相似度 → 点prompts → SAM2 Predictor → rough_mask")
        print("   📌 Memory阶段：编码support → Memory Attention → Mask Decoder")
        print("="*60)
        
        for epoch in range(1, self.epochs + 1):
            # 训练
            self.model.dino_lora.dino_model.train()
            self.model.sam2_lora.sam2_model.train()
            if self.model.prototype_adapter:
                self.model.prototype_adapter.train()
            
            epoch_loss = 0.0
            epoch_iou = 0.0
            
            pbar = tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch}/{self.epochs}")
            
            for i in pbar:
                episode = train_dataset.generate_episode(k_shot=k_shot)
                debug = (epoch == 1 and i == 0)  # 第一个epoch第一个episode调试
                
                loss, iou = self.train_episode(episode, debug=debug)
                
                epoch_loss += loss
                epoch_iou += iou
                
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'iou': f'{iou:.4f}'
                })
            
            # 学习率调度
            self.scheduler.step()
            
            avg_loss = epoch_loss / episodes_per_epoch
            avg_iou = epoch_iou / episodes_per_epoch
            
            print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}, LR={self.scheduler.get_last_lr()[0]:.6f}")
            
            # 验证
            if val_dataset and epoch % 5 == 0:
                val_iou = self.validate(val_dataset, k_shot)
                print(f"  Validation IoU: {val_iou:.4f}")
                
                if val_iou > self.best_iou:
                    self.best_iou = val_iou
                    self.model.save_all_lora_weights(self.output_dir / 'best_lora.pth')
                    print(f"  ✅ 保存最佳模型 (IoU={val_iou:.4f})")
            
            # 定期保存
            if epoch % 20 == 0:
                self.model.save_all_lora_weights(self.output_dir / f'epoch_{epoch}.pth')
        
        # 保存最终模型
        self.model.save_all_lora_weights(self.output_dir / 'final_lora.pth')
        print(f"\n✅ 训练完成！最佳IoU: {self.best_iou:.4f}")
    
    def validate(self, val_dataset, k_shot=1, num_episodes=20):
        """验证"""
        self.model.dino_lora.dino_model.eval()
        self.model.sam2_lora.sam2_model.eval()
        if self.model.prototype_adapter:
            self.model.prototype_adapter.eval()
        
        total_iou = 0.0
        
        with torch.no_grad():
            for _ in range(num_episodes):
                episode = val_dataset.generate_episode(k_shot=k_shot)
                
                # 清除memory
                self.memory_module.clear_memory()
                
                # CMRS（使用SAM2 Predictor）
                rough_mask, _, point_coords, point_labels = \
                    self.cmrs.generate_rough_mask_and_prompts(
                        episode['query_image'],
                        episode['support_images'],
                        episode['support_masks']
                    )
                
                # Memory Encoding
                for sup_img, sup_mask in zip(episode['support_images'], episode['support_masks']):
                    self.memory_module.encode_support(sup_img, sup_mask)
                
                # Prediction
                pred_logits, _ = self.memory_module.predict_query(
                    episode['query_image'],
                    rough_mask=rough_mask,
                    point_coords=point_coords,
                    point_labels=point_labels
                )
                
                # 计算IoU
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float().squeeze()
                H_pred, W_pred = pred_mask.shape
                
                gt_mask = F.interpolate(
                    torch.from_numpy(episode['query_mask']).float().unsqueeze(0).unsqueeze(0).to(self.device),
                    size=(H_pred, W_pred),
                    mode='nearest'
                ).squeeze()
                
                intersection = (pred_mask * gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum() - intersection
                iou = (intersection / (union + 1e-8)).item()
                
                total_iou += iou
        
        return total_iou / num_episodes


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SP-SAM端到端LoRA训练')
    
    parser.add_argument('--data_root', type=str, default='jiguangdatasets')
    parser.add_argument('--dino_type', type=str, default='dinov3', choices=['dinov2', 'dinov3'])
    parser.add_argument('--sam2_type', type=str, default='large')
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--dino_lora_rank', type=int, default=4)
    parser.add_argument('--sam2_lora_rank', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--k_shot', type=int, default=1)
    
    parser.add_argument('--output_dir', type=str, default='outputs/spsam_e2e')
    
    args = parser.parse_args()
    
    # 加载模型
    print("\n📦 加载模型...")
    
    from model_manager import ModelManager
    manager = ModelManager(device=args.device)
    
    if args.dino_type == "dinov3":
        dino_model, dino_transform = manager.load_dinov3_model()
    else:
        dino_model, dino_transform = manager.load_dinov2_model()
    
    sam2_model, sam2_predictor, _ = manager.load_sam2_model(sam2_model_type=args.sam2_type)
    
    # 加载数据集
    print("\n📁 加载数据集...")
    data_root = os.path.join(ROOT_DIR, args.data_root)
    
    train_dataset = SimpleDataset(data_root, mode='train')
    val_dataset = SimpleDataset(data_root, mode='test')
    
    if len(train_dataset) == 0:
        print("❌ 训练集为空!")
        return
    
    # 创建训练器
    trainer = SPSAMEndToEndTrainer(
        dino_model=dino_model,
        dino_transform=dino_transform,
        sam2_model=sam2_model,
        sam2_predictor=sam2_predictor,
        device=args.device,
        dino_lora_rank=args.dino_lora_rank,
        sam2_lora_rank=args.sam2_lora_rank,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=os.path.join(ROOT_DIR, args.output_dir),
    )
    
    # 开始训练
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset if len(val_dataset) > 0 else None,
        episodes_per_epoch=args.episodes_per_epoch,
        k_shot=args.k_shot,
    )


if __name__ == "__main__":
    main()
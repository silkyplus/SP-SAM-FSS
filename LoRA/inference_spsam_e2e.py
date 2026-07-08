"""
SP-SAM 完整推理脚本 (cmrs_memory模式)
======================================

完整的SP-SAM流程（与sp_sam_complete.py一致）：
1. CMRS阶段：DINO LoRA提取特征 → 相似度图 → 点prompts → SAM2 Predictor → rough_mask
2. Memory阶段：SAM2 LoRA (Memory Encoder → Memory Attention → Mask Decoder)

使用方法:
    # 评估测试集
    python LoRA/inference_spsam_e2e.py \
        --data_root jiguangdatasets \
        --lora_weights outputs/spsam_e2e/best_lora.pth \
        --evaluate --visualize
    
    # 单图推理
    python LoRA/inference_spsam_e2e.py \
        --query test.jpg \
        --support train1.jpg train2.jpg \
        --support_mask mask1.png mask2.png \
        --lora_weights outputs/spsam_e2e/best_lora.pth
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# 路径设置
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LORA_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from full_model_lora import SPSAMFullLoRA, LoRALinear


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
        return image, mask, sample['name']


# ============================================================
# SP-SAM 完整推理器 (cmrs_memory模式)
# 使用与sp_sam_complete.py相同的方法
# ============================================================

class SPSAMCompleteInferencer:
    """
    SP-SAM完整推理器 - cmrs_memory模式
    
    完整流程（与sp_sam_complete.py一致）：
    1. CMRS: DINO LoRA → 相似度图 → 点prompts → SAM2 Predictor → rough_mask
    2. Memory: SAM2 LoRA (Memory Encoder → Memory Attention → Mask Decoder)
    """
    
    def __init__(
        self,
        lora_weights_path: str = None,
        device: str = 'cuda',
        dino_type: str = 'dinov3',
        sam2_type: str = 'large',
        dino_lora_rank: int = 4,
        sam2_lora_rank: int = 4,
    ):
        self.device = device
        
        print("="*60)
        print("🚀 加载SP-SAM完整模型 (cmrs_memory模式)")
        print("="*60)
        
        # 加载基础模型
        from model_manager import ModelManager
        manager = ModelManager(device=device)
        
        if dino_type == "dinov3":
            dino_model, dino_transform = manager.load_dinov3_model()
        else:
            dino_model, dino_transform = manager.load_dinov2_model()
        
        sam2_model, sam2_predictor, _ = manager.load_sam2_model(sam2_model_type=sam2_type)
        
        # 保存SAM2 Predictor引用（用于CMRS阶段生成rough mask）
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
        
        # SAM2模型引用（LoRA注入后的）
        self.sam2_model = self.model.sam2_lora.sam2_model
        self.image_size = getattr(sam2_model, 'image_size', 1024)
        self.num_feature_levels = getattr(sam2_model, 'num_feature_levels', 3)
        
        # 加载LoRA权重
        if lora_weights_path and os.path.exists(lora_weights_path):
            print(f"\n📦 加载LoRA权重: {lora_weights_path}")
            self.model.load_all_lora_weights(lora_weights_path)
        else:
            print(f"\n⚠️ 未加载LoRA权重，使用原始模型")
        
        # 设置为评估模式
        self.model.dino_lora.dino_model.eval()
        self.sam2_model.eval()
        if self.model.prototype_adapter:
            self.model.prototype_adapter.eval()
        
        # 验证LoRA注入
        self._verify_lora()
        
        # Memory缓存
        self.memory_features = None
        self.memory_pos_enc = None
    
    def _verify_lora(self):
        """验证LoRA注入状态"""
        print("\n🔍 验证LoRA注入状态...")
        
        dino_lora_count = 0
        for name, module in self.model.dino_lora.dino_model.named_modules():
            if isinstance(module, LoRALinear):
                dino_lora_count += 1
        
        sam2_lora_count = 0
        for name, module in self.sam2_model.named_modules():
            if isinstance(module, LoRALinear):
                sam2_lora_count += 1
        
        print(f"   DINO LoRA层数: {dino_lora_count}")
        print(f"   SAM2 LoRA层数: {sam2_lora_count}")
        
        if dino_lora_count > 0 and sam2_lora_count > 0:
            print("   ✅ LoRA注入正常")
        else:
            print("   ⚠️ 警告: 某些LoRA层可能缺失")
    
    # ==================== CMRS阶段 ====================
    
    def compute_prototype(self, features, mask):
        """计算原型（Masked Average Pooling）"""
        C, H, W = features.shape
        
        mask_resized = F.interpolate(
            torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='nearest'
        ).squeeze().to(self.device)
        
        mask_sum = mask_resized.sum() + 1e-8
        prototype = (features * mask_resized.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
        
        return prototype
    
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
        sim_np = similarity_map.float().cpu().numpy().astype(np.float32)
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
    
    def generate_cmrs_outputs(self, query_img, support_imgs, support_masks):
        """
        CMRS阶段：生成rough_mask和prompts
        
        与sp_sam_complete.py一致的实现：
        1. DINO LoRA提取特征
        2. 计算相似度图
        3. 生成点提示
        4. SAM2 Predictor预测rough_mask
        """
        with torch.no_grad():
            # 1. 提取query特征
            query_features = self.model.extract_features(query_img)
            C, H_feat, W_feat = query_features.shape
            
            # 2. 计算每个support的prototype
            support_features_list = []
            support_masks_list = []
            
            for sup_img, sup_mask in zip(support_imgs, support_masks):
                sup_features = self.model.extract_features(sup_img)
                
                # 下采样mask到特征图尺寸
                mask_resized = F.interpolate(
                    torch.from_numpy(sup_mask).float().unsqueeze(0).unsqueeze(0),
                    size=(sup_features.shape[1], sup_features.shape[2]),
                    mode='nearest'
                ).squeeze().to(self.device)
                
                support_features_list.append(sup_features)
                support_masks_list.append(mask_resized)
            
            # 3. 计算prototypes并融合
            prototypes = []
            for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
                mask_sum = sup_mask.sum() + 1e-8
                proto = (sup_feat * sup_mask.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
                prototypes.append(proto)
            
            # 平均prototype
            avg_prototype = torch.stack(prototypes, dim=0).mean(dim=0)
            avg_prototype = F.normalize(avg_prototype, p=2, dim=0)
            
            # 4. 计算相似度图
            query_flat = query_features.reshape(C, -1).T
            query_flat = F.normalize(query_flat, p=2, dim=1)
            similarity_scores = torch.mv(query_flat, avg_prototype)
            similarity_map = similarity_scores.reshape(H_feat, W_feat)
            
            # 5. 采样点prompts
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
                sim_np = similarity_map.float().cpu().numpy()
                pred_prob = 1 / (1 + np.exp(-sim_np * 5))  # sigmoid
                pred_prob_full = cv2.resize(pred_prob, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
                rough_mask = (pred_prob_full > 0.5).astype(np.uint8)
        
        return {
            'rough_mask': rough_mask,
            'similarity_map': similarity_map.float().cpu().numpy().astype(np.float32),
            'point_coords': point_coords_img,
            'point_labels': point_labels,
        }
    
    # ==================== Memory阶段 ====================
    
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
    
    def encode_support_to_memory(self, support_imgs, support_masks):
        """
        将support样本编码到memory
        
        使用SAM2的：
        - Image Encoder (LoRA)
        - Memory Encoder (LoRA)
        """
        self.memory_features = []
        self.memory_pos_enc = None
        
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            for sup_img, sup_mask in zip(support_imgs, support_masks):
                img_tensor = self._prepare_image(sup_img)
                
                # Image Encoder
                backbone_out = self.sam2_model.forward_image(img_tensor)
                feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
                pix_feat = feature_maps[-1]
                
                # Memory Encoder
                mask_tensor = self._prepare_mask(sup_mask)
                high_res_masks = mask_tensor.float() * 20.0 - 10.0
                
                maskmem_out = self.sam2_model.memory_encoder(
                    pix_feat=pix_feat, masks=high_res_masks, skip_mask_sigmoid=True
                )
                
                # 存储memory
                self.memory_features.append(maskmem_out["vision_features"].clone())
                
                if self.memory_pos_enc is None:
                    self.memory_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]
    
    def predict_with_memory(self, query_img, rough_mask=None, point_coords=None, point_labels=None):
        """
        使用Memory机制预测query
        
        使用SAM2的：
        - Image Encoder (LoRA)
        - Memory Attention (LoRA)
        - Mask Decoder (LoRA)
        """
        if self.memory_features is None:
            raise ValueError("请先调用encode_support_to_memory()")
        
        original_size = query_img.size[::-1]  # (H, W)
        
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            img_tensor = self._prepare_image(query_img)
            B = img_tensor.shape[0]
            
            # Image Encoder
            backbone_out = self.sam2_model.forward_image(img_tensor)
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
            
            pix_feat_with_mem = self.sam2_model.memory_attention(
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
            
            # Mask Prompt（来自rough_mask）
            sam_mask_prompt = None
            if rough_mask is not None:
                mask_tensor = self._prepare_mask(rough_mask, target_size=self.image_size)
                high_res_mask_logits = mask_tensor.float() * 20.0 - 10.0
                sam_mask_prompt = F.interpolate(high_res_mask_logits, size=(256, 256),
                                                mode='bilinear', align_corners=False)
            
            # Prompt Encoder
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
            )
            image_pe = self.sam2_model.sam_prompt_encoder.get_dense_pe()
            
            # Mask Decoder
            low_res_masks, ious, _, _ = self.sam2_model.sam_mask_decoder(
                image_embeddings=pix_feat_with_mem, image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False, high_res_features=high_res_features
            )
            
            # 选择最佳mask
            best_idx = torch.argmax(ious[0])
            best_mask = low_res_masks[0, best_idx:best_idx+1].unsqueeze(0)
            best_iou = ious[0, best_idx]
            
            # 上采样到原图尺寸
            best_mask = best_mask.float()
            high_res_masks = F.interpolate(best_mask, size=(self.image_size, self.image_size),
                                           mode='bilinear', align_corners=False)
            final_masks = F.interpolate(high_res_masks, size=original_size,
                                        mode='bilinear', align_corners=False)
            
            pred_mask = (final_masks[0, 0] > 0).cpu().numpy().astype(np.uint8)
            score = float(best_iou.cpu())
        
        return pred_mask, score
    
    # ==================== 完整预测 ====================
    
    def predict(self, query_img, support_imgs, support_masks, debug=False):
        """
        完整SP-SAM预测 (cmrs_memory模式)
        
        与sp_sam_complete.py一致的流程：
        1. CMRS: DINO → 相似度 → 点prompts → SAM2 Predictor → rough_mask
        2. Memory: 编码support → Memory Attention → Mask Decoder → final_mask
        """
        if debug:
            print("\n[SP-SAM cmrs_memory模式]")
        
        # Stage 1: CMRS（使用SAM2 Predictor生成rough_mask）
        if debug:
            print("  Stage 1: CMRS (SAM2 Predictor)...")
        cmrs_outputs = self.generate_cmrs_outputs(query_img, support_imgs, support_masks)
        
        if debug:
            print(f"    Rough mask sum: {cmrs_outputs['rough_mask'].sum()}")
            print(f"    Points: {len(cmrs_outputs['point_coords'])}")
        
        # Stage 2: Memory
        if debug:
            print("  Stage 2: Memory Refinement...")
        
        # 编码support到memory
        self.encode_support_to_memory(support_imgs, support_masks)
        
        if debug:
            print(f"    Encoded {len(support_imgs)} support samples to memory")
        
        # 使用memory预测
        final_mask, score = self.predict_with_memory(
            query_img,
            rough_mask=cmrs_outputs['rough_mask'],
            point_coords=cmrs_outputs['point_coords'],
            point_labels=cmrs_outputs['point_labels']
        )
        
        if debug:
            print(f"    Final mask sum: {final_mask.sum()}, score: {score:.3f}")
        
        return {
            'final_mask': final_mask,
            'rough_mask': cmrs_outputs['rough_mask'],
            'similarity_map': cmrs_outputs['similarity_map'],
            'score': score,
        }


# ============================================================
# 评估函数
# ============================================================

def compute_iou(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_dice(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum())) if (pred.sum() + gt.sum()) > 0 else 0.0


def visualize_result(query_img, gt_mask, rough_mask, final_mask, similarity_map,
                     support_imgs=None, support_masks=None, save_path=None):
    """可视化结果"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    k_shot = len(support_imgs) if support_imgs else 0
    n_rows = 3 if k_shot > 0 else 2
    n_cols = max(3, k_shot * 2)
    
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
    # Support
    if k_shot > 0:
        for i in range(k_shot):
            ax = fig.add_subplot(n_rows, n_cols, i * 2 + 1)
            ax.imshow(support_imgs[i])
            ax.set_title(f'Support {i+1}')
            ax.axis('off')
            
            ax = fig.add_subplot(n_rows, n_cols, i * 2 + 2)
            sup_np = np.array(support_imgs[i])
            sup_overlay = sup_np.copy()
            mask = support_masks[i]
            if mask.max() > 0:
                sup_overlay[mask > 0] = sup_overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            ax.imshow(sup_overlay.astype(np.uint8))
            ax.set_title(f'Support {i+1} Mask')
            ax.axis('off')
        
        row2_start = n_cols + 1
        row3_start = n_cols * 2 + 1
    else:
        row2_start = 1
        row3_start = n_cols + 1
    
    # Query, GT, Similarity
    ax = fig.add_subplot(n_rows, n_cols, row2_start)
    ax.imshow(query_img)
    ax.set_title('Query Image')
    ax.axis('off')
    
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 1)
    if gt_mask is not None:
        # 乘以255确保二值mask正确显示
        ax.imshow(gt_mask * 255, cmap='gray', vmin=0, vmax=255)
        ax.set_title('Ground Truth')
    ax.axis('off')
    
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 2)
    ax.imshow(similarity_map, cmap='hot')
    ax.set_title('Similarity Map')
    ax.axis('off')
    
    # Rough, Final, Overlay
    iou_rough = compute_iou(rough_mask, gt_mask) if gt_mask is not None else 0
    ax = fig.add_subplot(n_rows, n_cols, row3_start)
    # 乘以255确保二值mask正确显示
    ax.imshow(rough_mask * 255, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'CMRS Rough\n(IoU: {iou_rough*100:.1f}%)')
    ax.axis('off')
    
    iou_final = compute_iou(final_mask, gt_mask) if gt_mask is not None else 0
    ax = fig.add_subplot(n_rows, n_cols, row3_start + 1)
    # 乘以255确保二值mask正确显示
    ax.imshow(final_mask * 255, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Memory Final\n(IoU: {iou_final*100:.1f}%)')
    ax.axis('off')
    
    ax = fig.add_subplot(n_rows, n_cols, row3_start + 2)
    query_np = np.array(query_img)
    overlay = query_np.copy()
    if final_mask.max() > 0:
        overlay[final_mask > 0] = overlay[final_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    ax.imshow(overlay.astype(np.uint8))
    ax.set_title('Prediction Overlay')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_dataset(
    inferencer: SPSAMCompleteInferencer,
    data_root: str,
    k_shot: int = 1,
    output_dir: str = None,
    visualize: bool = False,
):
    """评估测试集"""
    print(f"\n{'='*60}")
    print(f"📊 SP-SAM (cmrs_memory) 评估测试集")
    print(f"{'='*60}")
    
    train_dataset = SimpleDataset(data_root, mode='train')
    test_dataset = SimpleDataset(data_root, mode='test')
    
    if len(train_dataset) < k_shot:
        print(f"❌ 训练集样本不足")
        return
    
    if len(test_dataset) == 0:
        print(f"❌ 测试集为空")
        return
    
    # 选择support样本
    support_imgs = []
    support_masks = []
    
    print(f"\n📷 Support样本 (k={k_shot}):")
    for idx in range(k_shot):
        img, mask, name = train_dataset.get_sample(idx)
        support_imgs.append(img)
        support_masks.append(mask)
        print(f"   - {name}")
    
    # 评估
    results = []
    ious_rough = []
    ious_final = []
    dices_rough = []
    dices_final = []
    
    if output_dir:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 评估 {len(test_dataset)} 个测试样本...")
    
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        query_img, gt_mask, name = test_dataset.get_sample(idx)
        debug = (idx < 2)
        
        try:
            pred_results = inferencer.predict(
                query_img, support_imgs, support_masks,
                debug=debug
            )
            
            iou_rough = compute_iou(pred_results['rough_mask'], gt_mask)
            iou_final = compute_iou(pred_results['final_mask'], gt_mask)
            dice_rough = compute_dice(pred_results['rough_mask'], gt_mask)
            dice_final = compute_dice(pred_results['final_mask'], gt_mask)
            
            ious_rough.append(iou_rough)
            ious_final.append(iou_final)
            dices_rough.append(dice_rough)
            dices_final.append(dice_final)
            
            results.append({
                'name': name,
                'iou_rough': iou_rough,
                'iou_final': iou_final,
                'dice_rough': dice_rough,
                'dice_final': dice_final,
                'score': pred_results['score'],
            })
            
            if visualize and output_dir:
                visualize_result(
                    query_img, gt_mask,
                    pred_results['rough_mask'],
                    pred_results['final_mask'],
                    pred_results['similarity_map'],
                    support_imgs, support_masks,
                    save_path=vis_dir / f"{name}_iou{iou_final*100:.1f}.png"
                )
            
        except Exception as e:
            print(f"\n⚠️ 评估 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 统计结果
    mean_iou_rough = np.mean(ious_rough) if ious_rough else 0.0
    mean_iou_final = np.mean(ious_final) if ious_final else 0.0
    mean_dice_rough = np.mean(dices_rough) if dices_rough else 0.0
    mean_dice_final = np.mean(dices_final) if dices_final else 0.0
    
    print(f"\n{'='*60}")
    print(f"📈 SP-SAM (cmrs_memory) 评估结果")
    print(f"{'='*60}")
    print(f"  K-shot: {k_shot}")
    print(f"  测试样本数: {len(ious_final)}")
    print(f"")
    print(f"  📊 CMRS粗分割 (Rough Mask):")
    print(f"     Mean IoU:  {mean_iou_rough*100:.2f}%")
    print(f"     Mean Dice: {mean_dice_rough*100:.2f}%")
    print(f"")
    print(f"  📊 Memory精炼后 (Final Mask):")
    print(f"     Mean IoU:  {mean_iou_final*100:.2f}%")
    print(f"     Mean Dice: {mean_dice_final*100:.2f}%")
    print(f"")
    
    iou_improvement = mean_iou_final - mean_iou_rough
    dice_improvement = mean_dice_final - mean_dice_rough
    
    print(f"  📈 Memory精炼效果:")
    print(f"     IoU变化:  {iou_improvement*100:+.2f}%")
    print(f"     Dice变化: {dice_improvement*100:+.2f}%")
    
    if iou_improvement > 0.02:
        print(f"     ✅ Memory机制有效提升分割质量!")
    elif iou_improvement < -0.05:
        print(f"     ⚠️ Memory精炼效果下降")
    else:
        print(f"     ➡️ Memory精炼效果持平")
    
    print(f"{'='*60}")
    
    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'mode': 'cmrs_memory',
            'k_shot': k_shot,
            'n_samples': len(ious_final),
            'rough_mask': {
                'mean_iou': mean_iou_rough,
                'mean_dice': mean_dice_rough,
            },
            'final_mask': {
                'mean_iou': mean_iou_final,
                'mean_dice': mean_dice_final,
            },
            'improvement': {
                'iou': iou_improvement,
                'dice': dice_improvement,
            },
            'per_sample_results': results,
        }
        
        with open(output_path / 'cmrs_memory_evaluation.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ 结果保存到: {output_path / 'cmrs_memory_evaluation.json'}")
    
    return {
        'mean_iou_rough': mean_iou_rough,
        'mean_iou_final': mean_iou_final,
        'mean_dice_rough': mean_dice_rough,
        'mean_dice_final': mean_dice_final,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SP-SAM完整推理 (cmrs_memory模式)')
    
    # 单图推理
    parser.add_argument('--query', type=str, help='查询图像路径')
    parser.add_argument('--support', type=str, nargs='+', help='support图像路径列表')
    parser.add_argument('--support_mask', type=str, nargs='+', help='support mask路径列表')
    
    # 数据集评估
    parser.add_argument('--data_root', type=str, default='jiguangdatasets')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--k_shot', type=int, default=1)
    
    # 模型参数
    parser.add_argument('--lora_weights', type=str)
    parser.add_argument('--dino_type', type=str, default='dinov3')
    parser.add_argument('--sam2_type', type=str, default='large')
    parser.add_argument('--dino_lora_rank', type=int, default=4)
    parser.add_argument('--sam2_lora_rank', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs/spsam_e2e_results')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    # 处理路径
    lora_weights = None
    if args.lora_weights:
        if os.path.isabs(args.lora_weights):
            lora_weights = args.lora_weights
        else:
            lora_weights = os.path.join(ROOT_DIR, args.lora_weights)
    
    # 创建推理器
    inferencer = SPSAMCompleteInferencer(
        lora_weights_path=lora_weights,
        device=args.device,
        dino_type=args.dino_type,
        sam2_type=args.sam2_type,
        dino_lora_rank=args.dino_lora_rank,
        sam2_lora_rank=args.sam2_lora_rank,
    )
    
    if args.evaluate:
        data_root = os.path.join(ROOT_DIR, args.data_root)
        output_dir = os.path.join(ROOT_DIR, args.output_dir)
        
        evaluate_dataset(
            inferencer,
            data_root=data_root,
            k_shot=args.k_shot,
            output_dir=output_dir,
            visualize=args.visualize,
        )
    
    elif args.query and args.support and args.support_mask:
        # 单图推理
        query_img = Image.open(args.query).convert('RGB')
        support_imgs = [Image.open(p).convert('RGB') for p in args.support]
        support_masks = []
        for p in args.support_mask:
            mask = np.array(Image.open(p).convert('L'))
            mask = (mask > 0).astype(np.float32)
            support_masks.append(mask)
        
        results = inferencer.predict(query_img, support_imgs, support_masks, debug=True)
        
        # 保存结果
        output_dir = Path(os.path.join(ROOT_DIR, args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_path = output_dir / 'prediction.png'
        Image.fromarray(results['final_mask'] * 255).save(mask_path)
        print(f"\n✅ 预测mask保存到: {mask_path}")
        
        if args.visualize:
            visualize_result(
                query_img, None,
                results['rough_mask'],
                results['final_mask'],
                results['similarity_map'],
                support_imgs, support_masks,
                save_path=output_dir / 'visualization.png'
            )
            print(f"✅ 可视化保存到: {output_dir / 'visualization.png'}")
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("示例用法:")
        print("="*60)
        print("\n1. 评估测试集 (使用LoRA权重):")
        print("   python LoRA/inference_spsam_e2e.py \\")
        print("       --data_root jiguangdatasets \\")
        print("       --lora_weights outputs/spsam_e2e/best_lora.pth \\")
        print("       --evaluate --visualize")
        print("\n2. 评估测试集 (不使用LoRA，原始模型):")
        print("   python LoRA/inference_spsam_e2e.py \\")
        print("       --data_root jiguangdatasets \\")
        print("       --evaluate --visualize")
        print("\n3. 单图推理:")
        print("   python LoRA/inference_spsam_e2e.py \\")
        print("       --query test.jpg \\")
        print("       --support train1.jpg train2.jpg \\")
        print("       --support_mask mask1.png mask2.png \\")
        print("       --lora_weights outputs/spsam_e2e/best_lora.pth")


if __name__ == '__main__':
    main()
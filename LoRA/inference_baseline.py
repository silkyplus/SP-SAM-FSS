"""
SP-SAM 原始模型推理脚本（无LoRA微调）
=====================================

使用预训练的DINO + SAM2进行cmrs_memory模式推理
用于获取baseline性能

使用方法:
    # 评估测试集（原始模型baseline）
    python LoRA/inference_baseline.py \
        --data_root jiguangdatasets \
        --evaluate --visualize
    
    # 单图推理
    python LoRA/inference_baseline.py \
        --query test.jpg \
        --support train1.jpg \
        --support_mask mask1.png
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
import torch.nn.functional as F
from PIL import Image
import cv2

# 路径设置
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LORA_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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
# SP-SAM 原始模型推理器 (cmrs_memory模式，无LoRA)
# ============================================================

class SPSAMBaselineInferencer:
    """
    SP-SAM原始模型推理器 - 无LoRA微调
    
    使用预训练的DINO + SAM2进行cmrs_memory模式推理
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        dino_type: str = 'dinov3',
        sam2_type: str = 'large',
    ):
        self.device = device
        
        print("="*60)
        print("🚀 加载SP-SAM原始模型（无LoRA）")
        print("="*60)
        
        # 加载模型
        from model_manager import ModelManager
        manager = ModelManager(device=device)
        
        if dino_type == "dinov3":
            self.dino_model, self.dino_transform = manager.load_dinov3_model()
        else:
            self.dino_model, self.dino_transform = manager.load_dinov2_model()
        
        self.sam2_model, self.sam2_predictor, _ = manager.load_sam2_model(sam2_model_type=sam2_type)
        
        # SAM2参数
        self.image_size = getattr(self.sam2_model, 'image_size', 1024)
        self.num_feature_levels = getattr(self.sam2_model, 'num_feature_levels', 3)
        
        # 设置为评估模式
        self.dino_model.eval()
        self.sam2_model.eval()
        
        # Memory缓存
        self.memory_features = None
        self.memory_pos_enc = None
        
        print("✅ 原始模型加载完成（无LoRA）")
    
    # ==================== DINO特征提取 ====================
    
    def extract_dino_features(self, img_pil):
        """提取DINO特征"""
        img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            features = self.dino_model.get_intermediate_layers(
                img_tensor.to(torch.bfloat16)
            )[0]
            h = w = int(features.shape[1] ** 0.5)
            feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
        
        return feature_map.squeeze(0)
    
    def compute_prototype(self, features, mask):
        """计算原型"""
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
    
    # ==================== CMRS阶段 ====================
    
    def generate_cmrs_outputs(self, query_img, support_imgs, support_masks):
        """CMRS阶段：生成rough_mask和prompts"""
        with torch.no_grad():
            # 提取query特征
            query_features = self.extract_dino_features(query_img)
            C, H_feat, W_feat = query_features.shape
            
            # 计算每个support的prototype
            prototypes = []
            for sup_img, sup_mask in zip(support_imgs, support_masks):
                sup_features = self.extract_dino_features(sup_img)
                proto = self.compute_prototype(sup_features, sup_mask)
                prototypes.append(proto)
            
            # 平均prototype
            avg_prototype = torch.stack(prototypes, dim=0).mean(dim=0)
            
            # 计算相似度
            similarity_map = self.compute_similarity(query_features, avg_prototype)
            
            # 生成rough mask
            pred_prob = torch.sigmoid(similarity_map * 5)
            
            # 上采样到原图尺寸
            H_img, W_img = query_img.size[::-1]
            pred_prob_np = pred_prob.float().cpu().numpy().astype(np.float32)
            pred_prob_full = cv2.resize(pred_prob_np, (W_img, H_img), interpolation=cv2.INTER_LINEAR)
            rough_mask = (pred_prob_full > 0.5).astype(np.uint8)
            
            # 生成点prompts
            point_coords, point_labels = self._get_point_prompts(
                similarity_map, (H_img, W_img), (H_feat, W_feat)
            )
        
        return {
            'rough_mask': rough_mask,
            'similarity_map': similarity_map.float().cpu().numpy().astype(np.float32),
            'point_coords': point_coords,
            'point_labels': point_labels,
        }
    
    def _get_point_prompts(self, similarity_map, img_size, feat_size, top_k=5, neg_k=3):
        """从相似度图生成点prompts"""
        sim_np = similarity_map.float().cpu().numpy().astype(np.float32)
        H_feat, W_feat = feat_size
        H_img, W_img = img_size
        
        # 正样本点
        flat_idx = np.argsort(sim_np.flatten())
        pos_indices = flat_idx[-top_k:]
        
        pos_points = []
        for idx in pos_indices:
            y, x = np.unravel_index(idx, sim_np.shape)
            x_img = int(x * W_img / W_feat)
            y_img = int(y * H_img / H_feat)
            pos_points.append([x_img, y_img])
        
        # 负样本点
        neg_indices = flat_idx[:neg_k]
        neg_points = []
        for idx in neg_indices:
            y, x = np.unravel_index(idx, sim_np.shape)
            x_img = int(x * W_img / W_feat)
            y_img = int(y * H_img / H_feat)
            neg_points.append([x_img, y_img])
        
        point_coords = np.array(pos_points + neg_points)
        point_labels = np.array([1] * len(pos_points) + [0] * len(neg_points))
        
        return point_coords, point_labels
    
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
        """将support样本编码到memory"""
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
        """使用Memory机制预测query"""
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
            
            # Mask Prompt
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
        """完整SP-SAM预测 (cmrs_memory模式)"""
        if debug:
            print("\n[SP-SAM Baseline - 无LoRA]")
        
        # Stage 1: CMRS
        if debug:
            print("  Stage 1: CMRS...")
        cmrs_outputs = self.generate_cmrs_outputs(query_img, support_imgs, support_masks)
        
        if debug:
            print(f"    Rough mask sum: {cmrs_outputs['rough_mask'].sum()}")
            print(f"    Points: {len(cmrs_outputs['point_coords'])}")
        
        # Stage 2: Memory
        if debug:
            print("  Stage 2: Memory...")
        
        self.encode_support_to_memory(support_imgs, support_masks)
        
        if debug:
            print(f"    Encoded {len(support_imgs)} support samples to memory")
        
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
        ax.imshow(gt_mask, cmap='gray')
        ax.set_title('Ground Truth')
    ax.axis('off')
    
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 2)
    ax.imshow(similarity_map, cmap='hot')
    ax.set_title('Similarity Map')
    ax.axis('off')
    
    # Rough, Final, Overlay
    iou_rough = compute_iou(rough_mask, gt_mask) if gt_mask is not None else 0
    ax = fig.add_subplot(n_rows, n_cols, row3_start)
    ax.imshow(rough_mask, cmap='gray')
    ax.set_title(f'CMRS Rough\n(IoU: {iou_rough*100:.1f}%)')
    ax.axis('off')
    
    iou_final = compute_iou(final_mask, gt_mask) if gt_mask is not None else 0
    ax = fig.add_subplot(n_rows, n_cols, row3_start + 1)
    ax.imshow(final_mask, cmap='gray')
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
    
    plt.suptitle('SP-SAM Baseline (No LoRA)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_dataset(
    inferencer: SPSAMBaselineInferencer,
    data_root: str,
    k_shot: int = 1,
    output_dir: str = None,
    visualize: bool = False,
):
    """评估测试集"""
    print(f"\n{'='*60}")
    print(f"📊 SP-SAM Baseline (无LoRA) 评估测试集")
    print(f"{'='*60}")
    
    train_dataset = SimpleDataset(data_root, mode='train')
    test_dataset = SimpleDataset(data_root, mode='test')
    
    if len(train_dataset) < k_shot:
        print(f"❌ 训练集样本不足")
        return
    
    if len(test_dataset) == 0:
        print(f"❌ 测试集为空")
        return
    
    # 选择support样本（固定使用前k个）
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
    print(f"📈 SP-SAM Baseline (无LoRA) 评估结果")
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
    print(f"{'='*60}")
    
    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'mode': 'baseline_no_lora',
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
        
        with open(output_path / 'baseline_evaluation.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ 结果保存到: {output_path / 'baseline_evaluation.json'}")
    
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
    parser = argparse.ArgumentParser(description='SP-SAM原始模型推理（无LoRA）')
    
    # 单图推理
    parser.add_argument('--query', type=str, help='查询图像路径')
    parser.add_argument('--support', type=str, nargs='+', help='support图像路径列表')
    parser.add_argument('--support_mask', type=str, nargs='+', help='support mask路径列表')
    
    # 数据集评估
    parser.add_argument('--data_root', type=str, default='jiguangdatasets')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--k_shot', type=int, default=1)
    
    # 模型参数
    parser.add_argument('--dino_type', type=str, default='dinov3')
    parser.add_argument('--sam2_type', type=str, default='large')
    parser.add_argument('--device', type=str, default='cuda')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs/baseline_results')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = SPSAMBaselineInferencer(
        device=args.device,
        dino_type=args.dino_type,
        sam2_type=args.sam2_type,
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
        print("\n1. 评估测试集（Baseline）:")
        print("   python LoRA/inference_baseline.py \\")
        print("       --data_root jiguangdatasets \\")
        print("       --evaluate --visualize")
        print("\n2. 单图推理:")
        print("   python LoRA/inference_baseline.py \\")
        print("       --query test.jpg \\")
        print("       --support train1.jpg \\")
        print("       --support_mask mask1.png")


if __name__ == '__main__':
    main()

"""
SP-SAM 全模型LoRA训练脚本
=========================

对整个模型进行LoRA微调：
- DINO/DINOv3: 所有Transformer块的Attention和MLP
- SAM2: Image Encoder + Mask Decoder + Memory模块
- Prototype Adapter: 额外的适配层

使用方法:
    python LoRA/train_full_lora.py --data_root jiguangdatasets --epochs 100
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

# 导入LoRA模块
from full_model_lora import SPSAMFullLoRA, DinoFullLoRA, SAM2FullLoRA, PrototypeAdapter


# ============================================================
# 损失函数 (autocast安全版本)
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 确保float32
        pred = pred.float()
        target = target.float()
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # 确保float32，使用binary_cross_entropy_with_logits的等效计算
        pred = pred.float()
        target = target.float()
        
        # 手动计算BCE，避免autocast问题
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        # 确保float32并clamp
        pred = pred.float().clamp(1e-7, 1 - 1e-7)
        target = target.float()
        
        # 手动计算BCE，避免autocast问题
        bce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
        
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        
        loss = (self.bce_weight * bce_loss +
                self.dice_weight * dice_loss +
                self.focal_weight * focal_loss)
        return loss


# ============================================================
# 数据集
# ============================================================

class SimpleDataset:
    """简单的数据集类"""
    
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        
        # 查找图像和mask
        if mode == 'train':
            self.image_dir = os.path.join(data_root, 'train_images')
            self.mask_dir = os.path.join(data_root, 'train_masks')
        else:
            self.image_dir = os.path.join(data_root, 'test_images')
            self.mask_dir = os.path.join(data_root, 'test_masks')
        
        # 获取所有有效样本（图像和mask都存在）
        self.samples = []
        img_extensions = ['.png', '.PNG', '.bmp', '.BMP', '.jpg', '.JPG', '.jpeg', '.JPEG']
        
        if os.path.exists(self.image_dir) and os.path.exists(self.mask_dir):
            for f in sorted(os.listdir(self.image_dir)):
                # 检查是否是图像文件
                ext = os.path.splitext(f)[1]
                if ext not in img_extensions:
                    continue
                
                img_path = os.path.join(self.image_dir, f)
                base_name = os.path.splitext(f)[0]
                
                # 查找对应的mask（尝试多种后缀）
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
        
        # 验证第一个样本
        if len(self.samples) > 0:
            img, mask = self.get_sample(0)
            fg_ratio = mask.sum() / mask.size * 100
            print(f"   验证样本: {self.samples[0]['name']}")
            print(f"   图像尺寸: {img.size}")
            print(f"   Mask尺寸: {mask.shape}, 前景比例: {fg_ratio:.2f}%")
    
    def __len__(self):
        return len(self.samples)
    
    def get_sample(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载图像 (PIL)
        image = Image.open(sample['img_path']).convert('RGB')
        
        # 加载mask (PIL)
        # 注意：mask可能是调色板模式(P)，前景值可能不是255
        # 先转灰度，然后用阈值>0来二值化
        mask_img = Image.open(sample['mask_path'])
        mask_array = np.array(mask_img.convert('L'))
        
        # 使用阈值>0来二值化（适应不同的mask格式）
        # 如果mask的前景是任何非零值（如38, 255等），都能正确识别
        mask = (mask_array > 0).astype(np.float32)
        
        return image, mask
    
    def generate_episode(self, k_shot=1):
        """生成一个few-shot episode"""
        n = len(self.samples)
        if n < k_shot + 1:
            raise ValueError(f"数据集太小: {n} < {k_shot + 1}")
        
        # 随机选择样本
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
# 训练器
# ============================================================

class FullLoRATrainer:
    """全模型LoRA训练器"""
    
    def __init__(
        self,
        dino_model,
        dino_transform,
        sam2_model,
        sam2_predictor,
        device='cuda',
        # LoRA配置
        dino_lora_rank=4,
        sam2_lora_rank=4,
        # 训练配置
        lr=1e-4,
        epochs=100,
        output_dir='outputs',
    ):
        self.device = device
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # 损失函数
        self.criterion = CombinedLoss()
        
        # 优化器
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
        
        # 最佳结果
        self.best_iou = 0.0
    
    def compute_prototype(self, features, mask):
        """计算原型"""
        C, H, W = features.shape
        
        # 下采样mask
        mask_resized = F.interpolate(
            torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='nearest'
        ).squeeze().to(self.device)
        
        # Masked Average Pooling
        mask_sum = mask_resized.sum() + 1e-8
        prototype = (features * mask_resized.unsqueeze(0)).sum(dim=(1, 2)) / mask_sum
        
        return prototype, mask_resized
    
    def compute_similarity(self, query_features, prototype):
        """计算相似度图"""
        C, H, W = query_features.shape
        
        # 归一化
        query_flat = query_features.reshape(C, -1).T  # (H*W, C)
        query_norm = F.normalize(query_flat, p=2, dim=1)
        proto_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
        
        # 相似度
        similarity = torch.mm(query_norm, proto_norm.T).squeeze()
        similarity_map = similarity.reshape(H, W)
        
        return similarity_map
    
    def train_episode(self, episode, debug=False):
        """训练一个episode"""
        self.optimizer.zero_grad()
        
        # 兼容不同PyTorch版本
        if USE_NEW_AMP:
            amp_context = autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_context = autocast()
        
        with amp_context:
            # 提取support特征和计算原型
            prototypes = []
            for sup_img, sup_mask in zip(episode['support_images'], episode['support_masks']):
                sup_features = self.model.extract_features(sup_img)
                proto, mask_resized = self.compute_prototype(sup_features, sup_mask)
                prototypes.append(proto)
                
                if debug:
                    print(f"  [DEBUG] Support image size: {sup_img.size}")
                    print(f"  [DEBUG] Support mask shape: {sup_mask.shape}, sum: {sup_mask.sum()}, max: {sup_mask.max()}")
                    print(f"  [DEBUG] Support features shape: {sup_features.shape}")
                    print(f"  [DEBUG] Mask resized sum: {mask_resized.sum().item():.2f}")
                    print(f"  [DEBUG] Prototype norm: {proto.norm().item():.4f}")
            
            # 平均原型
            avg_prototype = torch.stack(prototypes, dim=0).mean(dim=0)
            
            # 提取query特征
            query_features = self.model.extract_features(episode['query_image'])
            
            if debug:
                print(f"  [DEBUG] Query image size: {episode['query_image'].size}")
                print(f"  [DEBUG] Query mask shape: {episode['query_mask'].shape}, sum: {episode['query_mask'].sum()}")
                print(f"  [DEBUG] Query features shape: {query_features.shape}")
                print(f"  [DEBUG] Avg prototype norm: {avg_prototype.norm().item():.4f}")
            
            # 计算相似度
            similarity_map = self.compute_similarity(query_features, avg_prototype)
            
            if debug:
                print(f"  [DEBUG] Similarity map - min: {similarity_map.min().item():.4f}, max: {similarity_map.max().item():.4f}, mean: {similarity_map.mean().item():.4f}")
            
            pred_mask = torch.sigmoid(similarity_map)
            
            if debug:
                print(f"  [DEBUG] Pred mask - min: {pred_mask.min().item():.4f}, max: {pred_mask.max().item():.4f}, mean: {pred_mask.mean().item():.4f}")
            
            # 准备GT
            C, H, W = query_features.shape
            gt_mask = F.interpolate(
                torch.from_numpy(episode['query_mask']).float().unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            ).squeeze().to(self.device)
            
            if debug:
                print(f"  [DEBUG] GT mask resized - shape: {gt_mask.shape}, sum: {gt_mask.sum().item():.2f}, max: {gt_mask.max().item()}")
        
        # 在autocast外计算损失（确保float32）
        pred_mask_f32 = pred_mask.float()
        gt_mask_f32 = gt_mask.float()
        loss = self.criterion(pred_mask_f32, gt_mask_f32)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.model.get_all_trainable_parameters() for p in group['params']],
            1.0
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 计算IoU
        with torch.no_grad():
            pred_binary = (pred_mask > 0.5).float()
            intersection = (pred_binary * gt_mask).sum()
            union = pred_binary.sum() + gt_mask.sum() - intersection
            iou = intersection / (union + 1e-8)
            
            if debug:
                print(f"  [DEBUG] Pred binary sum: {pred_binary.sum().item():.2f}")
                print(f"  [DEBUG] GT mask sum: {gt_mask.sum().item():.2f}")
                print(f"  [DEBUG] Intersection: {intersection.item():.2f}, Union: {union.item():.2f}")
                print(f"  [DEBUG] IoU: {iou.item():.4f}")
        
        return loss.item(), iou.item()
    
    def train(self, train_dataset, val_dataset=None, episodes_per_epoch=100, k_shot=1):
        """完整训练"""
        print("\n" + "="*60)
        print("🚀 开始全模型LoRA训练")
        print("="*60)
        
        for epoch in range(self.epochs):
            epoch_losses = []
            epoch_ious = []
            
            pbar = tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for i in pbar:
                episode = train_dataset.generate_episode(k_shot=k_shot)
                
                # 第一个epoch的第一个episode启用调试
                debug = (epoch == 0 and i == 0)
                if debug:
                    print("\n" + "="*50)
                    print("🔍 调试信息 (第一个episode)")
                    print("="*50)
                
                loss, iou = self.train_episode(episode, debug=debug)
                
                if debug:
                    print("="*50 + "\n")
                
                epoch_losses.append(loss)
                epoch_ious.append(iou)
                
                pbar.set_postfix({
                    'loss': f"{np.mean(epoch_losses[-10:]):.4f}",
                    'iou': f"{np.mean(epoch_ious[-10:]):.4f}",
                })
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印epoch结果
            avg_loss = np.mean(epoch_losses)
            avg_iou = np.mean(epoch_ious)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"  Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, LR: {lr:.6f}")
            
            # 验证
            if val_dataset is not None and (epoch + 1) % 5 == 0:
                val_iou = self.validate(val_dataset, k_shot=k_shot)
                print(f"  验证 IoU: {val_iou:.4f}")
                
                if val_iou > self.best_iou:
                    self.best_iou = val_iou
                    self.save_checkpoint("best_lora.pth")
            
            # 定期保存
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_checkpoint("final_lora.pth")
        print(f"\n✅ 训练完成! 最佳IoU: {self.best_iou:.4f}")
    
    def validate(self, val_dataset, k_shot=1, num_episodes=50):
        """验证"""
        ious = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                episode = val_dataset.generate_episode(k_shot=k_shot)
                
                # 提取特征和原型
                prototypes = []
                for sup_img, sup_mask in zip(episode['support_images'], episode['support_masks']):
                    sup_features = self.model.extract_features(sup_img)
                    proto, _ = self.compute_prototype(sup_features, sup_mask)
                    prototypes.append(proto)
                
                avg_prototype = torch.stack(prototypes, dim=0).mean(dim=0)
                query_features = self.model.extract_features(episode['query_image'])
                similarity_map = self.compute_similarity(query_features, avg_prototype)
                pred_mask = torch.sigmoid(similarity_map)
                
                # 计算IoU
                C, H, W = query_features.shape
                gt_mask = F.interpolate(
                    torch.from_numpy(episode['query_mask']).float().unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='nearest'
                ).squeeze().to(self.device)
                
                pred_binary = (pred_mask > 0.5).float()
                intersection = (pred_binary * gt_mask).sum()
                union = pred_binary.sum() + gt_mask.sum() - intersection
                iou = intersection / (union + 1e-8)
                ious.append(iou.item())
        
        return np.mean(ious)
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        path = self.output_dir / filename
        self.model.save_all_lora_weights(str(path))


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SP-SAM 全模型LoRA训练")
    
    # 数据
    parser.add_argument("--data_root", type=str, default="jiguangdatasets", help="数据集路径")
    parser.add_argument("--k_shot", type=int, default=1, help="Few-shot K值")
    
    # LoRA配置
    parser.add_argument("--dino_lora_rank", type=int, default=4, help="DINO LoRA rank")
    parser.add_argument("--sam2_lora_rank", type=int, default=4, help="SAM2 LoRA rank")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--episodes_per_epoch", type=int, default=100, help="每轮episode数")
    
    # 模型路径（根据你的实际路径修改）
    parser.add_argument("--dino_type", type=str, default="dinov3", choices=["dinov2", "dinov3"],
                       help="DINO类型")
    parser.add_argument("--sam2_type", type=str, default="large", help="SAM2模型类型")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/full_lora", help="输出目录")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SP-SAM 全模型LoRA微调")
    print("="*60)
    print(f"DINO LoRA Rank: {args.dino_lora_rank}")
    print(f"SAM2 LoRA Rank: {args.sam2_lora_rank}")
    print(f"K-shot: {args.k_shot}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("="*60)
    
    # 加载模型
    print("\n📦 加载模型...")
    
    try:
        from model_manager import ModelManager
        manager = ModelManager(device=args.device)
        
        # 加载DINO
        if args.dino_type == "dinov3":
            dino_model, dino_transform = manager.load_dinov3_model()
        else:
            dino_model, dino_transform = manager.load_dinov2_model()
        
        # 加载SAM2
        sam2_model, sam2_predictor, _ = manager.load_sam2_model(sam2_model_type=args.sam2_type)
        
        print("✅ 模型加载完成")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载数据集
    print("\n📁 加载数据集...")
    
    data_root = os.path.join(ROOT_DIR, args.data_root)
    if not os.path.exists(data_root):
        print(f"❌ 数据集路径不存在: {data_root}")
        return
    
    train_dataset = SimpleDataset(data_root, mode='train')
    val_dataset = SimpleDataset(data_root, mode='test')
    
    if len(train_dataset) == 0:
        print("❌ 训练集为空!")
        return
    
    # 创建训练器
    trainer = FullLoRATrainer(
        dino_model=dino_model,
        dino_transform=dino_transform,
        sam2_model=sam2_model,
        sam2_predictor=sam2_predictor,
        device=args.device,
        dino_lora_rank=args.dino_lora_rank,
        sam2_lora_rank=args.sam2_lora_rank,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
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
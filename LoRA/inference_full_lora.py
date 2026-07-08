"""
SP-SAM 完整推理脚本（真正使用SAM2）
====================================

正确的SP-SAM流程：
1. DINO提取特征 → 相似度图 → Rough Mask (CMRS)
2. 从Rough Mask生成SAM2 prompts (点/框)
3. SAM2 Image Encoder处理图像（使用LoRA）
4. SAM2 Mask Decoder生成精细mask（使用LoRA）

使用方法：
    python LoRA/inference_spsam.py --data_root jiguangdatasets --lora_weights outputs/full_lora/best_lora.pth --evaluate
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
# SP-SAM 完整推理器
# ============================================================

class SPSAMInferencer:
    """
    SP-SAM完整推理器
    
    真正使用LoRA微调后的SAM2模型进行分割
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
        print("🚀 加载SP-SAM完整模型")
        print("="*60)
        
        # 加载基础模型
        from model_manager import ModelManager
        manager = ModelManager(device=device)
        
        if dino_type == "dinov3":
            dino_model, dino_transform = manager.load_dinov3_model()
        else:
            dino_model, dino_transform = manager.load_dinov2_model()
        
        sam2_model, sam2_predictor, _ = manager.load_sam2_model(sam2_model_type=sam2_type)
        
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
        
        # 保存对SAM2模型的引用（这是LoRA注入后的模型）
        self.sam2_model = self.model.sam2_lora.sam2_model
        self.sam2_predictor = sam2_predictor  # predictor内部引用的是同一个sam2_model
        
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
        
        # 验证LoRA是否正确注入
        self._verify_lora_injection()
    
    def _verify_lora_injection(self):
        """验证LoRA是否正确注入到SAM2"""
        print("\n🔍 验证LoRA注入状态...")
        
        lora_count = 0
        for name, module in self.sam2_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_count += 1
        
        print(f"   SAM2模型中LoRA层数: {lora_count}")
        
        if lora_count == 0:
            print("   ⚠️ 警告: SAM2模型中没有LoRA层!")
        else:
            print(f"   ✅ SAM2 LoRA注入正常")
        
        # 检查predictor是否使用同一个模型
        if hasattr(self.sam2_predictor, 'model'):
            if self.sam2_predictor.model is self.sam2_model:
                print("   ✅ SAM2 Predictor使用LoRA模型")
            else:
                print("   ⚠️ SAM2 Predictor可能使用不同的模型实例!")
    
    def compute_prototype(self, features, mask):
        """计算原型"""
        C, H, W = features.shape
        
        mask_resized = F.interpolate(
            torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='nearest'
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
    
    def predict(
        self,
        query_img: Image.Image,
        support_imgs: list,
        support_masks: list,
        threshold: float = 0.5,
        debug: bool = False,
    ):
        """
        SP-SAM完整推理
        
        流程：
        1. DINO提取特征 → 相似度图 → Rough Mask (CMRS)
        2. 从Rough Mask生成SAM2 prompts
        3. SAM2生成精细mask（使用LoRA微调后的模型）
        """
        with torch.no_grad():
            # ========== 第1阶段：CMRS (粗分割) ==========
            if debug:
                print("\n[Stage 1] CMRS - 粗分割")
            
            # 提取support特征并计算原型
            prototypes = []
            for sup_img, sup_mask in zip(support_imgs, support_masks):
                sup_features = self.model.extract_features(sup_img)
                proto = self.compute_prototype(sup_features, sup_mask)
                prototypes.append(proto)
            
            avg_prototype = torch.stack(prototypes, dim=0).mean(dim=0)
            
            # 提取query特征
            query_features = self.model.extract_features(query_img)
            
            # 计算相似度
            similarity_map = self.compute_similarity(query_features, avg_prototype)
            
            # 生成粗分割
            pred_prob = torch.sigmoid(similarity_map * 5)
            
            # 上采样到原图尺寸
            H_img, W_img = query_img.size[::-1]
            pred_prob_np = pred_prob.cpu().numpy()
            pred_prob_full = np.array(
                Image.fromarray((pred_prob_np * 255).astype(np.uint8)).resize(
                    (W_img, H_img), Image.BILINEAR
                )
            ) / 255.0
            rough_mask = (pred_prob_full > threshold).astype(np.uint8)
            
            if debug:
                print(f"   Rough mask: shape={rough_mask.shape}, sum={rough_mask.sum()}")
            
            # ========== 第2阶段：SAM2精炼 ==========
            if debug:
                print("\n[Stage 2] SAM2精炼 (使用LoRA模型)")
            
            final_mask = self._sam2_predict(query_img, rough_mask, debug=debug)
        
        return {
            'final_mask': final_mask,
            'rough_mask': rough_mask,
            'similarity_map': similarity_map.cpu().numpy(),
            'pred_prob': pred_prob.cpu().numpy(),
        }
    
    def _sam2_predict(self, query_img, rough_mask, debug=False):
        """
        使用SAM2生成精细mask
        
        这里使用的是LoRA注入后的SAM2模型
        """
        try:
            if rough_mask.sum() == 0:
                if debug:
                    print("   Rough mask为空，跳过SAM2")
                return rough_mask
            
            img_np = np.array(query_img)
            H_img, W_img = rough_mask.shape
            
            # 设置图像 - 这会调用LoRA版本的image encoder
            self.sam2_predictor.set_image(img_np)
            
            if debug:
                print(f"   SAM2 image encoder已处理图像")
            
            # 从rough_mask生成prompts
            y_coords, x_coords = np.where(rough_mask > 0)
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            
            # 扩大bbox
            margin_x = max(10, int((x_max - x_min) * 0.15))
            margin_y = max(10, int((y_max - y_min) * 0.15))
            x_min = max(0, x_min - margin_x)
            x_max = min(W_img - 1, x_max + margin_x)
            y_min = max(0, y_min - margin_y)
            y_max = min(H_img - 1, y_max + margin_y)
            
            box = np.array([[x_min, y_min, x_max, y_max]])
            
            # 质心
            center_x = int(x_coords.mean())
            center_y = int(y_coords.mean())
            
            if debug:
                print(f"   Box: [{x_min}, {y_min}, {x_max}, {y_max}]")
                print(f"   Center: ({center_x}, {center_y})")
            
            # 使用Box + Point进行预测 - 这会调用LoRA版本的mask decoder
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                box=box,
                multimask_output=True
            )
            
            if masks is None or len(masks) == 0:
                if debug:
                    print("   SAM2返回空结果，使用rough mask")
                return rough_mask
            
            # 选择最佳mask
            best_idx = np.argmax(scores)
            final_mask = masks[best_idx].astype(np.uint8)
            
            if debug:
                print(f"   SAM2输出: {len(masks)}个mask, 最佳score={scores[best_idx]:.3f}")
                print(f"   Final mask sum: {final_mask.sum()}")
            
            # 验证结果合理性
            inter = (final_mask & rough_mask).sum()
            union = (final_mask | rough_mask).sum()
            iou_with_rough = inter / (union + 1e-8)
            
            if debug:
                print(f"   IoU with rough: {iou_with_rough:.3f}")
            
            if iou_with_rough < 0.3:
                if debug:
                    print("   IoU过低，SAM2可能分割错误，使用rough mask")
                return rough_mask
            
            return final_mask
            
        except Exception as e:
            print(f"   SAM2预测失败: {e}")
            import traceback
            traceback.print_exc()
            return rough_mask


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


def evaluate_dataset(
    inferencer: SPSAMInferencer,
    data_root: str,
    k_shot: int = 1,
    output_dir: str = None,
    visualize: bool = False,
):
    """评估测试集"""
    print(f"\n{'='*60}")
    print(f"📊 SP-SAM 评估测试集")
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
    
    # 第一个样本输出详细调试信息
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        query_img, gt_mask, name = test_dataset.get_sample(idx)
        debug = (idx < 2)  # 前2个样本输出调试信息
        
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
            continue
    
    # 统计结果
    mean_iou_rough = np.mean(ious_rough) if ious_rough else 0.0
    mean_iou_final = np.mean(ious_final) if ious_final else 0.0
    mean_dice_rough = np.mean(dices_rough) if dices_rough else 0.0
    mean_dice_final = np.mean(dices_final) if dices_final else 0.0
    
    print(f"\n{'='*60}")
    print(f"📈 SP-SAM 评估结果")
    print(f"{'='*60}")
    print(f"  K-shot: {k_shot}")
    print(f"  测试样本数: {len(ious_final)}")
    print(f"")
    print(f"  📊 CMRS粗分割 (Rough Mask):")
    print(f"     Mean IoU:  {mean_iou_rough*100:.2f}%")
    print(f"     Mean Dice: {mean_dice_rough*100:.2f}%")
    print(f"")
    print(f"  📊 SAM2精炼后 (Final Mask):")
    print(f"     Mean IoU:  {mean_iou_final*100:.2f}%")
    print(f"     Mean Dice: {mean_dice_final*100:.2f}%")
    print(f"")
    
    iou_improvement = mean_iou_final - mean_iou_rough
    dice_improvement = mean_dice_final - mean_dice_rough
    
    print(f"  📈 SAM2精炼效果:")
    print(f"     IoU变化:  {iou_improvement*100:+.2f}%")
    print(f"     Dice变化: {dice_improvement*100:+.2f}%")
    
    if iou_improvement > 0.02:
        print(f"     ✅ SAM2 LoRA有效提升分割质量!")
    elif iou_improvement < -0.05:
        print(f"     ⚠️ SAM2精炼效果下降")
    else:
        print(f"     ➡️ SAM2精炼效果持平")
    
    print(f"{'='*60}")
    
    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
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
        
        with open(output_path / 'spsam_evaluation.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ 结果保存到: {output_path / 'spsam_evaluation.json'}")
    
    return {
        'mean_iou_rough': mean_iou_rough,
        'mean_iou_final': mean_iou_final,
        'mean_dice_rough': mean_dice_rough,
        'mean_dice_final': mean_dice_final,
    }


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
    
    if k_shot > 0:
        # Support
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
    ax.imshow(gt_mask, cmap='gray')
    ax.set_title('Ground Truth')
    ax.axis('off')
    
    ax = fig.add_subplot(n_rows, n_cols, row2_start + 2)
    ax.imshow(similarity_map, cmap='hot')
    ax.set_title('Similarity Map')
    ax.axis('off')
    
    # Rough, Final, Overlay
    iou_rough = compute_iou(rough_mask, gt_mask)
    ax = fig.add_subplot(n_rows, n_cols, row3_start)
    ax.imshow(rough_mask, cmap='gray')
    ax.set_title(f'CMRS Rough\n(IoU: {iou_rough*100:.1f}%)')
    ax.axis('off')
    
    iou_final = compute_iou(final_mask, gt_mask)
    ax = fig.add_subplot(n_rows, n_cols, row3_start + 1)
    ax.imshow(final_mask, cmap='gray')
    ax.set_title(f'SAM2 Final\n(IoU: {iou_final*100:.1f}%)')
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


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SP-SAM完整推理')
    
    parser.add_argument('--data_root', type=str, default='jiguangdatasets')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--k_shot', type=int, default=1)
    
    parser.add_argument('--lora_weights', type=str)
    parser.add_argument('--dino_type', type=str, default='dinov3')
    parser.add_argument('--sam2_type', type=str, default='large')
    parser.add_argument('--dino_lora_rank', type=int, default=4)
    parser.add_argument('--sam2_lora_rank', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--output_dir', type=str, default='outputs/spsam_results')
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
    inferencer = SPSAMInferencer(
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
    else:
        parser.print_help()
        print("\n使用 --evaluate 进行评估")


if __name__ == '__main__':
    main()
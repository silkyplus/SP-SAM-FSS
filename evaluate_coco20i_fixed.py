"""
SP-SAM在COCO-20i上的完整评估脚本
===================================

支持四种评估模式：
1. rough_only: 纯CMRS生成的rough mask (消融实验基线)
2. cmrs_predictor: CMRS + SAM2 Predictor 
3. memory_only: 纯Memory机制 (跳过CMRS)
4. cmrs_memory: CMRS + Memory机制 (完整SP-SAM)

使用方法：
    # 评估单个fold
    python evaluate_coco20i.py --coco20i_root ./coco20i --coco_root D:/data/coco2014 --fold 0 --mode cmrs_memory
    
    # 评估所有folds
    python evaluate_coco20i.py --coco20i_root ./coco20i --coco_root D:/data/coco2014 --mode cmrs_memory
    
    # 使用排序后的support
    python evaluate_coco20i.py --coco20i_root ./coco20i --coco_root D:/data/coco2014 --use_ranked_support
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import time
import os
from typing import List, Dict, Optional
import argparse
import cv2

# 导入数据集
from coco20i_dataset import COCO20iDataset

# 导入SP-SAM模块
from sp_sam_complete import *

# 导入模型管理器
from src.model_manager import ModelManager
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# 评估指标计算
# ============================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    计算IoU
    
    Args:
        pred_mask: 预测mask [H, W]
        gt_mask: ground truth mask [H, W]
        
    Returns:
        iou: IoU值
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(iou)


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict:
    """
    计算完整的评估指标
    
    Returns:
        metrics: Dict with iou, precision, recall, f1
    """
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # IoU
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # Precision, Recall, F1
    tp = intersection
    fp = (pred_mask & ~gt_mask).sum()
    fn = (~pred_mask & gt_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


# ============================================================
# SP-SAM评估器
# ============================================================

class SPSAMEvaluator:
    """
    SP-SAM在COCO-20i上的评估器
    """
    
    def __init__(self, sp_sam_model: SPSAMModel, output_dir: str = 'evaluation_results'):
        """
        Args:
            sp_sam_model: SP-SAM模型实例
            output_dir: 结果输出目录
        """
        self.sp_sam = sp_sam_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 结果存储
        self.results = []
    
    def evaluate_episode(self, episode: Dict, 
                        use_cmrs: bool = True,
                        use_memory_refinement: bool = False,
                        use_predictor_refinement: bool = False,  # ★ 新增参数
                        rough_only: bool = False,
                        visualize: bool = False,
                        max_test_samples: Optional[int] = None) -> Optional[Dict]:
        """
        评估单个episode
        
        Args:
            episode: episode dict（包含support和test）
            use_cmrs: 是否使用CMRS
            use_memory_refinement: 是否使用Memory机制细化
            use_predictor_refinement: 是否使用SAM2 Predictor精炼（用于cmrs_predictor模式）
            rough_only: 是否只使用rough mask（消融实验）
            visualize: 是否可视化结果
            max_test_samples: 最大测试样本数（用于快速测试）
            
        Returns:
            episode_results: Dict
        """
        class_name = episode['class']
        class_id = episode['class_id']
        support_samples = episode['support']
        test_samples = episode['test']
        
        if len(support_samples) == 0:
            print(f"   ⚠️  类别 {class_name}: 无support样本，跳过")
            return None
        
        if len(test_samples) == 0:
            print(f"   ⚠️  类别 {class_name}: 无test样本，跳过")
            return None
        
        # 限制测试样本数
        if max_test_samples is not None and len(test_samples) > max_test_samples:
            print(f"   ℹ️  限制测试样本数: {len(test_samples)} -> {max_test_samples}")
            test_samples = test_samples[:max_test_samples]
        
        print(f"\n{'='*60}")
        print(f"评估类别: {class_name} (id={class_id})")
        if rough_only:
            print(f"模式: 纯CMRS (rough mask only)")
        else:
            print(f"模式: CMRS={'是' if use_cmrs else '否'}, Memory={'是' if use_memory_refinement else '否'}")
        print(f"Support: {len(support_samples)} samples | Test: {len(test_samples)} samples")
        print(f"{'='*60}")
        
        # 准备support data
        support_images = [s['img'] for s in support_samples]
        support_masks = [s['mask'] for s in support_samples]
        
        # ========================================
        # 优化：一次性设置support，缓存Memory
        # ========================================
        if use_memory_refinement:
            print("   📦 初始化Support Memory...")
            try:
                self.sp_sam.set_support(support_images, support_masks)
                use_optimized_api = True
            except Exception as e:
                print(f"   ⚠️  set_support失败: {e}, 使用传统API")
                use_optimized_api = False
        else:
            use_optimized_api = False
        
        # 评估每个test样本
        test_results = []
        total_time = 0
        
        for test_idx, test_sample in enumerate(tqdm(test_samples, desc=f"Testing {class_name}")):
            query_img = test_sample['img']
            gt_mask = test_sample['mask']
            img_name = test_sample.get('img_name', f'test_{test_idx}')
            
            # 预测
            start_time = time.time()
            try:
                if use_optimized_api:
                    # 优化API：复用缓存的support Memory
                    pred_results = self.sp_sam.predict_query(
                        query_img,
                        use_cmrs=use_cmrs,
                        use_memory_refinement=use_memory_refinement,
                        use_predictor_refinement=use_predictor_refinement  # ★ 传递新参数
                    )
                else:
                    # 传统API：每次传入support
                    pred_results = self.sp_sam.predict(
                        query_img,
                        support_images,
                        support_masks,
                        use_cmrs=use_cmrs,
                        use_memory_refinement=use_memory_refinement,
                        use_predictor_refinement=use_predictor_refinement  # ★ 传递新参数
                    )
                
                pred_mask = pred_results.get('final_mask')
                rough_mask = pred_results.get('rough_mask')
                
                # 如果是rough_only模式，使用rough_mask作为最终预测
                if rough_only:
                    if rough_mask is not None:
                        pred_mask = rough_mask
                    else:
                        print(f"   ⚠️  rough mask不可用: {img_name}")
                        pred_mask = np.zeros_like(gt_mask)
                elif pred_mask is None:
                    print(f"   ⚠️  预测失败: {img_name}")
                    pred_mask = np.zeros_like(gt_mask)
                
            except Exception as e:
                print(f"   ❌ 预测出错: {img_name}, {e}")
                import traceback
                traceback.print_exc()
                pred_mask = np.zeros_like(gt_mask)
                rough_mask = None
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 计算指标
            metrics = compute_metrics(pred_mask, gt_mask)
            
            test_results.append({
                'img_name': img_name,
                'iou': metrics['iou'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'inference_time': inference_time
            })
            
            # 可视化（可选）
            if visualize and test_idx < 5:  # 只可视化前5个
                self._visualize_result(
                    query_img, gt_mask, pred_mask, rough_mask,
                    class_name, img_name, metrics
                )
        
        # 计算该类别的平均指标
        mean_iou = np.mean([r['iou'] for r in test_results])
        mean_precision = np.mean([r['precision'] for r in test_results])
        mean_recall = np.mean([r['recall'] for r in test_results])
        mean_f1 = np.mean([r['f1'] for r in test_results])
        mean_time = np.mean([r['inference_time'] for r in test_results])
        
        episode_result = {
            'class': class_name,
            'class_id': class_id,
            'fold': episode['fold'],
            'k_shot': episode['k_shot'],
            'num_test_samples': len(test_results),
            'mean_iou': mean_iou,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_inference_time': mean_time,
            'total_time': total_time,
            'test_results': test_results
        }
        
        print(f"\n📊 {class_name} 结果:")
        print(f"   mIoU: {mean_iou:.4f} ({mean_iou*100:.2f}%)")
        print(f"   Precision: {mean_precision:.4f}")
        print(f"   Recall: {mean_recall:.4f}")
        print(f"   F1: {mean_f1:.4f}")
        print(f"   平均推理时间: {mean_time:.3f}s")
        
        self.results.append(episode_result)
        
        return episode_result
    
    def _visualize_result(self, query_img: Image.Image, gt_mask: np.ndarray, 
                         pred_mask: np.ndarray, rough_mask: Optional[np.ndarray],
                         class_name: str, img_name: str, metrics: Dict):
        """可视化单个预测结果"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Query图像
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image")
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(query_img)
        axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Rough Mask (如果有)
        if rough_mask is not None:
            axes[2].imshow(query_img)
            axes[2].imshow(rough_mask, alpha=0.5, cmap='jet')
            axes[2].set_title("Rough Mask (CMRS)")
            axes[2].axis('off')
        else:
            axes[2].axis('off')
        
        # Prediction
        axes[3].imshow(query_img)
        axes[3].imshow(pred_mask, alpha=0.5, cmap='jet')
        axes[3].set_title(f"Prediction\nIoU={metrics['iou']:.3f}")
        axes[3].axis('off')
        
        plt.suptitle(f"{class_name} - {img_name}")
        plt.tight_layout()
        
        # 保存
        vis_dir = self.output_dir / 'visualizations' / class_name
        vis_dir.mkdir(exist_ok=True, parents=True)
        save_path = vis_dir / f"{Path(img_name).stem}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def evaluate_fold(self, dataset: COCO20iDataset,
                     k_shot: int = 5,
                     use_cmrs: bool = True,
                     use_memory_refinement: bool = False,
                     use_predictor_refinement: bool = False,  # ★ 新增参数
                     rough_only: bool = False,
                     visualize: bool = False,
                     max_test_samples: Optional[int] = None) -> List[Dict]:
        """
        评估一个fold的所有类别
        
        Returns:
            fold_results: List of episode results
        """
        episodes = dataset.get_all_episodes(k_shot=k_shot)
        
        fold_results = []
        for episode in episodes:
            result = self.evaluate_episode(
                episode,
                use_cmrs=use_cmrs,
                use_memory_refinement=use_memory_refinement,
                use_predictor_refinement=use_predictor_refinement,  # ★ 传递新参数
                rough_only=rough_only,
                visualize=visualize,
                max_test_samples=max_test_samples
            )
            if result is not None:
                fold_results.append(result)
        
        # 计算fold的平均mIoU
        if fold_results:
            fold_mean_iou = np.mean([r['mean_iou'] for r in fold_results])
            
            print(f"\n{'='*60}")
            print(f"Fold {dataset.fold} 汇总:")
            print(f"   评估类别数: {len(fold_results)}")
            print(f"   平均mIoU: {fold_mean_iou:.4f} ({fold_mean_iou*100:.2f}%)")
            print(f"{'='*60}")
        
        return fold_results
    
    def evaluate_all_folds(self, coco20i_root: str, coco_root: str,
                          k_shot: int = 5,
                          use_cmrs: bool = True,
                          use_memory_refinement: bool = False,
                          use_predictor_refinement: bool = False,  # ★ 新增参数
                          rough_only: bool = False,
                          visualize: bool = False,
                          use_ranked_support: bool = True,
                          max_test_samples: Optional[int] = None) -> Dict:
        """
        评估所有4个folds
        
        Returns:
            all_results: Dict with fold results and overall metrics
        """
        all_fold_results = []
        
        for fold in range(4):
            print(f"\n{'#'*60}")
            print(f"# 评估 Fold {fold}")
            print(f"{'#'*60}")
            
            try:
                # 加载数据集
                dataset = COCO20iDataset(
                    coco20i_root,
                    coco_root,
                    fold,
                    use_ranked_support=use_ranked_support
                )
                
                # 评估fold
                fold_results = self.evaluate_fold(
                    dataset,
                    k_shot=k_shot,
                    use_cmrs=use_cmrs,
                    use_memory_refinement=use_memory_refinement,
                    use_predictor_refinement=use_predictor_refinement,  # ★ 传递新参数
                    rough_only=rough_only,
                    visualize=visualize,
                    max_test_samples=max_test_samples
                )
                
                if fold_results:
                    fold_mean_iou = np.mean([r['mean_iou'] for r in fold_results])
                    all_fold_results.append({
                        'fold': fold,
                        'mean_iou': fold_mean_iou,
                        'class_results': fold_results
                    })
                    
            except Exception as e:
                print(f"❌ Fold {fold} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算overall mIoU
        if all_fold_results:
            overall_miou = np.mean([r['mean_iou'] for r in all_fold_results])
            
            print(f"\n{'='*80}")
            print(f"COCO-20i 整体结果 ({k_shot}-shot)")
            print(f"{'='*80}")
            
            for fold_result in all_fold_results:
                fold = fold_result['fold']
                miou = fold_result['mean_iou']
                n_classes = len(fold_result['class_results'])
                print(f"Fold {fold}: {miou:.4f} ({miou*100:.2f}%) - {n_classes}个类别")
            
            print(f"\n总体mIoU: {overall_miou:.4f} ({overall_miou*100:.2f}%)")
            print(f"{'='*80}")
            
            # 保存结果
            self._save_all_results(all_fold_results, overall_miou, k_shot)
            
            return {
                'overall_miou': overall_miou,
                'fold_results': all_fold_results
            }
        else:
            print("\n❌ 没有成功评估任何fold")
            return None
    
    def _save_all_results(self, all_fold_results: List[Dict], 
                         overall_miou: float, k_shot: int):
        """保存所有fold的结果"""
        # 准备数据
        rows = []
        for fold_result in all_fold_results:
            fold = fold_result['fold']
            for class_result in fold_result['class_results']:
                rows.append({
                    'fold': fold,
                    'class': class_result['class'],
                    'class_id': class_result['class_id'],
                    'k_shot': k_shot,
                    'num_test_samples': class_result['num_test_samples'],
                    'mean_iou': class_result['mean_iou'],
                    'mean_precision': class_result['mean_precision'],
                    'mean_recall': class_result['mean_recall'],
                    'mean_f1': class_result['mean_f1'],
                    'mean_inference_time': class_result['mean_inference_time']
                })
        
        # 保存CSV
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f'coco20i_results_{k_shot}shot.csv'
        df.to_csv(csv_path, index=False)
        
        # 保存JSON
        results_json = {
            'k_shot': k_shot,
            'overall_miou': overall_miou,
            'fold_results': all_fold_results
        }
        json_path = self.output_dir / f'coco20i_results_{k_shot}shot.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n💾 结果已保存:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='SP-SAM在COCO-20i上的评估')
    parser.add_argument('--coco20i_root', type=str, required=True,
                       help='COCO-20i数据集根目录')
    parser.add_argument('--coco_root', type=str, required=True,
                       help='COCO原始数据集根目录')
    parser.add_argument('--fold', type=int, default=None,
                       help='评估特定fold (0-3)，默认评估所有folds')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Support样本数 (1 or 5)')
    parser.add_argument('--use_ranked_support', action='store_true',
                       help='使用排序后的support (samples_ranked.json)')
    parser.add_argument('--output_dir', type=str, default='coco20i_evaluation_results',
                       help='结果输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='每类最大测试样本数（用于快速测试）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型: dinov2_vitb14, dinov2_vitl14, dinov3_vitb16, dinov3_vitl16')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型: tiny, small, base_plus, large')
    parser.add_argument('--mode', type=str, default='cmrs_memory',
                       choices=['rough_only', 'cmrs_predictor', 'cmrs_memory', 'memory_only'],
                       help='''评估模式:
                           rough_only: 纯CMRS rough mask (消融实验基线)
                           cmrs_predictor: CMRS + SAM2 Predictor
                           cmrs_memory: CMRS + Memory机制 (完整SP-SAM)
                           memory_only: 纯Memory机制''')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # ============================================
    # 设置全局随机种子，确保可复现
    # ============================================
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 解析评估模式
    if args.mode == 'rough_only':
        use_cmrs = True
        use_memory_refinement = False
        use_predictor_refinement = False  # ★ 不使用Predictor精炼
        rough_only = True
        mode_desc = "纯CMRS rough mask (消融实验基线)"
    elif args.mode == 'cmrs_predictor':
        use_cmrs = True
        use_memory_refinement = False
        use_predictor_refinement = True   # ★ 使用Predictor精炼
        rough_only = False
        mode_desc = "CMRS + SAM2 Predictor"
    elif args.mode == 'cmrs_memory':
        use_cmrs = True
        use_memory_refinement = True
        use_predictor_refinement = False  # Memory模式下不需要
        rough_only = False
        mode_desc = "CMRS + Memory机制 (完整SP-SAM)"
    elif args.mode == 'memory_only':
        use_cmrs = False
        use_memory_refinement = True
        use_predictor_refinement = False  # Memory模式下不需要
        rough_only = False
        mode_desc = "纯Memory机制"
    else:
        use_cmrs = True
        use_memory_refinement = False
        use_predictor_refinement = True   # 默认使用Predictor精炼
        rough_only = False
        mode_desc = "CMRS + SAM2 Predictor"
    
    print("\n" + "="*80)
    print("SP-SAM在COCO-20i上的评估")
    print("="*80)
    print(f"COCO-20i路径: {args.coco20i_root}")
    print(f"COCO根目录: {args.coco_root}")
    print(f"K-shot: {args.k_shot}")
    print(f"评估模式: {args.mode}")
    print(f"模式说明: {mode_desc}")
    print(f"使用排序Support: {'是' if args.use_ranked_support else '否'}")
    print(f"DINO模型: {args.dino_model}")
    print(f"SAM2模型: {args.sam2_model}")
    print(f"随机种子: {args.random_seed}")
    print(f"设备: {args.device}")
    if args.max_test_samples:
        print(f"最大测试样本数: {args.max_test_samples}")
    print("="*80 + "\n")
    
    # 1. 加载模型
    print("步骤1: 加载模型...")
    manager = ModelManager(device=args.device)
    
    # 加载DINO
    if args.dino_model.startswith('dinov3'):
        dino_model, dino_transform = manager.load_dinov3_model(args.dino_model)
    else:
        dino_model, dino_transform = manager.load_dinov2_model(args.dino_model)
    
    # 加载SAM2
    sam2_model, sam2_predictor, _ = manager.load_sam2_model(args.sam2_model)
    
    print("✅ 模型加载完成\n")
    
    # 2. 创建SP-SAM
    print("步骤2: 创建SP-SAM模型...")
    sp_sam = SPSAMModel(
        sam2_model=sam2_model,
        sam2_predictor=sam2_predictor,
        dino_model=dino_model,
        dino_transform=dino_transform,
        device=args.device,
        sam2_model_type=args.sam2_model
    )
    print("✅ SP-SAM创建完成\n")
    
    # 3. 创建评估器
    evaluator = SPSAMEvaluator(sp_sam, output_dir=args.output_dir)
    
    # 4. 运行评估
    if args.fold is not None:
        # 评估单个fold
        print(f"\n步骤3: 评估Fold {args.fold}...")
        
        dataset = COCO20iDataset(
            args.coco20i_root,
            args.coco_root,
            args.fold,
            use_ranked_support=args.use_ranked_support
        )
        
        results = evaluator.evaluate_fold(
            dataset,
            k_shot=args.k_shot,
            use_cmrs=use_cmrs,
            use_memory_refinement=use_memory_refinement,
            use_predictor_refinement=use_predictor_refinement,  # ★ 传递新参数
            rough_only=rough_only,
            visualize=args.visualize,
            max_test_samples=args.max_test_samples
        )
        
        if results:
            fold_mean_iou = np.mean([r['mean_iou'] for r in results])
            print(f"\nFold {args.fold} mIoU: {fold_mean_iou:.4f} ({fold_mean_iou*100:.2f}%)")
    else:
        # 评估所有folds
        print("\n步骤3: 评估所有4个Folds...")
        
        results = evaluator.evaluate_all_folds(
            args.coco20i_root,
            args.coco_root,
            k_shot=args.k_shot,
            use_cmrs=use_cmrs,
            use_memory_refinement=use_memory_refinement,
            use_predictor_refinement=use_predictor_refinement,  # ★ 传递新参数
            rough_only=rough_only,
            visualize=args.visualize,
            use_ranked_support=args.use_ranked_support,
            max_test_samples=args.max_test_samples
        )
    
    print("\n" + "="*80)
    print(f"✅ 评估完成！({args.k_shot}-Shot, 模式: {args.mode})")
    print("="*80)


if __name__ == '__main__':
    main()
"""
ISIC2018 皮肤病变分割评估脚本
=============================

用于评估SP-SAM在医学图像分割任务上的Few-shot性能

支持4种消融实验模式：
1. rough_only: 纯CMRS rough mask (基线)
2. cmrs_predictor: CMRS + SAM2 Predictor  
3. memory_only: 纯Memory机制
4. cmrs_memory: CMRS + Memory (完整SP-SAM)

使用方法：
    # 基本评估
    python evaluate_isic2018.py --data_root ISIC2018_256 --k_shot 5 --mode cmrs_memory
    
    # 快速测试（限制query数量）
    python evaluate_isic2018.py --data_root ISIC2018_256 --k_shot 5 --mode cmrs_memory --max_queries 100
    
    # 带可视化
    python evaluate_isic2018.py --data_root ISIC2018_256 --k_shot 5 --mode cmrs_memory --visualize --vis_num 20
    
    # K-Fold交叉验证
    python evaluate_isic2018.py --data_root ISIC2018_256 --k_shot 5 --mode cmrs_memory --k_fold 5
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

from isic2018_dataset import ISIC2018Dataset, ISIC2018FoldDataset


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU (Intersection over Union)"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    return float(intersection / union) if union > 0 else 0.0


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算Dice系数"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    
    return float(2 * intersection / total) if total > 0 else 0.0


def compute_precision(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算精确率"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    tp = (pred & gt).sum()
    pred_sum = pred.sum()
    
    return float(tp / pred_sum) if pred_sum > 0 else 0.0


def compute_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算召回率"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    tp = (pred & gt).sum()
    gt_sum = gt.sum()
    
    return float(tp / gt_sum) if gt_sum > 0 else 0.0


def create_overlay(img: Image.Image, mask: np.ndarray, 
                   color: tuple = (255, 0, 0), alpha: float = 0.5) -> Image.Image:
    """创建mask叠加图像"""
    img_array = np.array(img).copy()
    
    mask_bool = mask > 0
    for c, color_val in enumerate(color):
        img_array[:, :, c] = np.where(
            mask_bool,
            img_array[:, :, c] * (1 - alpha) + color_val * alpha,
            img_array[:, :, c]
        )
    
    return Image.fromarray(img_array.astype(np.uint8))


def visualize_prediction(support_imgs: List[Image.Image],
                        support_masks: List[np.ndarray],
                        query_img: Image.Image,
                        query_mask: np.ndarray,
                        pred_mask: np.ndarray,
                        sample_id: str,
                        metrics: Dict,
                        save_path: str):
    """
    可视化单个预测结果
    
    布局：
    Row 1: Support samples with masks
    Row 2: Query | GT Mask | Pred Mask | Comparison Overlay
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
        return
    
    k_shot = len(support_imgs)
    n_support_cols = min(k_shot, 4)  # 最多显示4个support
    
    # 计算布局
    fig = plt.figure(figsize=(16, 8))
    
    # 上半部分：Support samples
    for i in range(min(k_shot, n_support_cols)):
        # Support图像 + mask叠加
        ax = fig.add_subplot(2, n_support_cols, i + 1)
        overlay = create_overlay(support_imgs[i], support_masks[i], color=(0, 255, 0))
        ax.imshow(overlay)
        ax.set_title(f'Support {i+1}', fontsize=10)
        ax.axis('off')
    
    # 下半部分：Query结果
    # Query图像
    ax1 = fig.add_subplot(2, 4, 5)
    ax1.imshow(query_img)
    ax1.set_title('Query Image', fontsize=10)
    ax1.axis('off')
    
    # GT Mask
    ax2 = fig.add_subplot(2, 4, 6)
    ax2.imshow(query_mask, cmap='gray')
    ax2.set_title('GT Mask', fontsize=10)
    ax2.axis('off')
    
    # Pred Mask
    ax3 = fig.add_subplot(2, 4, 7)
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title(f'Pred Mask\nIoU: {metrics["iou"]*100:.1f}%', fontsize=10)
    ax3.axis('off')
    
    # 对比图：GT(绿色) vs Pred(红色) vs 重叠(黄色)
    ax4 = fig.add_subplot(2, 4, 8)
    query_array = np.array(query_img).copy().astype(float)
    gt_bool = query_mask > 0
    pred_bool = pred_mask > 0
    
    # 绿色=GT only, 红色=Pred only, 黄色=Both
    overlay_img = query_array.copy()
    # GT区域加绿色
    overlay_img[gt_bool & ~pred_bool, 1] = np.clip(overlay_img[gt_bool & ~pred_bool, 1] + 150, 0, 255)
    # Pred区域加红色  
    overlay_img[pred_bool & ~gt_bool, 0] = np.clip(overlay_img[pred_bool & ~gt_bool, 0] + 150, 0, 255)
    # 重叠区域加黄色
    overlay_img[gt_bool & pred_bool, 0] = np.clip(overlay_img[gt_bool & pred_bool, 0] + 100, 0, 255)
    overlay_img[gt_bool & pred_bool, 1] = np.clip(overlay_img[gt_bool & pred_bool, 1] + 100, 0, 255)
    
    ax4.imshow(overlay_img.astype(np.uint8))
    ax4.set_title('GT(G) vs Pred(R)\nOverlap(Y)', fontsize=10)
    ax4.axis('off')
    
    # 添加指标信息
    metrics_text = f"IoU: {metrics['iou']*100:.2f}%  Dice: {metrics['dice']*100:.2f}%  " \
                   f"Prec: {metrics['precision']*100:.2f}%  Recall: {metrics['recall']*100:.2f}%"
    
    plt.suptitle(f'ISIC2018: {sample_id}\n{metrics_text}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_visualization(all_metrics: List[Dict], save_path: str):
    """创建结果汇总可视化"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    # 提取指标
    ious = [m['iou'] for m in all_metrics]
    dices = [m['dice'] for m in all_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # IoU分布直方图
    axes[0, 0].hist(ious, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious)*100:.2f}%')
    axes[0, 0].set_xlabel('IoU')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('IoU Distribution')
    axes[0, 0].legend()
    
    # Dice分布直方图
    axes[0, 1].hist(dices, bins=20, color='forestgreen', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(np.mean(dices), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dices)*100:.2f}%')
    axes[0, 1].set_xlabel('Dice')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Dice Distribution')
    axes[0, 1].legend()
    
    # IoU vs Dice散点图
    axes[1, 0].scatter(ious, dices, alpha=0.5, s=20)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 0].set_xlabel('IoU')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].set_title('IoU vs Dice')
    
    # 按IoU排序的曲线
    sorted_ious = sorted(ious, reverse=True)
    axes[1, 1].plot(range(len(sorted_ious)), [x * 100 for x in sorted_ious], 'b-', linewidth=1)
    axes[1, 1].fill_between(range(len(sorted_ious)), [x * 100 for x in sorted_ious], alpha=0.3)
    axes[1, 1].axhline(np.mean(ious) * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious)*100:.2f}%')
    axes[1, 1].set_xlabel('Sample Index (sorted)')
    axes[1, 1].set_ylabel('IoU (%)')
    axes[1, 1].set_title('Sorted IoU Scores')
    axes[1, 1].legend()
    
    plt.suptitle('ISIC2018 Few-Shot Segmentation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class ISIC2018Evaluator:
    """ISIC2018评估器"""
    
    def __init__(self, data_root: str, device: str = 'cuda',
                 dino_model: str = 'dinov3_vitb16',
                 sam2_model: str = 'large'):
        """
        Args:
            data_root: ISIC2018数据集根目录
            device: 计算设备
            dino_model: DINO模型类型
            sam2_model: SAM2模型类型
        """
        self.data_root = Path(data_root)
        self.device = device
        self.dino_model_type = dino_model
        self.sam2_model_type = sam2_model
        
        self.sp_sam = None
        self.vis_count = 0
    
    def _load_models(self):
        """延迟加载模型"""
        if self.sp_sam is not None:
            return
        
        print(f"\n{'='*60}")
        print(f"🚀 加载模型...")
        print(f"{'='*60}")
        
        try:
            from sp_sam_complete import SPSAMModel
            from src.model_manager import ModelManager
            
            # 加载模型管理器
            model_manager = ModelManager(device=self.device)
            
            # 加载DINO模型（根据类型选择dinov2或dinov3）
            print(f"   加载DINO模型: {self.dino_model_type}")
            if 'dinov3' in self.dino_model_type.lower():
                dino_model, dino_transform = model_manager.load_dinov3_model(
                    dinov3_model_name=self.dino_model_type
                )
            else:
                dino_model, dino_transform = model_manager.load_dinov2_model(
                    dinov2_model_name=self.dino_model_type
                )
            
            # 加载SAM2模型
            print(f"   加载SAM2模型: {self.sam2_model_type}")
            sam2_model, sam2_predictor, _ = model_manager.load_sam2_model(
                sam2_model_type=self.sam2_model_type
            )
            
            # 创建SP-SAM模型
            self.sp_sam = SPSAMModel(
                sam2_model=sam2_model,
                sam2_predictor=sam2_predictor,
                dino_model=dino_model,
                dino_transform=dino_transform,
                device=self.device,
                sam2_model_type=self.sam2_model_type
            )
            
            print(f"✅ 模型加载完成")
            
        except ImportError as e:
            print(f"❌ 模型加载失败: {e}")
            print("请确保sp_sam_complete.py和src/model_manager.py在正确路径")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def predict_single(self, query_img: Image.Image,
                      support_imgs: List[Image.Image],
                      support_masks: List[np.ndarray],
                      mode: str) -> Optional[np.ndarray]:
        """
        单张图像预测
        
        Args:
            query_img: 查询图像
            support_imgs: support图像列表
            support_masks: support mask列表
            mode: 预测模式
            
        Returns:
            预测的mask
        """
        # 解析模式
        if mode == 'rough_only':
            use_cmrs = True
            use_memory = False
            use_predictor = False
        elif mode == 'cmrs_predictor':
            use_cmrs = True
            use_memory = False
            use_predictor = True
        elif mode == 'memory_only':
            use_cmrs = False
            use_memory = True
            use_predictor = False
        elif mode == 'cmrs_memory':
            use_cmrs = True
            use_memory = True
            use_predictor = False
        else:
            raise ValueError(f"未知模式: {mode}")
        
        # 调用SP-SAM
        results = self.sp_sam.predict(
            query_img,
            support_images=support_imgs,
            support_masks=support_masks,
            use_cmrs=use_cmrs,
            use_memory_refinement=use_memory,
            use_predictor_refinement=use_predictor
        )
        
        # 获取结果
        if mode == 'rough_only':
            pred_mask = results.get('rough_mask')
        else:
            pred_mask = results.get('final_mask')
            if pred_mask is None:
                pred_mask = results.get('rough_mask')
        
        return pred_mask
    
    def evaluate(self, k_shot: int, mode: str,
                support_ratio: float = 0.1,
                max_queries: int = None,
                random_seed: int = 42,
                output_dir: str = None,
                visualize: bool = False,
                vis_num: int = 20,
                k_fold: int = None) -> Dict:
        """
        评估ISIC2018数据集
        
        Args:
            k_shot: support数量
            mode: 评估模式
            support_ratio: support集合比例
            max_queries: 最大query数量
            random_seed: 随机种子
            output_dir: 输出目录
            visualize: 是否可视化
            vis_num: 最大可视化样本数
            k_fold: K-Fold交叉验证（None表示不使用）
            
        Returns:
            results: 评估结果
        """
        self._load_models()
        
        print(f"\n{'='*60}")
        print(f"📊 ISIC2018 Few-Shot 分割评估")
        print(f"{'='*60}")
        print(f"模式: {mode}")
        print(f"K-shot: {k_shot}")
        print(f"Support比例: {support_ratio*100:.1f}%")
        if max_queries:
            print(f"最大Query数: {max_queries}")
        print(f"可视化: {'是' if visualize else '否'}" + (f' (最多{vis_num}个)' if visualize else ''))
        if k_fold:
            print(f"K-Fold: {k_fold}折交叉验证")
        print(f"{'='*60}")
        
        # 创建输出目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建可视化目录
        vis_dir = None
        if visualize and output_dir:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 重置可视化计数器
        self.vis_count = 0
        
        # K-Fold或单次评估
        if k_fold and k_fold > 1:
            all_fold_results = []
            for fold_idx in range(k_fold):
                print(f"\n--- Fold {fold_idx + 1}/{k_fold} ---")
                dataset = ISIC2018FoldDataset(
                    self.data_root,
                    n_folds=k_fold,
                    current_fold=fold_idx,
                    random_seed=random_seed
                )
                fold_result = self._evaluate_dataset(
                    dataset, k_shot, mode, max_queries,
                    visualize, vis_dir, vis_num
                )
                all_fold_results.append(fold_result)
            
            # 汇总K-Fold结果
            final_results = self._aggregate_fold_results(all_fold_results, k_fold, mode, k_shot)
        else:
            dataset = ISIC2018Dataset(
                self.data_root,
                support_ratio=support_ratio,
                random_seed=random_seed
            )
            final_results = self._evaluate_dataset(
                dataset, k_shot, mode, max_queries,
                visualize, vis_dir, vis_num
            )
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"📈 最终结果")
        print(f"{'='*60}")
        print(f"Mean IoU:       {final_results['mean_iou']*100:.2f}%")
        print(f"Mean Dice:      {final_results['mean_dice']*100:.2f}%")
        print(f"Mean Precision: {final_results['mean_precision']*100:.2f}%")
        print(f"Mean Recall:    {final_results['mean_recall']*100:.2f}%")
        print(f"评估样本数:     {final_results['n_samples']}")
        
        # 保存结果
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f'isic2018_{k_shot}shot_{mode}_{timestamp}.json'
            
            with open(output_file, 'w') as f:
                # 转换numpy类型
                save_results = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in final_results.items() if k != 'sample_metrics'
                }
                json.dump(save_results, f, indent=2)
            
            print(f"\n✅ 结果保存到: {output_file}")
            
            # 创建汇总可视化
            if 'sample_metrics' in final_results and final_results['sample_metrics']:
                summary_path = output_dir / f'summary_{k_shot}shot_{mode}.png'
                create_summary_visualization(final_results['sample_metrics'], str(summary_path))
                print(f"✅ 汇总图表保存到: {summary_path}")
        
        return final_results
    
    def _evaluate_dataset(self, dataset, k_shot: int, mode: str,
                         max_queries: int, visualize: bool,
                         vis_dir: Path, vis_num: int) -> Dict:
        """评估单个数据集划分"""
        
        # 获取episode
        episode = dataset.get_episode(k_shot=k_shot, max_queries=max_queries)
        
        support_samples = episode['support']
        query_samples = episode['query']
        
        if len(support_samples) == 0:
            print("❌ 没有support样本")
            return {'mean_iou': 0, 'mean_dice': 0, 'n_samples': 0}
        
        if len(query_samples) == 0:
            print("❌ 没有query样本")
            return {'mean_iou': 0, 'mean_dice': 0, 'n_samples': 0}
        
        print(f"   Support样本: {len(support_samples)}")
        print(f"   Query样本: {len(query_samples)}")
        
        # 准备support数据
        support_imgs = [s['img'] for s in support_samples]
        support_masks = [s['mask'] for s in support_samples]
        
        # 评估每个query
        all_metrics = []
        
        for query in tqdm(query_samples, desc="Evaluating", leave=False):
            try:
                pred_mask = self.predict_single(
                    query['img'], support_imgs, support_masks, mode
                )
                
                if pred_mask is not None:
                    # 计算指标
                    metrics = {
                        'sample_id': query['sample_id'],
                        'iou': compute_iou(pred_mask, query['mask']),
                        'dice': compute_dice(pred_mask, query['mask']),
                        'precision': compute_precision(pred_mask, query['mask']),
                        'recall': compute_recall(pred_mask, query['mask'])
                    }
                    all_metrics.append(metrics)
                    
                    # 可视化
                    if visualize and vis_dir and self.vis_count < vis_num:
                        vis_path = vis_dir / f'{query["sample_id"]}_iou{metrics["iou"]*100:.1f}.png'
                        visualize_prediction(
                            support_imgs, support_masks,
                            query['img'], query['mask'], pred_mask,
                            query['sample_id'], metrics, str(vis_path)
                        )
                        self.vis_count += 1
                        
            except Exception as e:
                print(f"   ⚠️ 预测失败 {query['sample_id']}: {e}")
                continue
        
        # 计算平均指标
        if all_metrics:
            result = {
                'mean_iou': np.mean([m['iou'] for m in all_metrics]),
                'mean_dice': np.mean([m['dice'] for m in all_metrics]),
                'mean_precision': np.mean([m['precision'] for m in all_metrics]),
                'mean_recall': np.mean([m['recall'] for m in all_metrics]),
                'n_samples': len(all_metrics),
                'sample_metrics': all_metrics
            }
        else:
            result = {
                'mean_iou': 0, 'mean_dice': 0,
                'mean_precision': 0, 'mean_recall': 0,
                'n_samples': 0, 'sample_metrics': []
            }
        
        return result
    
    def _aggregate_fold_results(self, fold_results: List[Dict],
                               k_fold: int, mode: str, k_shot: int) -> Dict:
        """汇总K-Fold结果"""
        all_metrics = []
        for fr in fold_results:
            all_metrics.extend(fr.get('sample_metrics', []))
        
        return {
            'dataset': 'ISIC2018',
            'mode': mode,
            'k_shot': k_shot,
            'k_fold': k_fold,
            'mean_iou': np.mean([m['iou'] for m in all_metrics]) if all_metrics else 0,
            'mean_dice': np.mean([m['dice'] for m in all_metrics]) if all_metrics else 0,
            'mean_precision': np.mean([m['precision'] for m in all_metrics]) if all_metrics else 0,
            'mean_recall': np.mean([m['recall'] for m in all_metrics]) if all_metrics else 0,
            'std_iou': np.std([m['iou'] for m in all_metrics]) if all_metrics else 0,
            'std_dice': np.std([m['dice'] for m in all_metrics]) if all_metrics else 0,
            'n_samples': len(all_metrics),
            'sample_metrics': all_metrics
        }


def main():
    parser = argparse.ArgumentParser(description='ISIC2018 Few-Shot分割评估')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='ISIC2018数据集根目录')
    parser.add_argument('--support_ratio', type=float, default=0.1,
                       help='Support集合比例')
    parser.add_argument('--max_queries', type=int, default=None,
                       help='最大Query数量（用于快速测试）')
    
    # 模型参数
    parser.add_argument('--k_shot', type=int, default=5,
                       help='K-shot设置')
    parser.add_argument('--mode', type=str, default='cmrs_memory',
                       choices=['rough_only', 'cmrs_predictor', 'memory_only', 'cmrs_memory'],
                       help='评估模式')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型类型 (dinov2_vitb14, dinov3_vitb16等)')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型类型')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    # 评估参数
    parser.add_argument('--k_fold', type=int, default=None,
                       help='K-Fold交叉验证折数')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_dir', type=str, default='isic2018_results',
                       help='输出目录')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                       help='是否保存可视化结果')
    parser.add_argument('--vis_num', type=int, default=20,
                       help='最大可视化样本数量')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ISIC2018Evaluator(
        data_root=args.data_root,
        device=args.device,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model
    )
    
    # 评估
    results = evaluator.evaluate(
        k_shot=args.k_shot,
        mode=args.mode,
        support_ratio=args.support_ratio,
        max_queries=args.max_queries,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        visualize=args.visualize,
        vis_num=args.vis_num,
        k_fold=args.k_fold
    )


if __name__ == '__main__':
    main()
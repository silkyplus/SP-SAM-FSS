"""
FSS-1000 评估脚本
==================

支持4种消融实验模式：
1. rough_only: 纯CMRS rough mask (基线)
2. cmrs_predictor: CMRS + SAM2 Predictor  
3. memory_only: 纯Memory机制
4. cmrs_memory: CMRS + Memory (完整SP-SAM)

使用方法：
    # 评估所有类别
    python evaluate_fss1000.py --data_root fewshot_data --k_shot 1 --mode cmrs_memory
    
    # 评估前100个类别（快速测试）
    python evaluate_fss1000.py --data_root fewshot_data --k_shot 1 --mode cmrs_memory --max_classes 100
    
    # 带可视化
    python evaluate_fss1000.py --data_root fewshot_data --k_shot 1 --mode cmrs_memory --visualize
    
    # 可视化指定数量的样本
    python evaluate_fss1000.py --data_root fewshot_data --k_shot 1 --mode cmrs_memory --visualize --vis_num 50
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

from fss1000_dataset import FSS1000Dataset


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算Dice系数"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum())) if (pred.sum() + gt.sum()) > 0 else 0.0


def create_overlay(img: Image.Image, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5) -> Image.Image:
    """创建mask叠加图像"""
    img_array = np.array(img).copy()
    
    # 创建彩色mask
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
                        cls_name: str,
                        iou: float,
                        save_path: str):
    """
    可视化单个预测结果
    
    布局：
    | Support 1 | Support 1 Mask | Query | GT Mask | Pred Mask | Overlay |
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
        return
    
    k_shot = len(support_imgs)
    
    # 计算布局
    n_cols = 6  # Support, Support Mask, Query, GT, Pred, Overlay
    n_rows = max(1, k_shot)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 绘制Support
    for i in range(k_shot):
        # Support图像
        axes[i, 0].imshow(support_imgs[i])
        axes[i, 0].set_title(f'Support {i+1}')
        axes[i, 0].axis('off')
        
        # Support Mask叠加
        overlay = create_overlay(support_imgs[i], support_masks[i], color=(0, 255, 0))
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f'Support {i+1} Mask')
        axes[i, 1].axis('off')
    
    # 清空多余的support行
    for i in range(k_shot, n_rows):
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    
    # Query相关（只在第一行显示）
    # Query图像
    axes[0, 2].imshow(query_img)
    axes[0, 2].set_title('Query')
    axes[0, 2].axis('off')
    
    # GT Mask
    axes[0, 3].imshow(query_mask, cmap='gray')
    axes[0, 3].set_title('GT Mask')
    axes[0, 3].axis('off')
    
    # Pred Mask
    axes[0, 4].imshow(pred_mask, cmap='gray')
    axes[0, 4].set_title(f'Pred Mask\nIoU: {iou*100:.1f}%')
    axes[0, 4].axis('off')
    
    # Overlay对比
    # GT用绿色，Pred用红色
    query_array = np.array(query_img).copy()
    gt_bool = query_mask > 0
    pred_bool = pred_mask > 0
    
    # 绿色=GT, 红色=Pred, 黄色=重叠
    overlay_img = query_array.copy().astype(float)
    # GT区域加绿色
    overlay_img[gt_bool, 1] = np.clip(overlay_img[gt_bool, 1] + 100, 0, 255)
    # Pred区域加红色
    overlay_img[pred_bool, 0] = np.clip(overlay_img[pred_bool, 0] + 100, 0, 255)
    
    axes[0, 5].imshow(overlay_img.astype(np.uint8))
    axes[0, 5].set_title('GT(G) vs Pred(R)')
    axes[0, 5].axis('off')
    
    # 清空多余的行
    for i in range(1, n_rows):
        for j in range(2, 6):
            axes[i, j].axis('off')
    
    plt.suptitle(f'{cls_name} - IoU: {iou*100:.2f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_visualization(results: List[Dict], save_path: str, top_k: int = 20):
    """
    创建结果汇总可视化
    
    显示最好和最差的类别
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    # 按IoU排序
    sorted_results = sorted(results, key=lambda x: x['iou'], reverse=True)
    
    # 取最好和最差的
    best = sorted_results[:top_k]
    worst = sorted_results[-top_k:]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # 最好的类别
    classes = [r['class'][:15] for r in best]
    ious = [r['iou'] * 100 for r in best]
    
    axes[0].barh(range(len(classes)), ious, color='green', alpha=0.7)
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels(classes)
    axes[0].set_xlabel('IoU (%)')
    axes[0].set_title(f'Top {top_k} Classes')
    axes[0].invert_yaxis()
    
    for i, v in enumerate(ious):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    
    # 最差的类别
    classes = [r['class'][:15] for r in worst]
    ious = [r['iou'] * 100 for r in worst]
    
    axes[1].barh(range(len(classes)), ious, color='red', alpha=0.7)
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel('IoU (%)')
    axes[1].set_title(f'Bottom {top_k} Classes')
    axes[1].invert_yaxis()
    
    for i, v in enumerate(ious):
        axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class FSS1000Evaluator:
    """FSS-1000评估器"""
    
    def __init__(self, data_root: str, device: str = 'cuda',
                 dino_model: str = 'dinov3_vitb16',
                 sam2_model: str = 'large'):
        """
        Args:
            data_root: FSS-1000数据集根目录
            device: 计算设备
            dino_model: DINO模型类型
            sam2_model: SAM2模型类型
        """
        self.data_root = Path(data_root)
        self.device = device
        self.dino_model_type = dino_model
        self.sam2_model_type = sam2_model
        
        # 加载数据集
        self.dataset = FSS1000Dataset(data_root)
        
        self.sp_sam = None
        
        # 可视化计数器
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
            
            manager = ModelManager(device=self.device)
            
            # DINO
            if self.dino_model_type.startswith('dinov3'):
                dino, dino_transform = manager.load_dinov3_model(self.dino_model_type)
            else:
                dino, dino_transform = manager.load_dinov2_model(self.dino_model_type)
            
            # SAM2
            sam2_model_obj, sam2_predictor, _ = manager.load_sam2_model(self.sam2_model_type)
            
            # SP-SAM
            self.sp_sam = SPSAMModel(
                sam2_model=sam2_model_obj,
                sam2_predictor=sam2_predictor,
                dino_model=dino,
                dino_transform=dino_transform,
                device=self.device,
                sam2_model_type=self.sam2_model_type
            )
            
            print(f"✅ 模型加载完成\n")
            
        except ImportError as e:
            print(f"❌ 无法导入模型: {e}")
            raise
    
    def predict_single(self, query_img, support_images: List, 
                      support_masks: List, mode: str) -> np.ndarray:
        """
        单次预测
        
        Args:
            query_img: query图像
            support_images: support图像列表
            support_masks: support mask列表
            mode: 预测模式
            
        Returns:
            pred_mask: 预测的mask
            
        模式对应：
            - rough_only: use_cmrs=True, use_memory=False, use_predictor=False → 只返回rough_mask
            - cmrs_predictor: use_cmrs=True, use_memory=False, use_predictor=True → CMRS + SAM2 Predictor精炼
            - memory_only: use_cmrs=False, use_memory=True → 只用Memory机制
            - cmrs_memory: use_cmrs=True, use_memory=True → CMRS + Memory（完整SP-SAM）
        """
        # 解析模式
        if mode == 'rough_only':
            use_cmrs = True
            use_memory = False
            use_predictor = False
        elif mode == 'cmrs_predictor':
            use_cmrs = True
            use_memory = False
            use_predictor = True  # ★ 关键：使用SAM2 Predictor精炼
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
        
        # 预测
        results = self.sp_sam.predict(
            query_img=query_img,
            support_images=support_images,
            support_masks=support_masks,
            use_cmrs=use_cmrs,
            use_memory_refinement=use_memory,
            use_predictor_refinement=use_predictor  # ★ 传递新参数
        )
        
        # 获取结果
        if mode == 'rough_only':
            pred_mask = results.get('rough_mask')
        else:
            pred_mask = results.get('final_mask')
            if pred_mask is None:
                pred_mask = results.get('rough_mask')
        
        return pred_mask
    
    def evaluate_class(self, cls_name: str, k_shot: int, mode: str,
                      random_seed: int = 42,
                      visualize: bool = False,
                      vis_dir: str = None,
                      vis_num: int = None) -> Dict:
        """
        评估单个类别
        
        Args:
            cls_name: 类别名称
            k_shot: support数量
            mode: 评估模式
            random_seed: 随机种子
            visualize: 是否可视化
            vis_dir: 可视化保存目录
            vis_num: 最大可视化数量
            
        Returns:
            results: 评估结果
        """
        # 获取episode
        episode = self.dataset.get_episode(cls_name, k_shot, 
                                           random_select=False, 
                                           random_seed=random_seed)
        
        support_samples = episode['support']
        query_samples = episode['query']
        
        if len(support_samples) == 0:
            return {'class': cls_name, 'iou': 0.0, 'dice': 0.0, 'count': 0}
        
        if len(query_samples) == 0:
            return {'class': cls_name, 'iou': 0.0, 'dice': 0.0, 'count': 0}
        
        # 准备support数据
        support_imgs = [s['img'] for s in support_samples]
        support_masks = [s['mask'] for s in support_samples]
        
        # 评估
        ious = []
        dices = []
        
        for query in query_samples:
            try:
                pred_mask = self.predict_single(
                    query['img'], support_imgs, support_masks, mode
                )
                
                if pred_mask is not None:
                    iou = compute_iou(pred_mask, query['mask'])
                    dice = compute_dice(pred_mask, query['mask'])
                    ious.append(iou)
                    dices.append(dice)
                    
                    # 可视化
                    if visualize and vis_dir:
                        # 检查是否达到最大可视化数量
                        if vis_num is None or self.vis_count < vis_num:
                            vis_path = Path(vis_dir) / f'{cls_name}_{query["img_name"]}_iou{iou*100:.1f}.png'
                            visualize_prediction(
                                support_imgs, support_masks,
                                query['img'], query['mask'], pred_mask,
                                cls_name, iou, str(vis_path)
                            )
                            self.vis_count += 1
            except Exception as e:
                continue
        
        mean_iou = np.mean(ious) if ious else 0.0
        mean_dice = np.mean(dices) if dices else 0.0
        
        return {
            'class': cls_name,
            'iou': mean_iou,
            'dice': mean_dice,
            'count': len(ious)
        }
    
    def evaluate(self, k_shot: int, mode: str,
                max_classes: int = None,
                random_seed: int = 42,
                output_dir: str = None,
                visualize: bool = False,
                vis_num: int = 50) -> Dict:
        """
        评估整个数据集
        
        Args:
            k_shot: support数量
            mode: 评估模式
            max_classes: 最大评估类别数（用于快速测试）
            random_seed: 随机种子
            output_dir: 输出目录
            visualize: 是否可视化
            vis_num: 最大可视化样本数
            
        Returns:
            results: 评估结果
        """
        self._load_models()
        
        classes = self.dataset.classes
        if max_classes:
            classes = classes[:max_classes]
        
        print(f"\n{'='*60}")
        print(f"📊 FSS-1000 评估")
        print(f"{'='*60}")
        print(f"模式: {mode}")
        print(f"K-shot: {k_shot}")
        print(f"类别数: {len(classes)}")
        print(f"可视化: {'是' if visualize else '否'}" + (f' (最多{vis_num}个)' if visualize else ''))
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
            print(f"可视化目录: {vis_dir}")
        
        # 重置可视化计数器
        self.vis_count = 0
        
        # 评估每个类别
        class_results = []
        all_ious = []
        all_dices = []
        
        for cls_name in tqdm(classes, desc="Evaluating"):
            result = self.evaluate_class(
                cls_name, k_shot, mode, random_seed,
                visualize=visualize,
                vis_dir=vis_dir,
                vis_num=vis_num
            )
            class_results.append(result)
            
            if result['count'] > 0:
                all_ious.append(result['iou'])
                all_dices.append(result['dice'])
        
        # 计算总体指标
        mean_iou = np.mean(all_ious) if all_ious else 0.0
        mean_dice = np.mean(all_dices) if all_dices else 0.0
        
        print(f"\n{'='*60}")
        print(f"📈 结果")
        print(f"{'='*60}")
        print(f"Mean IoU: {mean_iou*100:.2f}%")
        print(f"Mean Dice: {mean_dice*100:.2f}%")
        print(f"有效类别: {len(all_ious)}/{len(classes)}")
        if visualize:
            print(f"已保存可视化: {self.vis_count} 个")
        
        results = {
            'dataset': 'FSS-1000',
            'mode': mode,
            'k_shot': k_shot,
            'n_classes': len(classes),
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'class_results': class_results
        }
        
        # 保存结果
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f'fss1000_{k_shot}shot_{mode}_{timestamp}.json'
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ 结果保存到: {output_file}")
            
            # 创建汇总可视化
            if visualize and class_results:
                summary_path = output_dir / f'summary_{k_shot}shot_{mode}.png'
                create_summary_visualization(class_results, str(summary_path))
                print(f"✅ 汇总图表保存到: {summary_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='FSS-1000评估')
    parser.add_argument('--data_root', type=str, required=True,
                       help='FSS-1000数据集根目录')
    parser.add_argument('--k_shot', type=int, default=1,
                       help='K-shot设置')
    parser.add_argument('--mode', type=str, default='cmrs_memory',
                       choices=['rough_only', 'cmrs_predictor', 'memory_only', 'cmrs_memory'],
                       help='评估模式')
    parser.add_argument('--max_classes', type=int, default=None,
                       help='最大评估类别数（用于快速测试）')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型类型')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型类型')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_dir', type=str, default='fss1000_results',
                       help='输出目录')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                       help='是否保存可视化结果')
    parser.add_argument('--vis_num', type=int, default=50,
                       help='最大可视化样本数量')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = FSS1000Evaluator(
        data_root=args.data_root,
        device=args.device,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model
    )
    
    # 评估
    results = evaluator.evaluate(
        k_shot=args.k_shot,
        mode=args.mode,
        max_classes=args.max_classes,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        visualize=args.visualize,
        vis_num=args.vis_num
    )


if __name__ == '__main__':
    main()
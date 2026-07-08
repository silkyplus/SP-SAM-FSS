"""
Pascal-5i 评估脚本（新格式）
============================

适配新生成的Pascal-5i数据集格式

支持4种消融实验模式：
1. rough_only: 纯CMRS rough mask (基线)
2. cmrs_predictor: CMRS + SAM2 Predictor
3. memory_only: 纯Memory机制
4. cmrs_memory: CMRS + Memory (完整SP-SAM)

使用方法：
    # 评估单个fold
    python evaluate_pascal5i_new.py --data_root pascal5i_output --fold 0 --k_shot 1 --mode cmrs_memory
    
    # 评估所有folds
    python evaluate_pascal5i_new.py --data_root pascal5i_output --k_shot 1 --mode cmrs_memory
    
    # 使用预选择的support列表
    python evaluate_pascal5i_new.py --data_root pascal5i_output --fold 0 --k_shot 1 --sorted_support_dir pascal5i_output/fold0/class_lists_sorted
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

from pascal5i_dataset_new import Pascal5iDatasetNew


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


class Pascal5iEvaluator:
    """Pascal-5i评估器"""
    
    def __init__(self, data_root: str, device: str = 'cuda',
                 dino_model: str = 'dinov3_vitb16',
                 sam2_model: str = 'large'):
        """
        Args:
            data_root: Pascal-5i数据集根目录
            device: 计算设备
            dino_model: DINO模型类型
            sam2_model: SAM2模型类型
        """
        self.data_root = Path(data_root)
        self.device = device
        self.dino_model_type = dino_model
        self.sam2_model_type = sam2_model
        
        self.sp_sam = None
    
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
    
    def evaluate_class(self, dataset: Pascal5iDatasetNew, class_name: str,
                      k_shot: int, mode: str,
                      sorted_support_list: List[str] = None,
                      random_seed: int = 42,
                      max_query: int = None) -> Dict:
        """
        评估单个类别
        
        Args:
            dataset: 数据集
            class_name: 类别名称
            k_shot: support数量
            mode: 评估模式
            sorted_support_list: 预排序的support列表
            random_seed: 随机种子
            max_query: 最大query数量
            
        Returns:
            results: 评估结果
        """
        # 获取support样本
        support_samples = dataset.get_support_samples(
            class_name, k_shot, random_seed, sorted_support_list
        )
        
        if len(support_samples) == 0:
            print(f"   ⚠️ {class_name}: 没有support样本")
            return {'class': class_name, 'iou': 0.0, 'count': 0}
        
        # 获取query样本（排除support）
        support_ids = [s['img_name'] for s in support_samples]
        query_samples = dataset.get_query_samples(class_name, exclude_ids=support_ids)
        
        if len(query_samples) == 0:
            print(f"   ⚠️ {class_name}: 没有query样本")
            return {'class': class_name, 'iou': 0.0, 'count': 0}
        
        # 限制query数量
        if max_query is not None and len(query_samples) > max_query:
            query_samples = query_samples[:max_query]
        
        # 准备support数据
        support_imgs = [s['img'] for s in support_samples]
        support_masks = [s['mask'] for s in support_samples]
        
        # 评估
        ious = []
        for query in tqdm(query_samples, desc=f"   {class_name}", leave=False):
            try:
                pred_mask = self.predict_single(
                    query['img'], support_imgs, support_masks, mode
                )
                
                if pred_mask is not None:
                    iou = compute_iou(pred_mask, query['mask'])
                    ious.append(iou)
            except Exception as e:
                print(f"   ⚠️ 预测错误 {query['img_name']}: {e}")
                continue
        
        mean_iou = np.mean(ious) if ious else 0.0
        
        return {
            'class': class_name,
            'iou': mean_iou,
            'count': len(ious),
            'support_ids': support_ids
        }
    
    def evaluate_fold(self, fold: int, k_shot: int, mode: str,
                     sorted_support_dir: str = None,
                     random_seed: int = 42,
                     max_query: int = None) -> Dict:
        """
        评估单个fold
        
        Args:
            fold: fold编号
            k_shot: support数量
            mode: 评估模式
            sorted_support_dir: 预排序support文件目录
            random_seed: 随机种子
            max_query: 每个类别最大query数量
            
        Returns:
            results: 评估结果
        """
        self._load_models()
        
        # 加载数据集
        dataset = Pascal5iDatasetNew(self.data_root, fold)
        
        print(f"\n{'='*60}")
        print(f"📊 评估 Fold {fold} ({k_shot}-shot, 模式: {mode})")
        print(f"{'='*60}")
        
        # 加载预排序的support列表（如果有）
        sorted_lists = {}
        if sorted_support_dir:
            sorted_dir = Path(sorted_support_dir)
            if sorted_dir.exists():
                for cls in dataset.test_classes:
                    txt_path = sorted_dir / f'{cls}.txt'
                    if txt_path.exists():
                        with open(txt_path, 'r') as f:
                            sorted_lists[cls] = [l.strip() for l in f if l.strip()]
                        print(f"   ✅ 加载预排序列表: {cls} ({len(sorted_lists[cls])} items)")
        
        # 评估每个类别
        class_results = []
        for cls in dataset.test_classes:
            sorted_list = sorted_lists.get(cls)
            result = self.evaluate_class(
                dataset, cls, k_shot, mode,
                sorted_support_list=sorted_list,
                random_seed=random_seed,
                max_query=max_query
            )
            class_results.append(result)
            print(f"   {cls}: IoU = {result['iou']*100:.2f}% ({result['count']} samples)")
        
        # 计算平均
        mean_iou = np.mean([r['iou'] for r in class_results])
        
        print(f"\n   📈 Fold {fold} Mean IoU: {mean_iou*100:.2f}%")
        
        return {
            'fold': fold,
            'k_shot': k_shot,
            'mode': mode,
            'mean_iou': mean_iou,
            'class_results': class_results
        }
    
    def evaluate_all_folds(self, k_shot: int, mode: str,
                          sorted_support_base_dir: str = None,
                          random_seed: int = 42,
                          max_query: int = None) -> Dict:
        """
        评估所有folds
        
        Args:
            k_shot: support数量
            mode: 评估模式
            sorted_support_base_dir: 预排序support文件的基础目录
            random_seed: 随机种子
            max_query: 每个类别最大query数量
            
        Returns:
            results: 评估结果
        """
        all_results = []
        
        for fold in range(4):
            # 构建排序support目录路径
            sorted_dir = None
            if sorted_support_base_dir:
                sorted_dir = Path(sorted_support_base_dir) / f'fold{fold}' / 'class_lists_sorted'
                if not sorted_dir.exists():
                    sorted_dir = None
            
            result = self.evaluate_fold(
                fold, k_shot, mode,
                sorted_support_dir=sorted_dir,
                random_seed=random_seed,
                max_query=max_query
            )
            all_results.append(result)
        
        # 计算总体平均
        overall_mean_iou = np.mean([r['mean_iou'] for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"📊 Overall Results ({k_shot}-shot, 模式: {mode})")
        print(f"{'='*60}")
        for r in all_results:
            print(f"   Fold {r['fold']}: {r['mean_iou']*100:.2f}%")
        print(f"   ────────────────────")
        print(f"   Overall Mean IoU: {overall_mean_iou*100:.2f}%")
        print(f"{'='*60}")
        
        return {
            'k_shot': k_shot,
            'mode': mode,
            'overall_mean_iou': overall_mean_iou,
            'fold_results': all_results
        }


def main():
    parser = argparse.ArgumentParser(description='Pascal-5i评估（新格式）')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Pascal-5i数据集根目录')
    parser.add_argument('--fold', type=int, default=None,
                       help='评估特定fold (0-3)，不指定则评估所有folds')
    parser.add_argument('--k_shot', type=int, default=1,
                       help='K-shot设置')
    parser.add_argument('--mode', type=str, default='cmrs_memory',
                       choices=['rough_only', 'cmrs_predictor', 'memory_only', 'cmrs_memory'],
                       help='评估模式')
    parser.add_argument('--sorted_support_dir', type=str, default=None,
                       help='预排序support文件目录')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型类型')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型类型')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--max_query', type=int, default=None,
                       help='每个类别最大query数量')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = Pascal5iEvaluator(
        data_root=args.data_root,
        device=args.device,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model
    )
    
    # 评估
    if args.fold is not None:
        results = evaluator.evaluate_fold(
            fold=args.fold,
            k_shot=args.k_shot,
            mode=args.mode,
            sorted_support_dir=args.sorted_support_dir,
            random_seed=args.random_seed,
            max_query=args.max_query
        )
    else:
        results = evaluator.evaluate_all_folds(
            k_shot=args.k_shot,
            mode=args.mode,
            sorted_support_base_dir=args.sorted_support_dir,
            random_seed=args.random_seed,
            max_query=args.max_query
        )
    
    # 保存结果
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.fold is not None:
            output_file = output_dir / f'fold{args.fold}_{args.k_shot}shot_{args.mode}_{timestamp}.json'
        else:
            output_file = output_dir / f'all_folds_{args.k_shot}shot_{args.mode}_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 结果保存到: {output_file}")


if __name__ == '__main__':
    main()
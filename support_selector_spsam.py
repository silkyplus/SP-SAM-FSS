"""
Support直接评估选择工具 - 调用真实SP-SAM模型
=============================================

核心思路：用每个候选support实际跑SP-SAM，测量真实mIoU，选最好的

使用方法：
    python support_selector_spsam.py --data_root pascal-5 --fold 0 --generate_txt
    python support_selector_spsam.py --data_root pascal-5 --fold 1 --generate_txt
    python support_selector_spsam.py --data_root pascal-5 --fold 2 --generate_txt
    python support_selector_spsam.py --data_root pascal-5 --fold 3 --generate_txt
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import argparse
import json

# 导入你的模块
from pascal5i_dataset_filled import Pascal5iDatasetFilled as Pascal5iDataset
from sp_sam_complete import SPSAMModel
from src.model_manager import ModelManager


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


class SPSAMSupportSelector:
    """使用真实SP-SAM模型评估support效果"""
    
    FOLD_CLASSES = {
        0: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
        1: ['bus', 'car', 'cat', 'chair', 'cow'],
        2: ['diningtable', 'dog', 'horse', 'motorbike', 'person'],
        3: ['pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    }
    
    def __init__(self, data_root: str, fold: int, device: str = 'cuda',
                 dino_model: str = 'dinov3_vitb16', sam2_model: str = 'large'):
        self.data_root = Path(data_root)
        self.fold = fold
        self.device = device
        self.classes = self.FOLD_CLASSES[fold]
        
        # 加载数据集
        self.dataset = Pascal5iDataset(data_root, fold)
        
        # 加载模型
        print(f"\n{'='*50}")
        print(f"🚀 加载SP-SAM模型...")
        print(f"{'='*50}")
        
        manager = ModelManager(device=device)
        
        # DINO
        if dino_model.startswith('dinov3'):
            dino, dino_transform = manager.load_dinov3_model(dino_model)
        else:
            dino, dino_transform = manager.load_dinov2_model(dino_model)
        
        # SAM2
        sam2_model_obj, sam2_predictor, _ = manager.load_sam2_model(sam2_model)
        
        # SP-SAM
        self.sp_sam = SPSAMModel(
            sam2_model=sam2_model_obj,
            sam2_predictor=sam2_predictor,
            dino_model=dino,
            dino_transform=dino_transform,
            device=device,
            sam2_model_type=sam2_model
        )
        
        print(f"✅ 模型加载完成\n")
    
    def evaluate_support(self, support_sample: Dict, test_samples: List[Dict],
                        use_cmrs: bool = True, use_memory: bool = False) -> float:
        """
        评估单个support在多个test上的效果
        
        Args:
            support_sample: {'img': PIL.Image, 'mask': np.ndarray, ...}
            test_samples: List of test samples
            
        Returns:
            mean_iou: 平均IoU
        """
        support_imgs = [support_sample['img']]
        support_masks = [support_sample['mask']]
        
        ious = []
        for test in test_samples:
            try:
                results = self.sp_sam.predict(
                    query_img=test['img'],
                    support_images=support_imgs,
                    support_masks=support_masks,
                    use_cmrs=use_cmrs,
                    use_memory_refinement=use_memory
                )
                
                pred_mask = results.get('final_mask')
                if pred_mask is None:
                    pred_mask = results.get('rough_mask')
                
                if pred_mask is not None:
                    iou = compute_iou(pred_mask, test['mask'])
                    ious.append(iou)
            except Exception as e:
                # 静默跳过错误
                continue
        
        return np.mean(ious) if ious else 0.0
    
    def select_best_supports(self, class_name: str, 
                            n_test: int = 10,
                            top_k: int = 10,
                            use_cmrs: bool = True,
                            use_memory: bool = False) -> List[Dict]:
        """
        为某个类别选择最佳support
        
        Args:
            class_name: 类别名
            n_test: 用多少test样本评估
            top_k: 选择top-K个support
            
        Returns:
            ranked_supports: 排序后的support列表
        """
        # 加载所有train样本作为候选
        txt_path = self.dataset.train_dir / f'{class_name}.txt'
        if not txt_path.exists():
            return []
        
        with open(txt_path, 'r') as f:
            train_names = [l.strip() for l in f if l.strip()]
        
        # 加载test样本
        test_samples = self.dataset.get_test_samples(class_name)
        if len(test_samples) == 0:
            return []
        
        # 采样test
        np.random.shuffle(test_samples)
        test_subset = test_samples[:min(n_test, len(test_samples))]
        
        print(f"\n📊 {class_name}: {len(train_names)} candidates, {len(test_subset)} test samples")
        
        # 评估每个候选support
        results = []
        for name in tqdm(train_names, desc=f"   Evaluating {class_name}"):
            try:
                img, mask = self.dataset._load_image_and_mask('train', class_name, name)
                if mask.sum() < 100:  # 跳过空mask
                    continue
                
                support = {'img': img, 'mask': mask, 'img_name': name}
                iou = self.evaluate_support(support, test_subset, use_cmrs, use_memory)
                
                results.append({
                    'name': name,
                    'iou': iou
                })
            except Exception as e:
                continue
        
        # 按IoU排序
        results.sort(key=lambda x: x['iou'], reverse=True)
        
        # 打印Top-K
        print(f"\n   🏆 Top-{min(top_k, len(results))} supports:")
        for i, r in enumerate(results[:top_k], 1):
            print(f"      {i}. {r['name']}: IoU={r['iou']*100:.2f}%")
        
        return results
    
    def process_fold(self, n_test: int = 10, top_k: int = 10,
                    use_cmrs: bool = True, use_memory: bool = False,
                    output_file: str = None) -> Dict:
        """处理整个fold"""
        print(f"\n{'='*60}")
        print(f"🔍 处理Fold {self.fold}: {', '.join(self.classes)}")
        print(f"{'='*60}")
        
        all_results = {}
        for cls in self.classes:
            results = self.select_best_supports(
                cls, n_test=n_test, top_k=top_k,
                use_cmrs=use_cmrs, use_memory=use_memory
            )
            if results:
                all_results[cls] = results
        
        # 保存JSON
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'fold': self.fold,
                    'n_test': n_test,
                    'results': all_results
                }, f, indent=2)
            print(f"\n✅ 结果保存到: {output_file}")
        
        return all_results
    
    def generate_txt(self, all_results: Dict, output_dir: str = None):
        """生成排序后的txt文件"""
        if output_dir is None:
            output_dir = self.data_root / str(self.fold) / 'train_sorted_spsam'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📝 生成txt文件到: {output_dir}")
        
        for cls, results in all_results.items():
            txt_path = output_dir / f'{cls}.txt'
            with open(txt_path, 'w') as f:
                for r in results:
                    f.write(f"{r['name']}\n")
            print(f"   ✅ {cls}.txt ({len(results)} samples)")
        
        print(f"\n💡 使用方法：将 {output_dir} 中的txt复制到 train/ 目录")


def main():
    parser = argparse.ArgumentParser(description='SP-SAM直接评估选择Support')
    parser.add_argument('--data_root', type=str, default='pascal-5')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--n_test', type=int, default=10, help='评估用的test样本数')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--dino_model', type=str, default='dinov3_vitl16')
    parser.add_argument('--sam2_model', type=str, default='large')
    parser.add_argument('--mode', type=str, default='cmrs_memory',
                       choices=['cmrs_predictor', 'cmrs_memory'],
                       help='SP-SAM模式')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--generate_txt', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # 解析模式
    use_cmrs = True
    use_memory = args.mode == 'cmrs_memory'
    
    # 创建选择器
    selector = SPSAMSupportSelector(
        data_root=args.data_root,
        fold=args.fold,
        device=args.device,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model
    )
    
    # 输出文件
    if args.output is None:
        args.output = f'support_spsam_fold{args.fold}.json'
    
    # 处理
    results = selector.process_fold(
        n_test=args.n_test,
        top_k=args.top_k,
        use_cmrs=use_cmrs,
        use_memory=use_memory,
        output_file=args.output
    )
    
    # 生成txt
    if args.generate_txt:
        selector.generate_txt(results)


if __name__ == '__main__':
    main()

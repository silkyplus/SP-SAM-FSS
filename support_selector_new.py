"""
Support全面搜索选择器
======================

从训练集中选择最优support，在验证集上评估

策略：
- Support候选：从训练集(train)中选择
- Query评估：使用验证集(val)中的所有样本
- 输出：按IoU排序的txt文件

使用方法：
    # 全面搜索（所有train候选 vs 所有val query）
    python support_selector_exhaustive.py --data_root pascal5i_output --fold 0 --generate_txt
    
    # 限制最大候选数（加速）
    python support_selector_exhaustive.py --data_root pascal5i_output --fold 0 --max_candidates 500 --generate_txt
    
    # 两阶段搜索（先用DINO特征预筛选，再用SP-SAM精确评估）
    python support_selector_exhaustive.py --data_root pascal5i_output --fold 0 --prefilter 100 --generate_txt
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime

from pascal5i_dataset_new import Pascal5iDatasetNew


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算IoU"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 0.0


class ExhaustiveSupportSelector:
    """
    穷举式Support选择器
    
    全面评估所有候选support与所有query的组合
    """
    
    def __init__(self, data_root: str, fold: int, device: str = 'cuda',
                 dino_model: str = 'dinov3_vitb16', sam2_model: str = 'large'):
        """
        Args:
            data_root: Pascal-5i数据集根目录
            fold: fold编号 (0-3)
            device: 计算设备
            dino_model: DINO模型类型
            sam2_model: SAM2模型类型
        """
        self.data_root = Path(data_root)
        self.fold = fold
        self.device = device
        self.dino_model_type = dino_model
        self.sam2_model_type = sam2_model
        
        # 加载数据集
        self.dataset = Pascal5iDatasetNew(data_root, fold, use_fewshot_val=True)
        self.test_classes = self.dataset.test_classes
        
        # 模型（延迟加载）
        self.sp_sam = None
        self.dino_model = None
        self.dino_transform = None
        
        # 缓存
        self.feature_cache = {}
    
    def _load_dino_model(self):
        """加载DINO模型（用于预筛选）"""
        if self.dino_model is not None:
            return
        
        print(f"\n🔄 加载DINO模型: {self.dino_model_type}")
        
        try:
            from src.model_manager import ModelManager
            manager = ModelManager(device=self.device)
            
            if self.dino_model_type.startswith('dinov3'):
                self.dino_model, self.dino_transform = manager.load_dinov3_model(self.dino_model_type)
            else:
                self.dino_model, self.dino_transform = manager.load_dinov2_model(self.dino_model_type)
            
            self.dino_model.eval()
            print(f"✅ DINO模型加载完成")
        except Exception as e:
            print(f"⚠️ DINO模型加载失败: {e}")
            self.dino_model = None
    
    def _load_spsam_model(self):
        """加载SP-SAM模型"""
        if self.sp_sam is not None:
            return
        
        print(f"\n🔄 加载SP-SAM模型...")
        
        try:
            from sp_sam_complete import SPSAMModel
            from src.model_manager import ModelManager
            
            manager = ModelManager(device=self.device)
            
            # DINO
            if self.dino_model is None:
                if self.dino_model_type.startswith('dinov3'):
                    dino, dino_transform = manager.load_dinov3_model(self.dino_model_type)
                else:
                    dino, dino_transform = manager.load_dinov2_model(self.dino_model_type)
            else:
                dino, dino_transform = self.dino_model, self.dino_transform
            
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
            
            print(f"✅ SP-SAM模型加载完成")
        except Exception as e:
            print(f"❌ SP-SAM模型加载失败: {e}")
            raise
    
    def _extract_dino_feature(self, img: Image.Image, mask: np.ndarray = None) -> torch.Tensor:
        """
        提取DINO特征
        
        Args:
            img: PIL图像
            mask: 可选的mask，用于提取前景区域特征
            
        Returns:
            feature: 特征向量
        """
        self._load_dino_model()
        
        if self.dino_model is None:
            return None
        
        with torch.no_grad():
            img_tensor = self.dino_transform(img).unsqueeze(0).to(self.device)
            features = self.dino_model(img_tensor)
            
            # 全局特征
            if isinstance(features, dict):
                feat = features.get('x_norm_clstoken', features.get('x_norm_patchtokens'))
                if feat is not None:
                    feat = feat.mean(dim=1) if feat.dim() == 3 else feat
            else:
                feat = features.mean(dim=1) if features.dim() == 3 else features
            
            feat = F.normalize(feat, dim=-1)
            
        return feat.cpu()
    
    def _compute_feature_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """计算特征相似度"""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(F.cosine_similarity(feat1, feat2, dim=-1).item())
    
    def load_all_samples(self, class_name: str, 
                        max_candidates: int = None) -> Tuple[List[Dict], List[Dict]]:
        """
        加载某个类别的所有样本
        
        Support候选：从训练集(train)加载
        Query：从验证集(val)加载
        
        Args:
            class_name: 类别名称
            max_candidates: 最大候选数量（None表示全部）
            
        Returns:
            (support_candidates, query_samples): 候选support样本和query样本
        """
        class_id = self.dataset._name_to_id(class_name)
        
        print(f"\n📦 加载类别 {class_name} (ID={class_id}) 的样本...")
        
        # 1. 从训练集加载候选Support
        support_candidates = self._load_train_samples_for_class(class_name)
        print(f"   ✅ 训练集: {len(support_candidates)} 个候选Support")
        
        # 2. 从验证集加载Query
        query_samples = self._load_val_samples_for_class(class_name)
        print(f"   ✅ 验证集: {len(query_samples)} 个Query")
        
        # 如果限制了候选数量，随机采样
        if max_candidates and len(support_candidates) > max_candidates:
            random.shuffle(support_candidates)
            support_candidates = support_candidates[:max_candidates]
            print(f"   📉 Support限制为 {max_candidates} 个候选")
        
        return support_candidates, query_samples
    
    def _load_val_samples_for_class(self, class_name: str) -> List[Dict]:
        """从验证集加载指定类别的样本作为Query"""
        val_image_ids = self.dataset.val_class_to_images.get(class_name, [])
        
        query_samples = []
        for img_id in tqdm(val_image_ids, desc=f"   Loading val", leave=False):
            try:
                img, mask = self.dataset._load_image_and_mask('val', class_name, img_id)
                if mask.sum() > 100:
                    query_samples.append({
                        'img': img,
                        'mask': mask,
                        'img_name': img_id,
                        'class': class_name,
                        'source': 'val'
                    })
            except Exception as e:
                continue
        
        return query_samples
    
    def _load_train_samples_for_class(self, class_name: str) -> List[Dict]:
        """
        从训练集加载包含指定类别的样本作为候选Support
        
        使用masks_raw（保留了所有类别，包括测试类别）
        """
        class_id = self.dataset._name_to_id(class_name)
        train_samples = []
        
        # 遍历训练集信息，找到包含该类别的图像
        for item in tqdm(self.dataset.train_info, desc=f"   Loading train", leave=False):
            # 检查该图像是否包含目标类别
            all_classes = item.get('classes', [])
            if class_id not in all_classes:
                continue
            
            image_id = item['image_id']
            
            try:
                # 加载图像
                img = self.dataset._load_image('train', image_id)
                
                # 加载mask（使用masks_raw，保留测试类别）
                mask_dir = self.dataset.train_dir / 'masks_raw'
                if not mask_dir.exists():
                    mask_dir = self.dataset.train_dir / 'masks'
                
                mask_path = mask_dir / f'{image_id}.png'
                if not mask_path.exists():
                    continue
                
                full_mask = np.array(Image.open(mask_path))
                
                # 提取该类别的二值mask
                binary_mask = (full_mask == class_id).astype(np.uint8)
                
                if binary_mask.sum() > 100:
                    train_samples.append({
                        'img': img,
                        'mask': binary_mask,
                        'img_name': image_id,
                        'class': class_name,
                        'source': 'train'
                    })
            except Exception as e:
                continue
        
        return train_samples
    
    def prefilter_by_feature(self, samples: List[Dict], 
                            top_k: int = 100,
                            use_diversity: bool = True) -> List[Dict]:
        """
        使用特征相似度预筛选候选support
        
        策略：选择与其他样本平均相似度最高的样本（代表性强）
        同时考虑多样性，避免选择太相似的样本
        
        Args:
            samples: 候选support样本（来自训练集）
            top_k: 选择top-K个
            use_diversity: 是否考虑多样性
            
        Returns:
            filtered_samples: 筛选后的样本
        """
        if len(samples) <= top_k:
            return samples
        
        print(f"\n🔍 特征预筛选: {len(samples)} -> {top_k}")
        
        # 提取所有特征
        features = []
        valid_samples = []
        
        for sample in tqdm(samples, desc="   Extracting features"):
            feat = self._extract_dino_feature(sample['img'])
            if feat is not None:
                features.append(feat)
                valid_samples.append(sample)
        
        if len(features) == 0:
            print("   ⚠️ 无法提取特征，返回原始样本")
            return samples[:top_k]
        
        # 计算相似度矩阵
        features = torch.cat(features, dim=0)  # [N, D]
        sim_matrix = torch.mm(features, features.t())  # [N, N]
        
        # 计算每个样本的平均相似度（代表性得分）
        # 排除自身
        sim_matrix.fill_diagonal_(0)
        avg_sim = sim_matrix.mean(dim=1)  # [N]
        
        if use_diversity:
            # 贪心选择：每次选择与已选样本最不相似、但与整体最相似的样本
            selected_indices = []
            remaining = set(range(len(valid_samples)))
            
            # 首先选择平均相似度最高的样本
            first_idx = avg_sim.argmax().item()
            selected_indices.append(first_idx)
            remaining.remove(first_idx)
            
            for _ in tqdm(range(min(top_k - 1, len(remaining))), desc="   Diverse selection"):
                if not remaining:
                    break
                
                best_idx = None
                best_score = -float('inf')
                
                for idx in remaining:
                    # 与已选样本的最大相似度（越小越好 -> 多样性）
                    max_sim_to_selected = max(sim_matrix[idx, s].item() for s in selected_indices)
                    # 综合得分：代表性 - λ * 最大相似度
                    score = avg_sim[idx].item() - 0.5 * max_sim_to_selected
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining.remove(best_idx)
            
            filtered_samples = [valid_samples[i] for i in selected_indices]
        else:
            # 简单按平均相似度排序
            sorted_indices = avg_sim.argsort(descending=True)[:top_k]
            filtered_samples = [valid_samples[i] for i in sorted_indices]
        
        print(f"   ✅ 预筛选完成: {len(filtered_samples)} 个候选")
        return filtered_samples
    
    def evaluate_support_exhaustive(self, support_sample: Dict, 
                                   query_samples: List[Dict],
                                   use_cmrs: bool = True,
                                   use_memory: bool = True,
                                   progress_bar: bool = False) -> Tuple[float, List[float]]:
        """
        评估单个support在所有query上的效果
        
        Args:
            support_sample: support样本（来自训练集）
            query_samples: 所有query样本（来自验证集）
            use_cmrs: 是否使用CMRS
            use_memory: 是否使用Memory
            progress_bar: 是否显示进度条
            
        Returns:
            (mean_iou, all_ious): 平均IoU和每个query的IoU
        """
        self._load_spsam_model()
        
        support_imgs = [support_sample['img']]
        support_masks = [support_sample['mask']]
        
        ious = []
        iterator = tqdm(query_samples, leave=False) if progress_bar else query_samples
        
        for query in iterator:
            try:
                results = self.sp_sam.predict(
                    query_img=query['img'],
                    support_images=support_imgs,
                    support_masks=support_masks,
                    use_cmrs=use_cmrs,
                    use_memory_refinement=use_memory
                )
                
                pred_mask = results.get('final_mask')
                if pred_mask is None:
                    pred_mask = results.get('rough_mask')
                
                if pred_mask is not None:
                    iou = compute_iou(pred_mask, query['mask'])
                    ious.append(iou)
            except Exception as e:
                continue
        
        mean_iou = np.mean(ious) if ious else 0.0
        return mean_iou, ious
    
    def exhaustive_search(self, class_name: str,
                         max_candidates: int = None,
                         max_query: int = None,
                         prefilter_k: int = None,
                         use_cmrs: bool = True,
                         use_memory: bool = True,
                         top_k: int = 20) -> List[Dict]:
        """
        穷举搜索最优support
        
        Support从训练集选择，Query从验证集选择
        
        Args:
            class_name: 类别名称
            max_candidates: 最大候选support数量
            max_query: 最大query数量（None表示全部）
            prefilter_k: 预筛选数量（None表示不预筛选）
            use_cmrs: 是否使用CMRS
            use_memory: 是否使用Memory
            top_k: 返回top-K个结果
            
        Returns:
            ranked_supports: 排序后的support列表
        """
        # 加载样本：support从train，query从val
        candidates, query_samples = self.load_all_samples(class_name, max_candidates)
        
        if len(candidates) == 0:
            print(f"   ⚠️ 类别 {class_name} 训练集中没有可用的候选support")
            return []
        
        if len(query_samples) == 0:
            print(f"   ⚠️ 类别 {class_name} 验证集中没有可用的query样本")
            return []
        
        # 限制query数量
        if max_query and len(query_samples) > max_query:
            random.shuffle(query_samples)
            query_samples = query_samples[:max_query]
            print(f"   📉 Query限制为 {len(query_samples)} 个")
        
        # 预筛选
        if prefilter_k and len(candidates) > prefilter_k:
            candidates = self.prefilter_by_feature(candidates, top_k=prefilter_k)
        
        # 计算总评估次数
        total_evals = len(candidates) * (len(query_samples) - 1)
        print(f"\n📊 类别 {class_name}:")
        print(f"   候选Support: {len(candidates)}")
        print(f"   Query数量: {len(query_samples)}")
        print(f"   总评估次数: {total_evals:,}")
        
        # 穷举评估
        results = []
        
        for i, support in enumerate(tqdm(candidates, desc=f"   Evaluating {class_name}")):
            mean_iou, all_ious = self.evaluate_support_exhaustive(
                support, query_samples,
                use_cmrs=use_cmrs,
                use_memory=use_memory
            )
            
            results.append({
                'name': support['img_name'],
                'iou': mean_iou,
                'std': np.std(all_ious) if all_ious else 0.0,
                'n_eval': len(all_ious),
                'source': support.get('source', 'unknown')
            })
            
            # 定期打印进度
            if (i + 1) % 50 == 0:
                current_best = max(results, key=lambda x: x['iou'])
                print(f"      进度 {i+1}/{len(candidates)}, 当前最优: {current_best['name']} ({current_best['iou']*100:.2f}%)")
        
        # 按IoU排序
        results.sort(key=lambda x: x['iou'], reverse=True)
        
        # 打印Top-K
        print(f"\n   🏆 Top-{min(top_k, len(results))} supports for {class_name}:")
        for i, r in enumerate(results[:top_k], 1):
            source_tag = f"[{r.get('source', '?')}]"
            print(f"      {i}. {r['name']} {source_tag}: IoU={r['iou']*100:.2f}% (std={r['std']*100:.2f}%, n={r['n_eval']})")
        
        return results
    
    def process_fold(self, max_candidates: int = None,
                    max_query: int = None,
                    prefilter_k: int = None,
                    use_cmrs: bool = True,
                    use_memory: bool = True,
                    top_k: int = 20,
                    output_file: str = None) -> Dict:
        """处理整个fold"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{'='*70}")
        print(f"🔍 穷举搜索最优Support - Fold {self.fold}")
        print(f"{'='*70}")
        print(f"时间: {timestamp}")
        print(f"类别: {', '.join(self.test_classes)}")
        print(f"Support来源: 训练集 (train)")
        print(f"Query来源: 验证集 (val)")
        print(f"最大候选: {max_candidates or '全部'}")
        print(f"最大Query: {max_query or '全部'}")
        print(f"预筛选: {prefilter_k or '无'}")
        print(f"模式: CMRS={use_cmrs}, Memory={use_memory}")
        print(f"{'='*70}")
        
        all_results = {}
        
        for cls in self.test_classes:
            results = self.exhaustive_search(
                cls,
                max_candidates=max_candidates,
                max_query=max_query,
                prefilter_k=prefilter_k,
                use_cmrs=use_cmrs,
                use_memory=use_memory,
                top_k=top_k
            )
            if results:
                all_results[cls] = results
        
        # 保存结果
        if output_file:
            output_data = {
                'fold': self.fold,
                'timestamp': timestamp,
                'config': {
                    'max_candidates': max_candidates,
                    'max_query': max_query,
                    'prefilter_k': prefilter_k,
                    'use_cmrs': use_cmrs,
                    'use_memory': use_memory,
                    'support_source': 'train',
                    'query_source': 'val',
                    'dino_model': self.dino_model_type,
                    'sam2_model': self.sam2_model_type
                },
                'results': all_results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n✅ 结果保存到: {output_file}")
        
        # 打印摘要
        print(f"\n{'='*70}")
        print("📊 搜索结果摘要")
        print(f"{'='*70}")
        for cls, results in all_results.items():
            if results:
                best = results[0]
                print(f"   {cls}: 最优Support = {best['name']} (IoU={best['iou']*100:.2f}%)")
        
        return all_results
    
    def generate_sorted_txt(self, all_results: Dict, output_dir: str = None):
        """生成排序后的txt文件"""
        if output_dir is None:
            output_dir = self.data_root / f'fold{self.fold}' / 'support_ranking'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📝 生成排序后的txt文件到: {output_dir}")
        
        for cls, results in all_results.items():
            txt_path = output_dir / f'{cls}.txt'
            with open(txt_path, 'w') as f:
                for r in results:
                    f.write(f"{r['name']}\n")
            print(f"   ✅ {cls}.txt ({len(results)} samples, best IoU={results[0]['iou']*100:.2f}%)")
        
        # 同时生成一个最优support的汇总文件
        summary_path = output_dir / 'best_supports.json'
        best_supports = {}
        for cls, results in all_results.items():
            if results:
                best_supports[cls] = {
                    'image_id': results[0]['name'],
                    'iou': results[0]['iou'],
                    'source': results[0].get('source', 'train')
                }
        
        with open(summary_path, 'w') as f:
            json.dump(best_supports, f, indent=2)
        print(f"   ✅ best_supports.json")
        
        print(f"\n💡 使用方法：在评估时指定 --sorted_support_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='穷举搜索最优Support')
    parser.add_argument('--data_root', type=str, default='pascal5i_output',
                       help='Pascal-5i数据集根目录')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Fold编号')
    parser.add_argument('--max_candidates', type=int, default=100,
                       help='每个类别最大候选Support数量（默认全部）')
    parser.add_argument('--max_query', type=int, default=None,
                       help='每个类别最大Query数量（默认全部）')
    parser.add_argument('--prefilter', type=int, default=None,
                       help='特征预筛选数量（默认不预筛选）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='保留top-K个结果')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型类型')
    parser.add_argument('--sam2_model', type=str, default='large',
                       help='SAM2模型类型')
    parser.add_argument('--mode', type=str, default='cmrs_memory',
                       choices=['cmrs_predictor', 'cmrs_memory'],
                       help='SP-SAM模式')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径')
    parser.add_argument('--generate_txt', action='store_true',
                       help='生成排序后的txt文件')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 解析模式
    use_cmrs = True
    use_memory = args.mode == 'cmrs_memory'
    
    # 创建选择器
    selector = ExhaustiveSupportSelector(
        data_root=args.data_root,
        fold=args.fold,
        device=args.device,
        dino_model=args.dino_model,
        sam2_model=args.sam2_model
    )
    
    # 输出文件
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'support_optimal_fold{args.fold}_{timestamp}.json'
    
    # 处理
    results = selector.process_fold(
        max_candidates=args.max_candidates,
        max_query=args.max_query,
        prefilter_k=args.prefilter,
        use_cmrs=use_cmrs,
        use_memory=use_memory,
        top_k=args.top_k,
        output_file=args.output
    )
    
    # 生成txt
    if args.generate_txt:
        selector.generate_sorted_txt(results)


if __name__ == '__main__':
    main()
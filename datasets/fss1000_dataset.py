"""
FSS-1000 数据集加载器
======================

FSS-1000: 1000类的Few-Shot分割数据集
每个类别包含10张图像和对应的mask

数据集结构：
fewshot_data/
├── abacus/
│   ├── 1.jpg
│   ├── 1.png  (mask)
│   ├── 2.jpg
│   ├── 2.png
│   └── ...
├── accordion/
│   └── ...
└── ... (约1000个类别)

使用方法：
    dataset = FSS1000Dataset('fewshot_data')
    episode = dataset.get_episode('abacus', k_shot=1)
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


class FSS1000Dataset:
    """
    FSS-1000 数据集加载器
    """
    
    def __init__(self, root_dir: str, 
                 test_split: float = 0.5,
                 random_seed: int = 42):
        """
        Args:
            root_dir: 数据集根目录 (fewshot_data/)
            test_split: 测试集比例（每个类别中用于测试的图像比例）
            random_seed: 随机种子（用于划分train/test）
        """
        self.root_dir = Path(root_dir)
        self.test_split = test_split
        self.random_seed = random_seed
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root_dir}")
        
        # 获取所有类别
        self.classes = self._get_all_classes()
        
        # 为每个类别划分support/query
        self.class_splits = {}
        self._split_classes()
        
        print(f"📁 加载FSS-1000数据集")
        print(f"   根目录: {self.root_dir}")
        print(f"   类别数: {len(self.classes)}")
        print(f"   测试比例: {self.test_split}")
    
    def _get_all_classes(self) -> List[str]:
        """获取所有类别名称"""
        classes = []
        for item in sorted(self.root_dir.iterdir()):
            if item.is_dir():
                # 检查是否包含图像文件
                jpg_files = list(item.glob('*.jpg')) + list(item.glob('*.JPG'))
                if len(jpg_files) > 0:
                    classes.append(item.name)
        return classes
    
    def _split_classes(self):
        """为每个类别划分support和query"""
        rng = random.Random(self.random_seed)
        
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            
            # 获取所有图像（不含扩展名的文件名）
            image_ids = set()
            for f in cls_dir.iterdir():
                if f.suffix.lower() in ['.jpg', '.jpeg']:
                    image_ids.add(f.stem)
            
            # 过滤：只保留有对应mask的图像
            valid_ids = []
            for img_id in image_ids:
                mask_path = cls_dir / f'{img_id}.png'
                if mask_path.exists():
                    valid_ids.append(img_id)
            
            # 排序以保证一致性
            valid_ids = sorted(valid_ids)
            
            # 随机划分
            rng.shuffle(valid_ids)
            n_test = max(1, int(len(valid_ids) * self.test_split))
            
            self.class_splits[cls_name] = {
                'support': valid_ids[n_test:],  # 前半部分作为support候选
                'query': valid_ids[:n_test]     # 后半部分作为query
            }
    
    def _load_image(self, cls_name: str, img_id: str) -> Image.Image:
        """加载图像"""
        cls_dir = self.root_dir / cls_name
        
        # 尝试不同的扩展名
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG']:
            img_path = cls_dir / f'{img_id}{ext}'
            if img_path.exists():
                return Image.open(img_path).convert('RGB')
        
        raise FileNotFoundError(f"图像不存在: {cls_dir}/{img_id}.*")
    
    def _load_mask(self, cls_name: str, img_id: str) -> np.ndarray:
        """加载mask（二值化）"""
        cls_dir = self.root_dir / cls_name
        mask_path = cls_dir / f'{img_id}.png'
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask不存在: {mask_path}")
        
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 二值化：>127为前景
        binary_mask = (mask > 127).astype(np.uint8)
        
        return binary_mask
    
    def _load_sample(self, cls_name: str, img_id: str) -> Dict:
        """加载单个样本"""
        img = self._load_image(cls_name, img_id)
        mask = self._load_mask(cls_name, img_id)
        
        return {
            'img': img,
            'mask': mask,
            'class': cls_name,
            'img_name': img_id
        }
    
    def get_support_samples(self, cls_name: str, k_shot: int = 1,
                           random_select: bool = False,
                           random_seed: int = None) -> List[Dict]:
        """
        获取support样本
        
        Args:
            cls_name: 类别名称
            k_shot: 采样数量
            random_select: 是否随机选择
            random_seed: 随机种子
            
        Returns:
            support样本列表
        """
        if cls_name not in self.class_splits:
            raise ValueError(f"未知类别: {cls_name}")
        
        support_ids = self.class_splits[cls_name]['support']
        
        if len(support_ids) == 0:
            print(f"   ⚠️ 类别 {cls_name} 没有support样本")
            return []
        
        # 选择样本
        if random_select:
            rng = random.Random(random_seed)
            selected_ids = rng.sample(support_ids, min(k_shot, len(support_ids)))
        else:
            # 按顺序选择前k个
            selected_ids = support_ids[:k_shot]
        
        # 加载样本
        samples = []
        for img_id in selected_ids:
            try:
                sample = self._load_sample(cls_name, img_id)
                if sample['mask'].sum() > 0:
                    samples.append(sample)
            except Exception as e:
                print(f"   ⚠️ 加载 {cls_name}/{img_id} 失败: {e}")
        
        return samples
    
    def get_query_samples(self, cls_name: str) -> List[Dict]:
        """
        获取query样本
        
        Args:
            cls_name: 类别名称
            
        Returns:
            query样本列表
        """
        if cls_name not in self.class_splits:
            raise ValueError(f"未知类别: {cls_name}")
        
        query_ids = self.class_splits[cls_name]['query']
        
        samples = []
        for img_id in query_ids:
            try:
                sample = self._load_sample(cls_name, img_id)
                if sample['mask'].sum() > 0:
                    samples.append(sample)
            except Exception as e:
                print(f"   ⚠️ 加载 {cls_name}/{img_id} 失败: {e}")
        
        return samples
    
    def get_episode(self, cls_name: str, k_shot: int = 1,
                   random_select: bool = False,
                   random_seed: int = None) -> Dict:
        """
        获取一个完整的episode
        
        Args:
            cls_name: 类别名称
            k_shot: support数量
            random_select: 是否随机选择support
            random_seed: 随机种子
            
        Returns:
            episode字典
        """
        support = self.get_support_samples(cls_name, k_shot, random_select, random_seed)
        query = self.get_query_samples(cls_name)
        
        return {
            'class': cls_name,
            'support': support,
            'query': query,
            'k_shot': k_shot
        }
    
    def get_class_info(self, cls_name: str) -> Dict:
        """获取类别信息"""
        if cls_name not in self.class_splits:
            raise ValueError(f"未知类别: {cls_name}")
        
        split = self.class_splits[cls_name]
        return {
            'class': cls_name,
            'n_support': len(split['support']),
            'n_query': len(split['query']),
            'support_ids': split['support'],
            'query_ids': split['query']
        }
    
    def print_statistics(self, n_show: int = 10):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"FSS-1000 数据集统计")
        print(f"{'='*60}")
        print(f"总类别数: {len(self.classes)}")
        
        # 统计每个类别的样本数
        total_support = 0
        total_query = 0
        
        for cls_name in self.classes:
            split = self.class_splits[cls_name]
            total_support += len(split['support'])
            total_query += len(split['query'])
        
        print(f"总Support样本: {total_support}")
        print(f"总Query样本: {total_query}")
        print(f"平均每类Support: {total_support / len(self.classes):.1f}")
        print(f"平均每类Query: {total_query / len(self.classes):.1f}")
        
        # 显示部分类别
        print(f"\n前 {n_show} 个类别:")
        print(f"{'类别名':<30} {'Support':<10} {'Query':<10}")
        print("-" * 50)
        
        for cls_name in self.classes[:n_show]:
            split = self.class_splits[cls_name]
            print(f"{cls_name:<30} {len(split['support']):<10} {len(split['query']):<10}")
        
        if len(self.classes) > n_show:
            print(f"... 还有 {len(self.classes) - n_show} 个类别")
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        cls_name = self.classes[idx]
        return self.get_episode(cls_name)


def test_dataset():
    """测试数据集加载器"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("测试FSS-1000数据集加载器")
    print("="*60)
    
    # 加载数据集
    dataset = FSS1000Dataset('fewshot_data')
    dataset.print_statistics()
    
    # 获取一个episode
    if len(dataset.classes) > 0:
        cls_name = dataset.classes[0]
        episode = dataset.get_episode(cls_name, k_shot=1)
        
        print(f"\n测试Episode: {cls_name}")
        print(f"   Support数量: {len(episode['support'])}")
        print(f"   Query数量: {len(episode['query'])}")
        
        if len(episode['support']) > 0 and len(episode['query']) > 0:
            support = episode['support'][0]
            query = episode['query'][0]
            
            # 可视化
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Support
            axes[0, 0].imshow(support['img'])
            axes[0, 0].set_title(f"Support: {support['img_name']}")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(support['mask'], cmap='gray')
            axes[0, 1].set_title(f"Support Mask")
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(support['img'])
            axes[0, 2].imshow(support['mask'], alpha=0.5, cmap='jet')
            axes[0, 2].set_title("Support Overlay")
            axes[0, 2].axis('off')
            
            # Query
            axes[1, 0].imshow(query['img'])
            axes[1, 0].set_title(f"Query: {query['img_name']}")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(query['mask'], cmap='gray')
            axes[1, 1].set_title(f"Query Mask")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(query['img'])
            axes[1, 2].imshow(query['mask'], alpha=0.5, cmap='jet')
            axes[1, 2].set_title("Query Overlay")
            axes[1, 2].axis('off')
            
            plt.suptitle(f"FSS-1000: {cls_name}")
            plt.tight_layout()
            plt.savefig('fss1000_test.png', dpi=150)
            print(f"\n✅ 可视化已保存: fss1000_test.png")
            plt.close()


if __name__ == '__main__':
    test_dataset()

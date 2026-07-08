"""
ISIC2018 皮肤病变分割数据集加载器
==================================

ISIC2018: 皮肤镜图像病变分割数据集
单类别（病变区域）的语义分割任务

数据集结构：
ISIC2018_256/
├── image/                          # 原始图像
│   ├── ISIC_0000000.jpg
│   ├── ISIC_0000001.jpg
│   └── ...
├── gt/                             # 分割mask
│   ├── ISIC_0000000_segmentation.png
│   ├── ISIC_0000001_segmentation.png
│   └── ...
├── image.txt                       # 图像列表（可选）
├── gt.txt                          # mask列表（可选）
└── output/                         # 输出目录

使用方法：
    dataset = ISIC2018Dataset('ISIC2018_256')
    episode = dataset.get_episode(k_shot=5)
    
    # 获取support和query
    support_imgs = [s['img'] for s in episode['support']]
    support_masks = [s['mask'] for s in episode['support']]
    query_samples = episode['query']
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


class ISIC2018Dataset:
    """
    ISIC2018 皮肤病变分割数据集加载器
    
    用于Few-shot分割评估：
    - 从数据集中采样k个support样本
    - 其余作为query样本进行评估
    """
    
    def __init__(self, root_dir: str,
                 support_ratio: float = 0.1,
                 random_seed: int = 42,
                 max_samples: int = None):
        """
        Args:
            root_dir: 数据集根目录 (ISIC2018_256/)
            support_ratio: support集合比例（默认10%用于support）
            random_seed: 随机种子（用于划分support/query）
            max_samples: 最大样本数（用于快速测试，None表示使用全部）
        """
        self.root_dir = Path(root_dir)
        self.support_ratio = support_ratio
        self.random_seed = random_seed
        self.max_samples = max_samples
        
        # 设置路径
        self.image_dir = self.root_dir / 'image'
        self.gt_dir = self.root_dir / 'gt'
        
        # 验证目录存在
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root_dir}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT目录不存在: {self.gt_dir}")
        
        # 加载样本列表
        self.samples = self._load_samples()
        
        # 划分support和query
        self.support_ids = []
        self.query_ids = []
        self._split_dataset()
        
        print(f"📁 加载ISIC2018数据集")
        print(f"   根目录: {self.root_dir}")
        print(f"   总样本数: {len(self.samples)}")
        print(f"   Support样本: {len(self.support_ids)}")
        print(f"   Query样本: {len(self.query_ids)}")
    
    def _load_samples(self) -> List[str]:
        """加载所有有效样本ID"""
        samples = []
        
        # 遍历图像目录
        for img_file in sorted(self.image_dir.iterdir()):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # 提取样本ID (ISIC_0000000)
                sample_id = img_file.stem
                
                # 检查对应的mask是否存在
                mask_path = self.gt_dir / f'{sample_id}_segmentation.png'
                if mask_path.exists():
                    samples.append(sample_id)
        
        # 限制样本数量
        if self.max_samples and len(samples) > self.max_samples:
            rng = random.Random(self.random_seed)
            samples = rng.sample(samples, self.max_samples)
            samples = sorted(samples)
        
        return samples
    
    def _split_dataset(self):
        """划分support和query集合"""
        rng = random.Random(self.random_seed)
        
        # 复制并打乱
        all_ids = self.samples.copy()
        rng.shuffle(all_ids)
        
        # 按比例划分
        n_support = max(1, int(len(all_ids) * self.support_ratio))
        
        self.support_ids = all_ids[:n_support]
        self.query_ids = all_ids[n_support:]
    
    def _load_image(self, sample_id: str) -> Image.Image:
        """加载图像"""
        # 尝试不同扩展名
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = self.image_dir / f'{sample_id}{ext}'
            if img_path.exists():
                return Image.open(img_path).convert('RGB')
        
        raise FileNotFoundError(f"图像不存在: {self.image_dir}/{sample_id}.*")
    
    def _load_mask(self, sample_id: str) -> np.ndarray:
        """加载mask（二值化）"""
        mask_path = self.gt_dir / f'{sample_id}_segmentation.png'
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask不存在: {mask_path}")
        
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 二值化：>127为前景（病变区域）
        binary_mask = (mask > 127).astype(np.uint8)
        
        return binary_mask
    
    def _load_sample(self, sample_id: str) -> Dict:
        """加载单个样本"""
        img = self._load_image(sample_id)
        mask = self._load_mask(sample_id)
        
        return {
            'img': img,
            'mask': mask,
            'sample_id': sample_id,
            'class': 'lesion'  # ISIC只有一个类别
        }
    
    def get_support_samples(self, k_shot: int = 5,
                           random_select: bool = True,
                           random_seed: int = None) -> List[Dict]:
        """
        获取support样本
        
        Args:
            k_shot: 采样数量
            random_select: 是否随机选择
            random_seed: 随机种子
            
        Returns:
            support样本列表
        """
        if len(self.support_ids) == 0:
            print("   ⚠️ 没有support样本")
            return []
        
        # 选择样本
        if random_select:
            rng = random.Random(random_seed if random_seed else self.random_seed)
            k = min(k_shot, len(self.support_ids))
            selected_ids = rng.sample(self.support_ids, k)
        else:
            selected_ids = self.support_ids[:k_shot]
        
        # 加载样本
        samples = []
        for sample_id in selected_ids:
            try:
                sample = self._load_sample(sample_id)
                # 确保mask非空
                if sample['mask'].sum() > 0:
                    samples.append(sample)
            except Exception as e:
                print(f"   ⚠️ 加载 {sample_id} 失败: {e}")
        
        return samples
    
    def get_query_samples(self, max_queries: int = None) -> List[Dict]:
        """
        获取query样本
        
        Args:
            max_queries: 最大query数量（None表示全部）
            
        Returns:
            query样本列表
        """
        query_ids = self.query_ids
        if max_queries and len(query_ids) > max_queries:
            query_ids = query_ids[:max_queries]
        
        samples = []
        for sample_id in query_ids:
            try:
                sample = self._load_sample(sample_id)
                if sample['mask'].sum() > 0:
                    samples.append(sample)
            except Exception as e:
                print(f"   ⚠️ 加载 {sample_id} 失败: {e}")
        
        return samples
    
    def get_episode(self, k_shot: int = 5,
                   random_select: bool = True,
                   random_seed: int = None,
                   max_queries: int = None) -> Dict:
        """
        获取一个完整的episode
        
        Args:
            k_shot: support数量
            random_select: 是否随机选择support
            random_seed: 随机种子
            max_queries: 最大query数量
            
        Returns:
            episode字典
        """
        support = self.get_support_samples(k_shot, random_select, random_seed)
        query = self.get_query_samples(max_queries)
        
        return {
            'class': 'lesion',
            'support': support,
            'query': query,
            'k_shot': k_shot
        }
    
    def get_sample_by_id(self, sample_id: str) -> Dict:
        """根据ID获取样本"""
        return self._load_sample(sample_id)
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"ISIC2018 数据集统计")
        print(f"{'='*60}")
        print(f"数据集目录: {self.root_dir}")
        print(f"总样本数: {len(self.samples)}")
        print(f"Support样本: {len(self.support_ids)} ({self.support_ratio*100:.1f}%)")
        print(f"Query样本: {len(self.query_ids)} ({(1-self.support_ratio)*100:.1f}%)")
        
        # 统计mask信息
        print(f"\n采样mask统计（前10个support）:")
        for sample_id in self.support_ids[:10]:
            try:
                mask = self._load_mask(sample_id)
                fg_ratio = mask.sum() / mask.size * 100
                print(f"  {sample_id}: {fg_ratio:.1f}% 前景")
            except:
                pass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        return self._load_sample(sample_id)


class ISIC2018FoldDataset(ISIC2018Dataset):
    """
    ISIC2018 K-Fold交叉验证数据集
    
    用于更严格的评估：将数据集分成K折，每折轮流作为query
    """
    
    def __init__(self, root_dir: str,
                 n_folds: int = 5,
                 current_fold: int = 0,
                 random_seed: int = 42,
                 max_samples: int = None):
        """
        Args:
            root_dir: 数据集根目录
            n_folds: 折数
            current_fold: 当前折（0到n_folds-1）
            random_seed: 随机种子
            max_samples: 最大样本数
        """
        self.n_folds = n_folds
        self.current_fold = current_fold
        
        # 调用父类初始化（会调用_split_dataset）
        super().__init__(root_dir, 
                        support_ratio=0,  # 这里不用，我们自己划分
                        random_seed=random_seed,
                        max_samples=max_samples)
    
    def _split_dataset(self):
        """按K-Fold划分数据集"""
        rng = random.Random(self.random_seed)
        
        # 打乱样本
        all_ids = self.samples.copy()
        rng.shuffle(all_ids)
        
        # 分成K折
        fold_size = len(all_ids) // self.n_folds
        folds = []
        for i in range(self.n_folds):
            start = i * fold_size
            if i == self.n_folds - 1:
                # 最后一折包含剩余所有样本
                folds.append(all_ids[start:])
            else:
                folds.append(all_ids[start:start + fold_size])
        
        # 当前折作为query，其余作为support
        self.query_ids = folds[self.current_fold]
        self.support_ids = []
        for i, fold in enumerate(folds):
            if i != self.current_fold:
                self.support_ids.extend(fold)
        
        print(f"   Fold {self.current_fold + 1}/{self.n_folds}")


def test_dataset():
    """测试数据集加载器"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("测试ISIC2018数据集加载器")
    print("="*60)
    
    # 加载数据集（这里用实际路径）
    try:
        dataset = ISIC2018Dataset('ISIC2018_256', max_samples=100)
        dataset.print_statistics()
        
        # 获取一个episode
        episode = dataset.get_episode(k_shot=5)
        
        print(f"\n测试Episode:")
        print(f"   Support数量: {len(episode['support'])}")
        print(f"   Query数量: {len(episode['query'])}")
        
        if len(episode['support']) > 0 and len(episode['query']) > 0:
            support = episode['support'][0]
            query = episode['query'][0]
            
            # 可视化
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Support
            axes[0, 0].imshow(support['img'])
            axes[0, 0].set_title(f"Support: {support['sample_id']}")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(support['mask'], cmap='gray')
            axes[0, 1].set_title(f"Support Mask")
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(support['img'])
            axes[0, 2].imshow(support['mask'], alpha=0.5, cmap='Reds')
            axes[0, 2].set_title("Support Overlay")
            axes[0, 2].axis('off')
            
            # Query
            axes[1, 0].imshow(query['img'])
            axes[1, 0].set_title(f"Query: {query['sample_id']}")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(query['mask'], cmap='gray')
            axes[1, 1].set_title(f"Query Mask")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(query['img'])
            axes[1, 2].imshow(query['mask'], alpha=0.5, cmap='Reds')
            axes[1, 2].set_title("Query Overlay")
            axes[1, 2].axis('off')
            
            plt.suptitle("ISIC2018: Skin Lesion Segmentation")
            plt.tight_layout()
            plt.savefig('isic2018_test.png', dpi=150)
            print(f"\n✅ 可视化已保存: isic2018_test.png")
            plt.close()
            
    except FileNotFoundError as e:
        print(f"⚠️ 数据集未找到: {e}")
        print("请确保数据集路径正确")


if __name__ == '__main__':
    test_dataset()
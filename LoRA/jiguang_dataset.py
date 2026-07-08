"""
Jiguang 数据集加载器
=====================

数据集结构：
jiguangdatasets/
├── train_images/   (.png, .bmp)
├── train_masks/    (.png)
├── test_images/    (.png, .bmp)  
├── test_masks/     (.png)

使用方法：
    dataset = JiguangDataset('jiguangdatasets', split='train')
    episode = dataset.get_episode(k_shot=1)
"""

import os
import sys

# 添加父目录到路径，以便导入根目录中的模块
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LORA_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import torch
from torch.utils.data import Dataset


class JiguangDataset(Dataset):
    """
    Jiguang Few-Shot 分割数据集
    
    设计说明：
    - 训练时：从train_images中随机选择k个作为support，其余作为query
    - 测试时：从train_images选support，test_images作为query
    """
    
    def __init__(self, root_dir: str, 
                 split: str = 'train',
                 k_shot: int = 1,
                 random_seed: int = 42,
                 transform=None):
        """
        Args:
            root_dir: 数据集根目录 (jiguangdatasets/)
            split: 'train' 或 'test'
            k_shot: few-shot的k值
            random_seed: 随机种子
            transform: 图像变换（可选）
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.k_shot = k_shot
        self.random_seed = random_seed
        self.transform = transform
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root_dir}")
        
        # 加载图像列表
        self.train_samples = self._load_samples('train')
        self.test_samples = self._load_samples('test')
        
        # 根据split决定使用哪些数据
        if split == 'train':
            self.samples = self.train_samples
        else:
            self.samples = self.test_samples
        
        print(f"📁 加载Jiguang数据集 ({split})")
        print(f"   根目录: {self.root_dir}")
        print(f"   训练样本: {len(self.train_samples)}")
        print(f"   测试样本: {len(self.test_samples)}")
    
    def _load_samples(self, split: str) -> List[Dict]:
        """加载指定split的所有样本"""
        img_dir = self.root_dir / f'{split}_images'
        mask_dir = self.root_dir / f'{split}_masks'
        
        if not img_dir.exists() or not mask_dir.exists():
            print(f"⚠️ {split}目录不完整")
            return []
        
        samples = []
        
        # 支持的图像格式
        img_extensions = ['.png', '.PNG', '.bmp', '.BMP', '.jpg', '.JPG', '.jpeg', '.JPEG']
        
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix not in img_extensions:
                continue
            
            # 查找对应的mask（假设mask是.png格式）
            img_stem = img_path.stem
            mask_path = mask_dir / f'{img_stem}.png'
            
            if not mask_path.exists():
                # 尝试其他可能的mask文件名
                mask_path = mask_dir / f'{img_stem}.PNG'
                if not mask_path.exists():
                    continue
            
            samples.append({
                'img_path': str(img_path),
                'mask_path': str(mask_path),
                'img_name': img_stem
            })
        
        return samples
    
    def _load_image(self, path: str) -> Image.Image:
        """加载图像"""
        return Image.open(path).convert('RGB')
    
    def _load_mask(self, path: str) -> np.ndarray:
        """加载mask（二值化）"""
        mask = np.array(Image.open(path).convert('L'))
        # 二值化：>127为前景
        binary_mask = (mask > 127).astype(np.uint8)
        return binary_mask
    
    def _load_sample(self, sample_info: Dict) -> Dict:
        """加载单个样本"""
        img = self._load_image(sample_info['img_path'])
        mask = self._load_mask(sample_info['mask_path'])
        
        return {
            'img': img,
            'mask': mask,
            'img_name': sample_info['img_name']
        }
    
    def get_episode(self, k_shot: int = None, random_seed: int = None) -> Dict:
        """
        获取一个few-shot episode
        
        训练模式：从训练集随机选k个support，其余作为query
        测试模式：从训练集选k个support，测试集作为query
        
        Args:
            k_shot: support数量（默认使用初始化时的k_shot）
            random_seed: 随机种子
            
        Returns:
            episode字典
        """
        k = k_shot if k_shot is not None else self.k_shot
        seed = random_seed if random_seed is not None else self.random_seed
        
        rng = random.Random(seed)
        
        if self.split == 'train':
            # 训练模式：从训练集中分出support和query
            all_indices = list(range(len(self.train_samples)))
            rng.shuffle(all_indices)
            
            support_indices = all_indices[:k]
            query_indices = all_indices[k:]
            
            support_samples = [self._load_sample(self.train_samples[i]) for i in support_indices]
            query_samples = [self._load_sample(self.train_samples[i]) for i in query_indices]
        else:
            # 测试模式：训练集作为support候选，测试集作为query
            train_indices = list(range(len(self.train_samples)))
            rng.shuffle(train_indices)
            support_indices = train_indices[:k]
            
            support_samples = [self._load_sample(self.train_samples[i]) for i in support_indices]
            query_samples = [self._load_sample(s) for s in self.test_samples]
        
        return {
            'support': support_samples,
            'query': query_samples,
            'k_shot': k
        }
    
    def get_training_batch(self, batch_size: int = 4, k_shot: int = None) -> List[Dict]:
        """
        获取一个训练batch
        
        每个episode包含k个support和1个query
        """
        k = k_shot if k_shot is not None else self.k_shot
        
        batches = []
        rng = random.Random()
        
        for _ in range(batch_size):
            indices = list(range(len(self.train_samples)))
            rng.shuffle(indices)
            
            # 选择k个support和1个query
            support_indices = indices[:k]
            query_idx = indices[k] if len(indices) > k else indices[0]
            
            support_samples = [self._load_sample(self.train_samples[i]) for i in support_indices]
            query_sample = self._load_sample(self.train_samples[query_idx])
            
            batches.append({
                'support': support_samples,
                'query': query_sample,
                'k_shot': k
            })
        
        return batches
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """用于DataLoader的接口"""
        sample = self._load_sample(self.samples[idx])
        
        if self.transform:
            sample['img'] = self.transform(sample['img'])
        
        return sample
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"Jiguang 数据集统计")
        print(f"{'='*60}")
        print(f"训练样本数: {len(self.train_samples)}")
        print(f"测试样本数: {len(self.test_samples)}")
        
        if len(self.train_samples) > 0:
            sample = self._load_sample(self.train_samples[0])
            print(f"图像尺寸: {sample['img'].size}")
            print(f"Mask尺寸: {sample['mask'].shape}")


class JiguangTrainDataset(Dataset):
    """
    用于训练的Jiguang数据集
    
    每次__getitem__返回一个episode：k个support + 1个query
    """
    
    def __init__(self, root_dir: str, 
                 k_shot: int = 1,
                 num_episodes: int = 1000,
                 transform=None,
                 mask_transform=None):
        """
        Args:
            root_dir: 数据集根目录
            k_shot: few-shot的k值
            num_episodes: 每个epoch的episode数量
            transform: 图像变换
            mask_transform: mask变换
        """
        self.root_dir = Path(root_dir)
        self.k_shot = k_shot
        self.num_episodes = num_episodes
        self.transform = transform
        self.mask_transform = mask_transform
        
        # 加载训练样本
        self.samples = self._load_samples('train')
        
        if len(self.samples) < k_shot + 1:
            raise ValueError(f"训练样本数({len(self.samples)})不足，需要至少{k_shot + 1}个")
        
        print(f"📁 JiguangTrainDataset")
        print(f"   样本数: {len(self.samples)}")
        print(f"   K-shot: {k_shot}")
        print(f"   Episodes/epoch: {num_episodes}")
    
    def _load_samples(self, split: str) -> List[Dict]:
        """加载样本列表"""
        img_dir = self.root_dir / f'{split}_images'
        mask_dir = self.root_dir / f'{split}_masks'
        
        samples = []
        img_extensions = ['.png', '.PNG', '.bmp', '.BMP', '.jpg', '.JPG']
        
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix not in img_extensions:
                continue
            
            mask_path = mask_dir / f'{img_path.stem}.png'
            if not mask_path.exists():
                mask_path = mask_dir / f'{img_path.stem}.PNG'
            
            if mask_path.exists():
                samples.append({
                    'img_path': str(img_path),
                    'mask_path': str(mask_path)
                })
        
        return samples
    
    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')
    
    def _load_mask(self, path: str) -> np.ndarray:
        mask = np.array(Image.open(path).convert('L'))
        return (mask > 127).astype(np.uint8)
    
    def __len__(self):
        return self.num_episodes
    
    def __getitem__(self, idx):
        """返回一个训练episode"""
        # 随机选择k+1个样本
        indices = random.sample(range(len(self.samples)), self.k_shot + 1)
        
        support_imgs = []
        support_masks = []
        
        for i in range(self.k_shot):
            sample = self.samples[indices[i]]
            img = self._load_image(sample['img_path'])
            mask = self._load_mask(sample['mask_path'])
            
            if self.transform:
                img = self.transform(img)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            
            support_imgs.append(img)
            support_masks.append(mask)
        
        # Query
        query_sample = self.samples[indices[self.k_shot]]
        query_img = self._load_image(query_sample['img_path'])
        query_mask = self._load_mask(query_sample['mask_path'])
        
        if self.transform:
            query_img = self.transform(query_img)
        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)
        
        return {
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'query_img': query_img,
            'query_mask': query_mask
        }


def test_dataset():
    """测试数据集加载器"""
    print("\n" + "="*60)
    print("测试Jiguang数据集加载器")
    print("="*60)
    
    # 数据集路径（相对于根目录）
    data_root = os.path.join(ROOT_DIR, 'jiguangdatasets')
    
    try:
        dataset = JiguangDataset(data_root, split='train')
        dataset.print_statistics()
        
        # 测试获取episode
        episode = dataset.get_episode(k_shot=1)
        print(f"\nEpisode:")
        print(f"   Support数量: {len(episode['support'])}")
        print(f"   Query数量: {len(episode['query'])}")
        
        if len(episode['support']) > 0:
            sup = episode['support'][0]
            print(f"   Support图像尺寸: {sup['img'].size}")
            print(f"   Support Mask尺寸: {sup['mask'].shape}")
            print(f"   Support Mask前景比例: {sup['mask'].mean()*100:.2f}%")
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_dataset()

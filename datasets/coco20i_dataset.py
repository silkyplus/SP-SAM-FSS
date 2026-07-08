"""
COCO-20i数据集加载器
====================

支持标准的COCO-20i Few-Shot语义分割数据集
基于make_coco20i_standard.py生成的数据结构
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


class COCO20iDataset:
    """
    COCO-20i数据集加载器
    
    数据集结构：
    coco20i/
      ├── class_info.json      # 类别映射信息
      ├── paths.json           # 原始图像路径
      ├── fold0/
      │   ├── fold_info.json   # fold信息（novel/base类）
      │   ├── test/
      │   │   ├── samples.json
      │   │   └── masks/
      │   └── support/
      │       ├── samples.json (或 samples_ranked.json)
      │       └── masks/
      ├── fold1/
      ├── fold2/
      └── fold3/
    """
    
    def __init__(self, root_dir: str, coco_root: str, fold: int, 
                 use_ranked_support: bool = True):
        """
        Args:
            root_dir: COCO-20i数据集根目录 (e.g., './coco20i')
            coco_root: COCO原始数据集根目录 (e.g., 'D:/data/coco2014')
            fold: fold编号 (0, 1, 2, 3)
            use_ranked_support: 是否使用排序后的support (samples_ranked.json)
        """
        self.root_dir = Path(root_dir)
        self.coco_root = Path(coco_root)
        self.fold = fold
        self.use_ranked_support = use_ranked_support
        
        # 检查目录是否存在
        if not self.root_dir.exists():
            raise FileNotFoundError(f"COCO-20i目录不存在: {self.root_dir}")
        
        # 加载类别信息
        with open(self.root_dir / 'class_info.json', 'r') as f:
            self.class_info = json.load(f)
        
        self.class_id_to_name = {
            int(k): v for k, v in self.class_info['class_id_to_name'].items()
        }
        
        # 加载路径配置
        with open(self.root_dir / 'paths.json', 'r') as f:
            paths = json.load(f)
        
        self.train_img_dir = Path(paths['train_images'])
        self.val_img_dir = Path(paths['val_images'])
        
        # 加载fold信息
        self.fold_dir = self.root_dir / f'fold{fold}'
        if not self.fold_dir.exists():
            raise FileNotFoundError(f"Fold目录不存在: {self.fold_dir}")
        
        with open(self.fold_dir / 'fold_info.json', 'r') as f:
            self.fold_info = json.load(f)
        
        self.novel_classes = self.fold_info['novel_classes']
        self.base_classes = self.fold_info['base_classes']
        self.novel_class_names = self.fold_info['novel_class_names']
        
        # 加载test样本
        self.test_dir = self.fold_dir / 'test'
        with open(self.test_dir / 'samples.json', 'r') as f:
            self.test_samples = json.load(f)
        
        # 加载support样本
        self.support_dir = self.fold_dir / 'support'
        
        if use_ranked_support and (self.support_dir / 'samples_ranked.json').exists():
            # 使用排序后的support
            with open(self.support_dir / 'samples_ranked.json', 'r') as f:
                self.support_samples = json.load(f)
            print(f"   使用排序后的Support (samples_ranked.json)")
        else:
            # 使用原始support
            with open(self.support_dir / 'samples.json', 'r') as f:
                self.support_samples = json.load(f)
            print(f"   使用原始Support (samples.json)")
        
        self.test_mask_dir = self.test_dir / 'masks'
        self.support_mask_dir = self.support_dir / 'masks'
        
        print(f"📁 加载COCO-20i Fold {fold}")
        print(f"   Novel类: {len(self.novel_classes)}个 - {self.novel_class_names[:3]}...")
        print(f"   Base类: {len(self.base_classes)}个")
        print(f"   测试样本: {len(self.test_samples)}个")
        
        # 统计support样本数
        total_support = 0
        for class_id in self.novel_classes:
            support_list = self.support_samples.get(str(class_id), [])
            total_support += len(support_list)
        print(f"   Support样本: {total_support}个")
    
    def _load_image(self, image_file: str, from_train: bool = False) -> Image.Image:
        """加载图像"""
        img_dir = self.train_img_dir if from_train else self.val_img_dir
        img_path = img_dir / image_file
        
        if not img_path.exists():
            raise FileNotFoundError(f"图像不存在: {img_path}")
        
        img = Image.open(img_path).convert('RGB')
        return img
    
    def _load_mask(self, mask_file: str, from_support: bool = False) -> np.ndarray:
        """加载mask"""
        mask_dir = self.support_mask_dir if from_support else self.test_mask_dir
        mask_path = mask_dir / mask_file
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask不存在: {mask_path}")
        
        mask = np.array(Image.open(mask_path))
        # 转换为二值mask (0或1)
        mask = (mask > 127).astype(np.uint8)
        
        return mask
    
    def get_test_samples(self, class_id: int) -> List[Dict]:
        """
        获取指定类别的test样本
        
        Returns:
            List of dicts with keys: img, mask, class_id, class_name, img_name
        """
        if class_id not in self.novel_classes:
            raise ValueError(f"类别 {class_id} 不在Fold {self.fold}的novel类中")
        
        class_name = self.class_id_to_name.get(class_id, f'class_{class_id}')
        
        # 筛选该类的test样本
        test_samples = []
        for sample_info in self.test_samples:
            if sample_info['class_id'] == class_id:
                try:
                    img = self._load_image(sample_info['image_file'], from_train=False)
                    mask = self._load_mask(sample_info['mask_file'], from_support=False)
                    
                    if mask.sum() > 0:
                        test_samples.append({
                            'img': img,
                            'mask': mask,
                            'class_id': class_id,
                            'class_name': class_name,
                            'img_name': sample_info['image_file'],
                            'image_id': sample_info['image_id']
                        })
                except Exception as e:
                    print(f"   ⚠️  加载test样本失败: {sample_info['image_file']}, {e}")
                    continue
        
        return test_samples
    
    def get_support_samples(self, class_id: int, k_shot: int = 5,
                           random_seed: Optional[int] = None) -> List[Dict]:
        """
        获取指定类别的support样本
        
        Args:
            class_id: 类别ID
            k_shot: 要获取的support数量
            random_seed: 随机种子（如果使用随机采样）
            
        Returns:
            List of dicts with keys: img, mask, class_id, class_name, img_name
        """
        if class_id not in self.novel_classes:
            raise ValueError(f"类别 {class_id} 不在Fold {self.fold}的novel类中")
        
        class_name = self.class_id_to_name.get(class_id, f'class_{class_id}')
        
        # 获取该类的候选support
        support_candidates = self.support_samples.get(str(class_id), [])
        
        if len(support_candidates) == 0:
            print(f"   ⚠️  类别 {class_name} (id={class_id}) 没有support样本")
            return []
        
        # 如果使用ranked support，直接取前k个
        # 否则随机采样
        if self.use_ranked_support:
            # ranked support已经按IoU降序排列，直接取前k个
            selected = support_candidates[:min(k_shot, len(support_candidates))]
        else:
            # 随机采样
            if random_seed is not None:
                random.seed(random_seed)
            
            if len(support_candidates) <= k_shot:
                selected = support_candidates
            else:
                selected = random.sample(support_candidates, k_shot)
        
        # 加载图像和mask
        support_samples = []
        for sample_info in selected:
            try:
                img = self._load_image(sample_info['image_file'], from_train=True)
                mask = self._load_mask(sample_info['mask_file'], from_support=True)
                
                if mask.sum() > 0:
                    support_samples.append({
                        'img': img,
                        'mask': mask,
                        'class_id': class_id,
                        'class_name': class_name,
                        'img_name': sample_info['image_file'],
                        'image_id': sample_info['image_id']
                    })
                else:
                    print(f"   ⚠️  跳过空mask: {sample_info['image_file']}")
            except Exception as e:
                print(f"   ⚠️  加载support样本失败: {sample_info['image_file']}, {e}")
                continue
        
        return support_samples
    
    def get_episode(self, class_id: int, k_shot: int = 5,
                   random_seed: Optional[int] = None) -> Dict:
        """
        获取一个完整的episode
        
        Returns:
            episode dict with keys: class, class_id, support, test, fold, k_shot
        """
        class_name = self.class_id_to_name.get(class_id, f'class_{class_id}')
        
        support_samples = self.get_support_samples(class_id, k_shot, random_seed)
        test_samples = self.get_test_samples(class_id)
        
        return {
            'class': class_name,
            'class_id': class_id,
            'support': support_samples,
            'test': test_samples,
            'fold': self.fold,
            'k_shot': k_shot
        }
    
    def get_all_episodes(self, k_shot: int = 5,
                        random_seed: Optional[int] = None) -> List[Dict]:
        """
        获取该fold的所有novel类别的episodes
        
        Returns:
            List of episode dicts
        """
        episodes = []
        
        print(f"\n加载 Fold {self.fold} 的所有episodes (k_shot={k_shot}):")
        
        for class_id in self.novel_classes:
            class_name = self.class_id_to_name.get(class_id, f'class_{class_id}')
            
            try:
                episode = self.get_episode(class_id, k_shot, random_seed)
                episodes.append(episode)
                
                print(f"   ✅ {class_name} (id={class_id}): "
                      f"{len(episode['support'])} support, "
                      f"{len(episode['test'])} test")
            except Exception as e:
                print(f"   ❌ {class_name} (id={class_id}): 加载失败 - {e}")
                continue
        
        return episodes


# 测试代码
def test_coco20i_dataset():
    """测试COCO-20i数据集加载器"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*80)
    print("测试COCO-20i数据集加载器")
    print("="*80)
    
    # 加载数据集
    dataset = COCO20iDataset(
        root_dir='./coco20i',
        coco_root='D:/data/coco2014',
        fold=0,
        use_ranked_support=True
    )
    
    # 获取一个episode
    class_id = dataset.novel_classes[0]
    episode = dataset.get_episode(class_id, k_shot=5)
    
    print(f"\n测试Episode:")
    print(f"   类别: {episode['class']} (id={episode['class_id']})")
    print(f"   Support: {len(episode['support'])} samples")
    print(f"   Test: {len(episode['test'])} samples")
    
    if len(episode['support']) > 0 and len(episode['test']) > 0:
        support = episode['support'][0]
        test = episode['test'][0]
        
        print(f"\nSupport样本:")
        print(f"   图像: {support['img_name']}")
        print(f"   Mask形状: {support['mask'].shape}")
        print(f"   前景像素: {support['mask'].sum()}")
        
        print(f"\nTest样本:")
        print(f"   图像: {test['img_name']}")
        print(f"   Mask形状: {test['mask'].shape}")
        print(f"   前景像素: {test['mask'].sum()}")
        
        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Support
        axes[0, 0].imshow(support['img'])
        axes[0, 0].set_title(f"Support Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(support['mask'], cmap='gray')
        axes[0, 1].set_title(f"Support Mask")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(support['img'])
        axes[0, 2].imshow(support['mask'], alpha=0.5, cmap='jet')
        axes[0, 2].set_title("Support Overlay")
        axes[0, 2].axis('off')
        
        # Test
        axes[1, 0].imshow(test['img'])
        axes[1, 0].set_title(f"Test Image")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(test['mask'], cmap='gray')
        axes[1, 1].set_title(f"Test Mask")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(test['img'])
        axes[1, 2].imshow(test['mask'], alpha=0.5, cmap='jet')
        axes[1, 2].set_title("Test Overlay")
        axes[1, 2].axis('off')
        
        plt.suptitle(f"COCO-20i Fold {dataset.fold} - {episode['class']}")
        plt.tight_layout()
        plt.savefig('coco20i_dataset_test.png', dpi=150)
        print(f"\n✅ 可视化已保存: coco20i_dataset_test.png")
        plt.close()


if __name__ == '__main__':
    test_coco20i_dataset()

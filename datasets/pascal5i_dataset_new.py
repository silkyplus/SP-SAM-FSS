"""
Pascal-5i 数据集加载器（新格式）
================================

适配新生成的Pascal-5i数据集格式

数据集结构：
pascal5i_output/
├── metadata.json
├── fold0/
│   ├── train.txt              # 训练图像ID列表
│   ├── val.txt                # 完整验证集列表 (1449张)
│   ├── val_fewshot.txt        # few-shot评估用验证集 (只含测试类别)
│   ├── train_info.json        # 训练集详细信息
│   ├── val_info.json          # 验证集详细信息
│   ├── train/
│   │   ├── images/            # 训练图像 (.jpg)
│   │   ├── masks/             # 标准mask (255→0)
│   │   ├── masks_noleak/      # 无泄露mask
│   │   └── masks_raw/         # 原始mask (保留255)
│   ├── val/
│   │   ├── images/
│   │   ├── masks/
│   │   └── masks_raw/
│   └── class_lists/           # 按类别划分的图像列表
│       ├── aeroplane.txt
│       └── ...
├── fold1/
├── fold2/
└── fold3/
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random


class Pascal5iDatasetNew:
    """
    Pascal-5i 数据集加载器（新格式）
    
    支持新生成的Pascal-5i数据集结构
    """
    
    # PASCAL-5i的类别划分
    FOLD_CLASSES = {
        0: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
        1: ['bus', 'car', 'cat', 'chair', 'cow'],
        2: ['diningtable', 'dog', 'horse', 'motorbike', 'person'],
        3: ['pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    }
    
    # 类别名到ID的映射
    CLASS_NAME_TO_ID = {
        'background': 0,
        'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
        'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
        'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
        'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
    }
    
    def __init__(self, root_dir: str, fold: int, 
                 mask_type: str = 'masks',
                 use_noleak_for_train: bool = True,
                 use_fewshot_val: bool = True):
        """
        Args:
            root_dir: 数据集根目录 (e.g., 'pascal5i_output')
            fold: fold编号 (0, 1, 2, 3)
            mask_type: mask类型 ('masks', 'masks_noleak', 'masks_raw')
            use_noleak_for_train: 训练集是否使用无泄露版本的mask
            use_fewshot_val: 是否只使用包含测试类别的验证图像（推荐True）
        """
        self.root_dir = Path(root_dir)
        self.fold = fold
        self.mask_type = mask_type
        self.use_noleak_for_train = use_noleak_for_train
        self.use_fewshot_val = use_fewshot_val
        
        # fold目录
        self.fold_dir = self.root_dir / f'fold{fold}'
        
        # 检查目录是否存在
        if not self.fold_dir.exists():
            raise FileNotFoundError(f"Fold目录不存在: {self.fold_dir}")
        
        # 设置路径
        self.train_dir = self.fold_dir / 'train'
        self.val_dir = self.fold_dir / 'val'
        self.class_lists_dir = self.fold_dir / 'class_lists'
        
        # 获取该fold的测试类别
        self.test_classes = self.FOLD_CLASSES[fold]
        self.train_classes = [c for c in self.CLASS_NAME_TO_ID.keys() 
                              if c != 'background' and c not in self.test_classes]
        
        # 加载详细信息
        self.train_info = self._load_json(self.fold_dir / 'train_info.json')
        self.val_info = self._load_json(self.fold_dir / 'val_info.json')
        
        # 加载筛选后的验证集列表（用于few-shot评估）
        self.val_fewshot_ids = self._load_txt_list(self.fold_dir / 'val_fewshot.txt')
        
        # 构建类别到图像的映射
        self._build_class_to_images()
        
        # 统计信息
        val_count = len(self.val_fewshot_ids) if use_fewshot_val else len(self.val_info)
        
        print(f"📁 加载Pascal-5i Fold {fold} (新格式)")
        print(f"   测试类别: {', '.join(self.test_classes)}")
        print(f"   训练集: {len(self.train_info)} 张图像")
        print(f"   验证集: {val_count} 张图像" + (" (筛选后)" if use_fewshot_val else " (完整)"))
        print(f"   Mask类型: {mask_type}")
    
    def _load_json(self, path: Path) -> List[Dict]:
        """加载JSON文件"""
        if not path.exists():
            print(f"   ⚠️ 文件不存在: {path}")
            return []
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_txt_list(self, path: Path) -> List[str]:
        """加载txt列表文件"""
        if not path.exists():
            return []
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _build_class_to_images(self):
        """构建类别到图像的映射"""
        # 训练集：按训练类别组织
        self.train_class_to_images = {cls: [] for cls in self.train_classes}
        for item in self.train_info:
            for cls_id in item.get('train_classes', []):
                cls_name = self._id_to_name(cls_id)
                if cls_name in self.train_class_to_images:
                    self.train_class_to_images[cls_name].append(item['image_id'])
        
        # 验证集：按测试类别组织
        self.val_class_to_images = {cls: [] for cls in self.test_classes}
        
        # 确定使用哪些验证图像
        if self.use_fewshot_val and self.val_fewshot_ids:
            # 使用筛选后的验证集
            valid_val_ids = set(self.val_fewshot_ids)
        else:
            # 使用完整验证集
            valid_val_ids = None
        
        for item in self.val_info:
            image_id = item['image_id']
            
            # 如果使用筛选后的验证集，检查是否在列表中
            if valid_val_ids is not None and image_id not in valid_val_ids:
                continue
            
            for cls_id in item.get('test_classes', []):
                cls_name = self._id_to_name(cls_id)
                if cls_name in self.val_class_to_images:
                    self.val_class_to_images[cls_name].append(image_id)
        
        # 也尝试从class_lists目录加载（如果存在且使用筛选验证集）
        if self.use_fewshot_val and self.class_lists_dir.exists():
            for cls_name in self.test_classes:
                txt_path = self.class_lists_dir / f'{cls_name}.txt'
                if txt_path.exists():
                    with open(txt_path, 'r') as f:
                        image_ids = [line.strip() for line in f if line.strip()]
                    # 如果class_lists中的数量更多，使用它
                    if len(image_ids) > len(self.val_class_to_images[cls_name]):
                        self.val_class_to_images[cls_name] = image_ids
    
    def _id_to_name(self, cls_id: int) -> str:
        """类别ID转名称"""
        for name, id_ in self.CLASS_NAME_TO_ID.items():
            if id_ == cls_id:
                return name
        return 'unknown'
    
    def _name_to_id(self, cls_name: str) -> int:
        """类别名称转ID"""
        return self.CLASS_NAME_TO_ID.get(cls_name, 0)
    
    def _load_image(self, split: str, image_id: str) -> Image.Image:
        """加载图像"""
        if split == 'train':
            img_path = self.train_dir / 'images' / f'{image_id}.jpg'
        else:
            img_path = self.val_dir / 'images' / f'{image_id}.jpg'
        
        if not img_path.exists():
            raise FileNotFoundError(f"图像不存在: {img_path}")
        
        return Image.open(img_path).convert('RGB')
    
    def _load_mask(self, split: str, image_id: str, 
                   class_name: str = None) -> np.ndarray:
        """
        加载mask
        
        Args:
            split: 'train' 或 'val'
            image_id: 图像ID
            class_name: 如果指定，返回该类别的二值mask；否则返回完整mask
            
        Returns:
            mask: numpy数组
        """
        # 确定mask目录
        if split == 'train':
            if self.use_noleak_for_train:
                mask_dir = self.train_dir / 'masks_noleak'
            else:
                mask_dir = self.train_dir / self.mask_type
        else:
            mask_dir = self.val_dir / self.mask_type
        
        # 如果目录不存在，回退到masks
        if not mask_dir.exists():
            if split == 'train':
                mask_dir = self.train_dir / 'masks'
            else:
                mask_dir = self.val_dir / 'masks'
        
        mask_path = mask_dir / f'{image_id}.png'
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask不存在: {mask_path}")
        
        mask = np.array(Image.open(mask_path))
        
        # 如果指定了类别，返回二值mask
        if class_name is not None:
            cls_id = self._name_to_id(class_name)
            binary_mask = (mask == cls_id).astype(np.uint8)
            return binary_mask
        
        return mask
    
    def _load_image_and_mask(self, split: str, class_name: str,
                             image_id: str) -> Tuple[Image.Image, np.ndarray]:
        """加载图像和对应类别的二值mask"""
        img = self._load_image(split, image_id)
        mask = self._load_mask(split, image_id, class_name)
        return img, mask
    
    def get_support_samples(self, class_name: str, k_shot: int = 1,
                           random_seed: int = None,
                           sorted_list: List[str] = None) -> List[Dict]:
        """
        获取support样本
        
        注意：对于few-shot分割，support通常从验证集中采样
        因为训练集的mask可能不包含测试类别（被mask掉了）
        
        Args:
            class_name: 类别名称
            k_shot: 采样数量
            random_seed: 随机种子
            sorted_list: 预排序的图像ID列表（如果提供，按此顺序选择）
            
        Returns:
            support_samples: List of {'img', 'mask', 'class', 'img_name'}
        """
        if class_name not in self.test_classes:
            raise ValueError(f"类别 {class_name} 不在Fold {self.fold}的测试类别中")
        
        # 获取该类别的图像列表
        image_ids = self.val_class_to_images.get(class_name, [])
        
        if len(image_ids) == 0:
            print(f"   ⚠️ 类别 {class_name} 没有可用样本")
            return []
        
        # 选择样本
        if sorted_list is not None:
            # 使用预排序列表
            available = [img_id for img_id in sorted_list if img_id in image_ids]
            sampled_ids = available[:k_shot]
        elif random_seed is not None:
            # 随机采样
            rng = random.Random(random_seed)
            sampled_ids = rng.sample(image_ids, min(k_shot, len(image_ids)))
        else:
            # 按顺序取前k个
            sampled_ids = image_ids[:k_shot]
        
        # 加载样本
        support_samples = []
        for img_id in sampled_ids:
            try:
                img, mask = self._load_image_and_mask('val', class_name, img_id)
                
                if mask.sum() > 0:
                    support_samples.append({
                        'img': img,
                        'mask': mask,
                        'class': class_name,
                        'img_name': img_id
                    })
                else:
                    print(f"   ⚠️ 跳过空mask: {img_id}")
            except FileNotFoundError as e:
                print(f"   ⚠️ {e}")
                continue
        
        return support_samples
    
    def get_query_samples(self, class_name: str, 
                         exclude_ids: List[str] = None) -> List[Dict]:
        """
        获取query样本（用于评估）
        
        Args:
            class_name: 类别名称
            exclude_ids: 要排除的图像ID列表（通常是support样本）
            
        Returns:
            query_samples: List of {'img', 'mask', 'class', 'img_name'}
        """
        if class_name not in self.test_classes:
            raise ValueError(f"类别 {class_name} 不在Fold {self.fold}的测试类别中")
        
        # 获取该类别的图像列表
        image_ids = self.val_class_to_images.get(class_name, [])
        
        # 排除support样本
        if exclude_ids:
            image_ids = [img_id for img_id in image_ids if img_id not in exclude_ids]
        
        if len(image_ids) == 0:
            print(f"   ⚠️ 类别 {class_name} 没有query样本")
            return []
        
        # 加载样本
        query_samples = []
        for img_id in image_ids:
            try:
                img, mask = self._load_image_and_mask('val', class_name, img_id)
                
                if mask.sum() > 0:
                    query_samples.append({
                        'img': img,
                        'mask': mask,
                        'class': class_name,
                        'img_name': img_id
                    })
            except FileNotFoundError as e:
                print(f"   ⚠️ {e}")
                continue
        
        return query_samples
    
    # 兼容旧接口
    def get_test_samples(self, class_name: str) -> List[Dict]:
        """获取test样本（兼容旧接口）"""
        return self.get_query_samples(class_name)
    
    def get_episode(self, class_name: str, k_shot: int = 1,
                   random_seed: int = None) -> Dict:
        """
        获取一个完整的episode
        
        Args:
            class_name: 类别名称
            k_shot: support样本数量
            random_seed: 随机种子
            
        Returns:
            episode: {'class', 'support', 'query', 'fold', 'k_shot'}
        """
        support_samples = self.get_support_samples(class_name, k_shot, random_seed)
        support_ids = [s['img_name'] for s in support_samples]
        query_samples = self.get_query_samples(class_name, exclude_ids=support_ids)
        
        return {
            'class': class_name,
            'support': support_samples,
            'query': query_samples,
            'test': query_samples,  # 兼容旧代码
            'fold': self.fold,
            'k_shot': k_shot
        }
    
    def get_all_episodes(self, k_shot: int = 1,
                        random_seed: int = None) -> List[Dict]:
        """获取该fold所有类别的episodes"""
        episodes = []
        
        for class_name in self.test_classes:
            episode = self.get_episode(class_name, k_shot, random_seed)
            episodes.append(episode)
            
            print(f"   ✅ {class_name}: "
                  f"{len(episode['support'])} support, "
                  f"{len(episode['query'])} query")
        
        return episodes
    
    def get_class_image_count(self, class_name: str) -> Dict[str, int]:
        """获取某个类别的图像数量统计"""
        return {
            'val': len(self.val_class_to_images.get(class_name, []))
        }
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"Pascal-5i Fold {self.fold} 统计")
        print(f"{'='*60}")
        
        val_mode = "筛选后" if self.use_fewshot_val else "完整"
        print(f"\n验证集模式: {val_mode}")
        
        print(f"\n测试类别 (用于few-shot评估):")
        for cls in self.test_classes:
            count = len(self.val_class_to_images.get(cls, []))
            print(f"   {cls}: {count} 张图像")
        
        total_val = sum(len(self.val_class_to_images.get(cls, [])) 
                       for cls in self.test_classes)
        print(f"\n总计: {total_val} 个(类别, 图像)对")
        
        # 注意：一张图像可能包含多个类别，所以总数可能大于实际图像数
        unique_images = set()
        for cls in self.test_classes:
            unique_images.update(self.val_class_to_images.get(cls, []))
        print(f"唯一图像数: {len(unique_images)}")


# 兼容旧代码的别名
Pascal5iDataset = Pascal5iDatasetNew


def test_dataset():
    """测试数据集加载器"""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("测试Pascal-5i数据集加载器（新格式）")
    print("="*60)
    
    # 加载数据集
    dataset = Pascal5iDatasetNew('pascal5i_output', fold=0)
    dataset.print_statistics()
    
    # 获取一个episode
    episode = dataset.get_episode('aeroplane', k_shot=1, random_seed=42)
    
    print(f"\n测试Episode:")
    print(f"   类别: {episode['class']}")
    print(f"   Support数量: {len(episode['support'])}")
    print(f"   Query数量: {len(episode['query'])}")
    
    if len(episode['support']) > 0 and len(episode['query']) > 0:
        support = episode['support'][0]
        query = episode['query'][0]
        
        print(f"\nSupport样本:")
        print(f"   图像名: {support['img_name']}")
        print(f"   图像大小: {support['img'].size}")
        print(f"   Mask形状: {support['mask'].shape}")
        print(f"   前景像素: {support['mask'].sum()}")
        
        print(f"\nQuery样本:")
        print(f"   图像名: {query['img_name']}")
        print(f"   图像大小: {query['img'].size}")
        print(f"   Mask形状: {query['mask'].shape}")
        print(f"   前景像素: {query['mask'].sum()}")
        
        # 可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Support
        axes[0, 0].imshow(support['img'])
        axes[0, 0].set_title(f"Support: {support['img_name']}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(support['mask'], cmap='gray')
        axes[0, 1].set_title(f"Support Mask ({support['mask'].sum()} px)")
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
        axes[1, 1].set_title(f"Query Mask ({query['mask'].sum()} px)")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(query['img'])
        axes[1, 2].imshow(query['mask'], alpha=0.5, cmap='jet')
        axes[1, 2].set_title("Query Overlay")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('pascal5i_new_test.png', dpi=150)
        print(f"\n✅ 可视化已保存: pascal5i_new_test.png")
        plt.close()


if __name__ == '__main__':
    test_dataset()
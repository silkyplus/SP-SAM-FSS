"""
Support样本自动选择工具 V2 - 多因素综合评分版
=============================================

功能：计算每个类别中train样本与test样本的相似度，自动选择最优support

V2改进：
1. 添加多因素综合评分（相似度 + Mask质量 + 物体质量）
2. 根据类别特性调整各因素权重
3. 支持选择评分模式（similarity_only / comprehensive）

使用方法：
    # 使用纯相似度评分（V1兼容模式）
    python similar_calucate_v2.py --data_root pascal-5 --fold 0 --score_mode similarity_only
    
    # 使用多因素综合评分（V2新模式）
    python similar_calucate_v2.py --data_root pascal-5 --fold 0 --score_mode comprehensive
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse
import json

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SupportSelector:
    """
    Support样本自动选择器 V2
    
    计算train样本与test样本的相似度，支持多因素综合评分
    """
    
    # PASCAL-5i的类别划分
    FOLD_CLASSES = {
        0: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
        1: ['bus', 'car', 'cat', 'chair', 'cow'],
        2: ['diningtable', 'dog', 'horse', 'motorbike', 'person'],
        3: ['pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    }
    
    # ========================================================
    # V2新增：类别特性配置（根据实验结果调整权重）
    # ========================================================
    # 基于你的实验结果分析：
    # - cat, cow, aeroplane: 相似度选择有效，保持高相似度权重
    # - dog, car, bottle, bus, bicycle: 相似度选择失效，降低相似度权重
    CLASS_CONFIG = {
        # 类内变化小的类别：相似度权重高（这些类别V1效果好）
        'aeroplane': {'sim_weight': 0.6, 'mask_weight': 0.25, 'obj_weight': 0.15},
        'cat': {'sim_weight': 0.6, 'mask_weight': 0.25, 'obj_weight': 0.15},
        'cow': {'sim_weight': 0.6, 'mask_weight': 0.25, 'obj_weight': 0.15},
        'sheep': {'sim_weight': 0.6, 'mask_weight': 0.25, 'obj_weight': 0.15},
        'train': {'sim_weight': 0.55, 'mask_weight': 0.25, 'obj_weight': 0.20},
        'bird': {'sim_weight': 0.5, 'mask_weight': 0.30, 'obj_weight': 0.20},
        'boat': {'sim_weight': 0.5, 'mask_weight': 0.30, 'obj_weight': 0.20},
        'horse': {'sim_weight': 0.5, 'mask_weight': 0.30, 'obj_weight': 0.20},
        
        # 类内变化大的类别：降低相似度权重，提高mask/物体质量权重（这些类别V1效果差）
        'dog': {'sim_weight': 0.25, 'mask_weight': 0.40, 'obj_weight': 0.35},
        'car': {'sim_weight': 0.25, 'mask_weight': 0.40, 'obj_weight': 0.35},
        'bottle': {'sim_weight': 0.25, 'mask_weight': 0.40, 'obj_weight': 0.35},
        'bus': {'sim_weight': 0.30, 'mask_weight': 0.40, 'obj_weight': 0.30},
        'bicycle': {'sim_weight': 0.30, 'mask_weight': 0.40, 'obj_weight': 0.30},
        'person': {'sim_weight': 0.30, 'mask_weight': 0.35, 'obj_weight': 0.35},
        'motorbike': {'sim_weight': 0.35, 'mask_weight': 0.35, 'obj_weight': 0.30},
        
        # 小物体/难分割类别：物体质量权重高
        'chair': {'sim_weight': 0.25, 'mask_weight': 0.35, 'obj_weight': 0.40},
        'pottedplant': {'sim_weight': 0.30, 'mask_weight': 0.35, 'obj_weight': 0.35},
        'tvmonitor': {'sim_weight': 0.35, 'mask_weight': 0.35, 'obj_weight': 0.30},
        
        # 大物体/背景复杂类别：mask质量权重高
        'diningtable': {'sim_weight': 0.20, 'mask_weight': 0.50, 'obj_weight': 0.30},
        'sofa': {'sim_weight': 0.30, 'mask_weight': 0.40, 'obj_weight': 0.30},
        
        # 默认配置
        'default': {'sim_weight': 0.40, 'mask_weight': 0.35, 'obj_weight': 0.25},
    }
    
    def __init__(self, data_root: str, fold: int, device: str = 'cuda'):
        """
        Args:
            data_root: 数据集根目录 (pascal-5)
            fold: fold编号 (0-3)
            device: 计算设备
        """
        self.data_root = Path(data_root)
        self.fold = fold
        self.device = device
        self.fold_dir = self.data_root / str(fold)
        
        if not self.fold_dir.exists():
            raise FileNotFoundError(f"Fold目录不存在: {self.fold_dir}")
        
        self.train_dir = self.fold_dir / 'train'
        self.test_dir = self.fold_dir / 'test'
        self.classes = self.FOLD_CLASSES[fold]
        
        # 模型（延迟加载）
        self.dino_model = None
        self.dino_transform = None
        
        print(f"\n{'='*60}")
        print(f"📁 Support样本选择器 V2 (多因素评分)")
        print(f"{'='*60}")
        print(f"   数据集路径: {self.data_root}")
        print(f"   Fold: {fold}")
        print(f"   类别: {', '.join(self.classes)}")
        print(f"   设备: {device}")
    
    def load_dino_model(self, model_name: str = 'dinov2_vitl14', 
                        weights_path: str = None,
                        dino_path: str = 'facebookresearch_dinov2_main'):
        """加载DINO模型"""
        from torchvision import transforms
        
        print(f"\n📦 加载DINO模型: {model_name}")
        
        try:
            if weights_path:
                self.dino_model = torch.hub.load(
                    repo_or_dir=dino_path,
                    model=model_name,
                    source='local',
                    pretrained=False
                )
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.dino_model.load_state_dict(state_dict, strict=False)
                print(f"   ✅ 本地权重加载成功: {weights_path}")
            else:
                self.dino_model = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov2",
                    model=model_name
                )
                print(f"   ✅ 从torch.hub加载成功")
            
            self.dino_model = self.dino_model.to(torch.bfloat16)
            self.dino_model.eval()
            self.dino_model.to(self.device)
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            raise
        
        # 预处理变换
        self.dino_transform = transforms.Compose([
            transforms.Resize(size=(518, 518), 
                            interpolation=transforms.InterpolationMode.BICUBIC, 
                            antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        print(f"   ✅ DINO模型准备完成\n")
    
    def _load_image_list(self, txt_path: Path) -> List[str]:
        """从txt文件加载图像列表"""
        if not txt_path.exists():
            return []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]
    
    def _fill_contour_mask(self, mask_gray: np.ndarray) -> np.ndarray:
        """将轮廓mask填充成区域mask"""
        binary = (mask_gray > 128).astype(np.uint8) * 255
        if binary.sum() < 100:
            binary = (mask_gray > 10).astype(np.uint8) * 255
        if binary.sum() < 100:
            return np.zeros_like(mask_gray, dtype=np.uint8)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask_gray, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                cv2.drawContours(filled_mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        return filled_mask
    
    def _load_image_and_mask(self, split: str, class_name: str, 
                            image_name: str) -> Tuple[Image.Image, np.ndarray]:
        """加载图像和mask"""
        split_dir = self.train_dir if split == 'train' else self.test_dir
        
        img_path = split_dir / 'origin' / f'{image_name}.jpg'
        if not img_path.exists():
            raise FileNotFoundError(f"图像不存在: {img_path}")
        img = Image.open(img_path).convert('RGB')
        
        mask_path = split_dir / 'groundtruth' / f'{image_name}.png'
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask不存在: {mask_path}")
        
        mask_img = cv2.imread(str(mask_path))
        if mask_img is None:
            raise ValueError(f"无法读取mask: {mask_path}")
        
        if len(mask_img.shape) == 3:
            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_img
        
        binary_mask = self._fill_contour_mask(mask_gray)
        return img, binary_mask
    
    # ========================================================
    # V2新增：Mask质量评估
    # ========================================================
    def compute_mask_quality_score(self, mask: np.ndarray) -> Dict[str, float]:
        """
        计算mask质量分数
        
        考虑因素：
        1. 覆盖率：5%-40%为最佳
        2. 连通性：单个连通区域为最佳
        3. 紧凑度：形状接近圆形/矩形
        4. 边界规则度：边界平滑、规则
        """
        H, W = mask.shape
        total_pixels = H * W
        fg_pixels = mask.sum()
        
        # 1. 覆盖率分数 (5%-40%最佳，太小或太大都扣分)
        coverage = fg_pixels / total_pixels
        if coverage < 0.01:
            coverage_score = 0.0
        elif coverage < 0.05:
            coverage_score = coverage / 0.05 * 0.5  # 0-0.5
        elif coverage <= 0.40:
            coverage_score = 1.0  # 最佳范围
        elif coverage <= 0.60:
            coverage_score = 1.0 - (coverage - 0.40) / 0.20 * 0.5  # 1.0-0.5
        else:
            coverage_score = max(0.2, 0.5 - (coverage - 0.60) / 0.40 * 0.3)  # 0.5-0.2
        
        # 2. 连通性分数 (单个连通区域为最佳)
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_uint8)
        num_components = num_labels - 1  # 减去背景
        
        if num_components == 0:
            connectivity_score = 0.0
        elif num_components == 1:
            connectivity_score = 1.0
        elif num_components == 2:
            connectivity_score = 0.7
        elif num_components <= 4:
            connectivity_score = 0.5
        else:
            connectivity_score = 0.3
        
        # 3. 紧凑度分数 (使用轮廓面积与周长的比值)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                # 圆的紧凑度为1，其他形状<1
                circularity = 4 * np.pi * area / (perimeter ** 2)
                compactness_score = min(circularity, 1.0)
            else:
                compactness_score = 0.0
        else:
            compactness_score = 0.0
        
        # 4. 边界规则度 (使用多边形近似)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 顶点数量适中(4-20)表示边界规则
            num_vertices = len(approx)
            if 4 <= num_vertices <= 20:
                boundary_score = 1.0
            elif num_vertices < 4:
                boundary_score = 0.5
            elif num_vertices <= 50:
                boundary_score = 1.0 - (num_vertices - 20) / 30 * 0.5
            else:
                boundary_score = 0.3
        else:
            boundary_score = 0.0
        
        # 综合分数
        total_score = (
            coverage_score * 0.35 +
            connectivity_score * 0.25 +
            compactness_score * 0.20 +
            boundary_score * 0.20
        )
        
        return {
            'total': total_score,
            'coverage': coverage,
            'coverage_score': coverage_score,
            'connectivity_score': connectivity_score,
            'compactness_score': compactness_score,
            'boundary_score': boundary_score,
            'num_components': num_components
        }
    
    # ========================================================
    # V2新增：物体质量评估
    # ========================================================
    def compute_object_quality_score(self, mask: np.ndarray) -> Dict[str, float]:
        """
        计算目标物体质量分数
        
        考虑因素：
        1. 位置：接近图像中心为佳
        2. 大小：适中大小为佳
        3. 完整性：不被边缘截断
        """
        H, W = mask.shape
        
        if mask.sum() == 0:
            return {'total': 0.0, 'position_score': 0.0, 'size_score': 0.0, 'completeness_score': 0.0}
        
        # 获取目标区域
        coords = np.argwhere(mask > 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 1. 位置分数 (中心区域为佳)
        center_y = (y_min + y_max) / 2
        center_x = (x_min + x_max) / 2
        
        # 计算到图像中心的距离
        img_center_y, img_center_x = H / 2, W / 2
        dist_to_center = np.sqrt((center_y - img_center_y)**2 + (center_x - img_center_x)**2)
        max_dist = np.sqrt(img_center_y**2 + img_center_x**2)
        
        position_score = 1.0 - (dist_to_center / max_dist)
        
        # 2. 大小分数 (bbox占图像10%-70%为佳)
        bbox_area = (y_max - y_min) * (x_max - x_min)
        bbox_ratio = bbox_area / (H * W)
        
        if bbox_ratio < 0.05:
            size_score = bbox_ratio / 0.05 * 0.5
        elif bbox_ratio <= 0.70:
            size_score = 1.0
        else:
            size_score = max(0.3, 1.0 - (bbox_ratio - 0.70) / 0.30)
        
        # 3. 完整性分数 (不被边缘截断)
        border_margin = 5
        touches_border = (
            y_min < border_margin or 
            x_min < border_margin or 
            y_max > H - border_margin or 
            x_max > W - border_margin
        )
        
        if touches_border:
            # 计算被截断的程度
            border_pixels = 0
            border_pixels += (mask[:border_margin, :] > 0).sum()
            border_pixels += (mask[-border_margin:, :] > 0).sum()
            border_pixels += (mask[:, :border_margin] > 0).sum()
            border_pixels += (mask[:, -border_margin:] > 0).sum()
            
            border_ratio = border_pixels / max(mask.sum(), 1)
            completeness_score = max(0.3, 1.0 - border_ratio * 2)
        else:
            completeness_score = 1.0
        
        total_score = (
            position_score * 0.3 +
            size_score * 0.4 +
            completeness_score * 0.3
        )
        
        return {
            'total': total_score,
            'position_score': position_score,
            'size_score': size_score,
            'completeness_score': completeness_score,
            'bbox_ratio': bbox_ratio
        }
    
    # ========================================================
    # V2新增：综合评分计算
    # ========================================================
    def compute_comprehensive_score(self, 
                                     similarity_score: float,
                                     mask_quality: Dict,
                                     object_quality: Dict,
                                     class_name: str) -> float:
        """
        计算综合评分
        
        根据类别特性调整各因素权重
        """
        config = self.CLASS_CONFIG.get(class_name, self.CLASS_CONFIG['default'])
        
        sim_weight = config['sim_weight']
        mask_weight = config['mask_weight']
        obj_weight = config['obj_weight']
        
        # 归一化相似度到0-1（假设相似度范围是0.3-0.9）
        norm_sim = max(0, min(1, (similarity_score - 0.3) / 0.6))
        
        total_score = (
            norm_sim * sim_weight +
            mask_quality['total'] * mask_weight +
            object_quality['total'] * obj_weight
        )
        
        return total_score
    
    def extract_masked_features(self, img: Image.Image, mask: np.ndarray) -> torch.Tensor:
        """
        提取masked特征（只提取前景区域的特征）
        
        Args:
            img: PIL Image
            mask: [H, W] binary mask
            
        Returns:
            masked_prototype: [C] 前景区域的平均特征向量
        """
        img_tensor = self.dino_transform(img)[None].to(self.device)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            features = self.dino_model.get_intermediate_layers(
                img_tensor.to(torch.bfloat16)
            )[0]  # [1, H*W, C]
            
            h = w = int(features.shape[1]**0.5)
            feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # [1, C, H, W]
            feature_map = feature_map.float().squeeze(0)  # [C, H, W]
        
        # 下采样mask到特征图大小
        C, H_feat, W_feat = feature_map.shape
        resized_mask = cv2.resize(
            mask.astype(np.float32),
            (W_feat, H_feat),
            interpolation=cv2.INTER_NEAREST
        )
        resized_mask = (resized_mask > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(resized_mask).to(self.device)
        
        # 计算前景区域的平均特征
        mask_expanded = mask_tensor.unsqueeze(0)  # [1, H, W]
        masked_features = feature_map * mask_expanded  # [C, H, W]
        
        num_fg_pixels = mask_tensor.sum()
        if num_fg_pixels > 0:
            prototype = masked_features.sum(dim=[1, 2]) / num_fg_pixels  # [C]
        else:
            prototype = torch.zeros(C, device=self.device)
        
        # L2归一化
        prototype = F.normalize(prototype, p=2, dim=0)
        
        return prototype
    
    def extract_spatial_features(self, img: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取空间特征（保留空间结构）
        """
        img_tensor = self.dino_transform(img)[None].to(self.device)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            features = self.dino_model.get_intermediate_layers(
                img_tensor.to(torch.bfloat16)
            )[0]
            
            h = w = int(features.shape[1]**0.5)
            feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)
            feature_map = feature_map.float().squeeze(0)
        
        C, H_feat, W_feat = feature_map.shape
        resized_mask = cv2.resize(
            mask.astype(np.float32),
            (W_feat, H_feat),
            interpolation=cv2.INTER_NEAREST
        )
        resized_mask = (resized_mask > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(resized_mask).to(self.device)
        
        return feature_map, mask_tensor
    
    def compute_prototype_similarity(self, proto1: torch.Tensor, proto2: torch.Tensor) -> float:
        """计算两个原型向量的余弦相似度"""
        return float(torch.dot(proto1, proto2).cpu())
    
    def compute_spatial_similarity(self, feat1: torch.Tensor, mask1: torch.Tensor,
                                   feat2: torch.Tensor, mask2: torch.Tensor,
                                   method: str = 'max') -> float:
        """
        计算两个样本的空间特征相似度（像素级匹配）
        """
        C, H, W = feat1.shape
        
        # 获取前景像素特征
        mask1_flat = mask1.reshape(-1) > 0.5
        mask2_flat = mask2.reshape(-1) > 0.5
        
        feat1_flat = feat1.reshape(C, -1).T  # [H*W, C]
        feat2_flat = feat2.reshape(C, -1).T
        
        fg_feat1 = feat1_flat[mask1_flat]  # [N1, C]
        fg_feat2 = feat2_flat[mask2_flat]  # [N2, C]
        
        if fg_feat1.shape[0] == 0 or fg_feat2.shape[0] == 0:
            return 0.0
        
        # L2归一化
        fg_feat1 = F.normalize(fg_feat1, p=2, dim=1)
        fg_feat2 = F.normalize(fg_feat2, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(fg_feat1, fg_feat2.T)  # [N1, N2]
        
        if method == 'max':
            max_sim_1to2, _ = similarity_matrix.max(dim=1)
            max_sim_2to1, _ = similarity_matrix.max(dim=0)
            similarity = (max_sim_1to2.mean() + max_sim_2to1.mean()) / 2
        else:
            similarity = similarity_matrix.mean()
        
        return float(similarity.cpu())
    
    def compute_train_test_similarity(self, class_name: str, 
                                      method: str = 'prototype',
                                      score_mode: str = 'comprehensive',
                                      verbose: bool = True) -> Dict:
        """
        计算某个类别中所有train样本与test样本的相似度
        
        Args:
            class_name: 类别名称
            method: 'prototype' (原型向量) 或 'spatial' (空间匹配)
            score_mode: 'similarity_only' (纯相似度) 或 'comprehensive' (综合评分)
            verbose: 是否显示详细进度
            
        Returns:
            results: 包含相似度矩阵和综合评分的字典
        """
        # 加载train和test样本列表
        train_txt = self.train_dir / f'{class_name}.txt'
        test_txt = self.test_dir / f'{class_name}.txt'
        
        train_names = self._load_image_list(train_txt)
        test_names = self._load_image_list(test_txt)
        
        if len(train_names) == 0 or len(test_names) == 0:
            print(f"   ⚠️  {class_name}: train={len(train_names)}, test={len(test_names)}")
            return None
        
        if verbose:
            print(f"\n📊 处理类别: {class_name}")
            print(f"   Train样本: {len(train_names)}")
            print(f"   Test样本: {len(test_names)}")
            if score_mode == 'comprehensive':
                config = self.CLASS_CONFIG.get(class_name, self.CLASS_CONFIG['default'])
                print(f"   权重配置: sim={config['sim_weight']}, mask={config['mask_weight']}, obj={config['obj_weight']}")
        
        # 提取所有train样本的特征
        train_data = []  # [(name, feature, mask_ratio, mask_quality, obj_quality, original_mask), ...]
        
        if verbose:
            print(f"   提取Train特征...")
        
        for name in tqdm(train_names, desc=f"   Train", disable=not verbose):
            try:
                img, mask = self._load_image_and_mask('train', class_name, name)
                mask_ratio = mask.sum() / mask.size
                
                if mask.sum() < 100:  # 跳过mask太小的样本
                    continue
                
                if method == 'prototype':
                    feat = self.extract_masked_features(img, mask)
                else:
                    feat, mask_ds = self.extract_spatial_features(img, mask)
                    feat = (feat, mask_ds)
                
                # V2新增：计算mask和物体质量
                mask_quality = self.compute_mask_quality_score(mask)
                obj_quality = self.compute_object_quality_score(mask)
                
                train_data.append((name, feat, mask_ratio, mask_quality, obj_quality))
                
            except Exception as e:
                if verbose:
                    print(f"      ⚠️ 跳过 {name}: {e}")
                continue
        
        # 提取所有test样本的特征
        test_data = []
        
        if verbose:
            print(f"   提取Test特征...")
        
        for name in tqdm(test_names, desc=f"   Test", disable=not verbose):
            try:
                img, mask = self._load_image_and_mask('test', class_name, name)
                mask_ratio = mask.sum() / mask.size
                
                if mask.sum() < 100:
                    continue
                
                if method == 'prototype':
                    feat = self.extract_masked_features(img, mask)
                else:
                    feat, mask_ds = self.extract_spatial_features(img, mask)
                    feat = (feat, mask_ds)
                
                test_data.append((name, feat, mask_ratio))
                
            except Exception as e:
                if verbose:
                    print(f"      ⚠️ 跳过 {name}: {e}")
                continue
        
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"   ⚠️  有效样本不足: train={len(train_data)}, test={len(test_data)}")
            return None
        
        # 计算相似度矩阵
        if verbose:
            print(f"   计算相似度矩阵...")
        
        n_train = len(train_data)
        n_test = len(test_data)
        similarity_matrix = np.zeros((n_train, n_test))
        
        for i, (train_name, train_feat, _, _, _) in enumerate(tqdm(train_data, desc="   计算相似度", disable=not verbose)):
            for j, (test_name, test_feat, _) in enumerate(test_data):
                if method == 'prototype':
                    sim = self.compute_prototype_similarity(train_feat, test_feat)
                else:
                    sim = self.compute_spatial_similarity(
                        train_feat[0], train_feat[1],
                        test_feat[0], test_feat[1]
                    )
                similarity_matrix[i, j] = sim
        
        # 计算每个train样本的平均相似度
        train_avg_sim = similarity_matrix.mean(axis=1)
        
        # V2新增：计算综合评分
        comprehensive_scores = []
        for i, (name, feat, mask_ratio, mask_quality, obj_quality) in enumerate(train_data):
            if score_mode == 'comprehensive':
                comp_score = self.compute_comprehensive_score(
                    train_avg_sim[i], mask_quality, obj_quality, class_name
                )
            else:
                # 纯相似度模式
                comp_score = train_avg_sim[i]
            comprehensive_scores.append(comp_score)
        
        comprehensive_scores = np.array(comprehensive_scores)
        
        results = {
            'class_name': class_name,
            'train_samples': [(d[0], d[2], d[3], d[4]) for d in train_data],  # (name, mask_ratio, mask_quality, obj_quality)
            'test_samples': [(d[0], d[2]) for d in test_data],
            'similarity_matrix': similarity_matrix,
            'train_avg_sim': train_avg_sim,
            'comprehensive_scores': comprehensive_scores,  # V2新增
            'score_mode': score_mode
        }
        
        return results
    
    def get_top_k_supports(self, results: Dict, top_k: int = 5) -> List[Dict]:
        """
        获取评分最高的前K个train样本
        
        Args:
            results: compute_train_test_similarity的返回值
            top_k: 选择前K个
            
        Returns:
            top_supports: [{'name': ..., 'similarity': ..., 'comprehensive_score': ..., ...}, ...]
        """
        train_samples = results['train_samples']
        train_avg_sim = results['train_avg_sim']
        comprehensive_scores = results['comprehensive_scores']
        score_mode = results['score_mode']
        
        # 按综合评分排序
        sorted_indices = np.argsort(comprehensive_scores)[::-1]  # 降序
        
        top_supports = []
        for idx in sorted_indices[:top_k]:
            name, mask_ratio, mask_quality, obj_quality = train_samples[idx]
            avg_sim = train_avg_sim[idx]
            comp_score = comprehensive_scores[idx]
            
            top_supports.append({
                'name': name,
                'similarity': float(avg_sim),
                'comprehensive_score': float(comp_score),
                'mask_ratio': float(mask_ratio),
                'mask_quality_score': float(mask_quality['total']),
                'object_quality_score': float(obj_quality['total']),
                'mask_coverage': float(mask_quality['coverage']),
                'mask_components': int(mask_quality['num_components'])
            })
        
        return top_supports
    
    def analyze_all_classes(self, method: str = 'prototype', 
                           score_mode: str = 'comprehensive',
                           top_k: int = 5,
                           output_file: str = None) -> Dict:
        """
        分析所有类别，输出最优support
        
        Args:
            method: 'prototype' 或 'spatial'
            score_mode: 'similarity_only' (纯相似度V1) 或 'comprehensive' (综合评分V2)
            top_k: 每个类别选择前K个
            output_file: 输出文件路径（可选）
            
        Returns:
            all_results: {class_name: top_supports, ...}
        """
        print(f"\n{'='*60}")
        print(f"🔍 分析所有类别的Support选择")
        print(f"{'='*60}")
        print(f"   方法: {method}")
        print(f"   评分模式: {score_mode}")
        print(f"   Top-K: {top_k}")
        
        all_results = {}
        
        for class_name in self.classes:
            results = self.compute_train_test_similarity(
                class_name, method=method, score_mode=score_mode
            )
            
            if results is None:
                continue
            
            top_supports = self.get_top_k_supports(results, top_k=top_k)
            
            # 获取所有样本的评分（用于生成txt）
            all_scores = []
            for i, (name, mask_ratio, mask_quality, obj_quality) in enumerate(results['train_samples']):
                all_scores.append({
                    'name': name,
                    'similarity': float(results['train_avg_sim'][i]),
                    'comprehensive_score': float(results['comprehensive_scores'][i]),
                    'mask_ratio': float(mask_ratio),
                    'mask_quality_score': float(mask_quality['total']),
                    'object_quality_score': float(obj_quality['total'])
                })
            
            all_results[class_name] = {
                'top_supports': top_supports,
                'n_train': len(results['train_samples']),
                'n_test': len(results['test_samples']),
                'all_scores': all_scores,
                'config': self.CLASS_CONFIG.get(class_name, self.CLASS_CONFIG['default'])
            }
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"📋 最优Support选择结果 (Fold {self.fold})")
        print(f"{'='*60}")
        
        for class_name, data in all_results.items():
            config = data['config']
            print(f"\n🏷️  {class_name}:")
            print(f"   Train样本数: {data['n_train']}, Test样本数: {data['n_test']}")
            if score_mode == 'comprehensive':
                print(f"   权重: sim={config['sim_weight']}, mask={config['mask_weight']}, obj={config['obj_weight']}")
            print(f"   推荐的Top-{top_k} Support:")
            
            for rank, support in enumerate(data['top_supports'], 1):
                print(f"      {rank}. {support['name']}")
                if score_mode == 'comprehensive':
                    print(f"         综合分: {support['comprehensive_score']:.4f} | "
                          f"相似度: {support['similarity']:.4f} | "
                          f"Mask质量: {support['mask_quality_score']:.4f} | "
                          f"物体质量: {support['object_quality_score']:.4f}")
                else:
                    print(f"         相似度: {support['similarity']:.4f}, Mask覆盖率: {support['mask_ratio']*100:.1f}%")
        
        # 保存结果
        if output_file:
            save_data = {
                'fold': self.fold,
                'method': method,
                'score_mode': score_mode,
                'top_k': top_k,
                'results': {}
            }
            
            for class_name, data in all_results.items():
                save_data['results'][class_name] = {
                    'config': data['config'],
                    'recommended_supports': [s['name'] for s in data['top_supports']],
                    'top_supports_detail': data['top_supports'],
                    'all_train_scores': data['all_scores']
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ 结果已保存到: {output_file}")
        
        return all_results
    
    def generate_txt_files(self, all_results: Dict, output_dir: str = None):
        """
        生成排序后的txt文件，可直接用于评估
        """
        if output_dir is None:
            output_dir = self.fold_dir / 'train_sorted_v2'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📝 生成排序后的txt文件...")
        
        for class_name, data in all_results.items():
            # 按综合评分排序所有train样本
            sorted_samples = sorted(
                data['all_scores'],
                key=lambda x: x['comprehensive_score'],  # 按综合评分排序
                reverse=True
            )
            
            # 写入txt文件
            txt_path = output_dir / f'{class_name}.txt'
            with open(txt_path, 'w') as f:
                for sample in sorted_samples:
                    f.write(f"{sample['name']}\n")
            
            print(f"   ✅ {class_name}.txt ({len(sorted_samples)} 样本)")
        
        print(f"\n✅ 排序后的txt文件已保存到: {output_dir}")
        print(f"💡 提示: 将此目录下的txt文件复制到train目录，即可使用排序后的support")


def main():
    parser = argparse.ArgumentParser(description='Support样本自动选择工具 V2')
    parser.add_argument('--data_root', type=str, default='pascal-5',
                       help='数据集根目录')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3],
                       help='Fold编号')
    parser.add_argument('--top_k', type=int, default=5,
                       help='选择前K个最优support')
    parser.add_argument('--method', type=str, default='prototype',
                       choices=['prototype', 'spatial'],
                       help='相似度计算方法: prototype(原型向量) 或 spatial(空间匹配)')
    # V2新增：评分模式
    parser.add_argument('--score_mode', type=str, default='comprehensive',
                       choices=['similarity_only', 'comprehensive'],
                       help='评分模式: similarity_only(纯相似度V1) 或 comprehensive(综合评分V2)')
    parser.add_argument('--dino_model', type=str, default='dinov3_vitb16',
                       help='DINO模型名称')
    parser.add_argument('--dino_weights', type=str, default=r'D:\vscode\python_project\dinov3-main-\dinov3-main\weights\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                       help='DINO权重路径（可选）')
    parser.add_argument('--dino_path', type=str, default='dinov3_main',
                       help='DINO项目路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径')
    parser.add_argument('--generate_txt', action='store_true',
                       help='是否生成排序后的txt文件')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 创建选择器
    selector = SupportSelector(
        data_root=args.data_root,
        fold=args.fold,
        device=args.device
    )
    
    # 加载DINO模型
    selector.load_dino_model(
        model_name=args.dino_model,
        weights_path=args.dino_weights,
        dino_path=args.dino_path
    )
    
    # 设置输出文件
    if args.output is None:
        args.output = f'support_selection_fold{args.fold}_{args.method}_{args.score_mode}.json'
    
    # 分析所有类别
    all_results = selector.analyze_all_classes(
        method=args.method,
        score_mode=args.score_mode,
        top_k=args.top_k,
        output_file=args.output
    )
    
    # 生成排序后的txt文件
    if args.generate_txt:
        selector.generate_txt_files(all_results)


if __name__ == '__main__':
    main()
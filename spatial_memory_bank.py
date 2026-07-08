"""
方案A: 空间特征 Memory Bank + 像素级匹配
=======================================

核心改进：
1. 保留完整的空间特征图，而非压缩成单个原型向量
2. 使用像素级匹配（pixel-wise matching）计算相似度
3. 支持多种匹配策略：最大相似度、平均相似度、加权相似度

参考论文：FS-SAM2 的 memory attention 机制
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# 1. 空间特征 Memory Bank 构建
# ============================================================

def build_spatial_memory_bank(reference_data, dino_model, dino_transform, 
                               device='cpu', max_features_per_class=5):
    """
    构建保留空间特征的 Memory Bank
    
    与原始方法的区别：
    - 原始方法：将所有特征平均成一个向量 [C]
    - 新方法：保留完整的空间特征图 [C, H, W] 和对应的 mask
    
    Args:
        reference_data: 参考数据字典
            {
                'class_name': [
                    {'img_path': str, 'masks': [mask1, mask2, ...]},
                    ...
                ]
            }
        dino_model: DINO 模型 (v2 或 v3)
        dino_transform: 图像预处理变换
        device: 计算设备
        max_features_per_class: 每个类别最多保留的特征数量（控制内存）
    
    Returns:
        memory_bank: 空间特征记忆库
            {
                'class_name': {
                    'features': List[Tensor[C, H, W]],  # 特征图列表
                    'masks': List[Tensor[H, W]],        # 对应的 mask 列表
                    'img_paths': List[str]              # 图像路径（用于调试）
                }
            }
    """
    memory_bank = {}
    
    print("="*60)
    print("🔨 构建空间特征 Memory Bank")
    print("="*60)
    
    for class_name, ref_images_masks in tqdm(reference_data.items(), desc="Processing classes"):
        class_features = []
        class_masks = []
        class_img_paths = []
        
        for data in ref_images_masks:
            img_path = data['img_path']
            instance_masks = data['masks']
            
            # 加载图像
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = dino_transform(img_pil)[None].to(device)
            
            # 提取完整的空间特征图
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                features = dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
                h = w = int(features.shape[1]**0.5)
                # [1, H*W, C] -> [1, C, H, W]
                feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)
                feature_map = feature_map.float()  # 转回 float32
            
            # 处理每个实例 mask
            for mask_np in instance_masks:
                if mask_np.sum() == 0:
                    continue
                    
                # 调整 mask 到特征图大小
                resized_mask = cv2.resize(
                    mask_np.astype(np.float32), 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                )
                resized_mask = (resized_mask > 0.5).astype(np.float32)
                
                # 检查 mask 是否有效
                if resized_mask.sum() < 4:  # 至少4个像素
                    continue
                
                # 保存特征图和 mask
                class_features.append(feature_map.squeeze(0).cpu())  # [C, H, W]
                class_masks.append(torch.from_numpy(resized_mask))    # [H, W]
                class_img_paths.append(img_path)
                
                # 控制每个类别的特征数量
                if len(class_features) >= max_features_per_class:
                    break
            
            if len(class_features) >= max_features_per_class:
                break
        
        if len(class_features) > 0:
            memory_bank[class_name] = {
                'features': class_features,
                'masks': class_masks,
                'img_paths': class_img_paths
            }
            print(f"   ✅ {class_name}: {len(class_features)} 个空间特征")
        else:
            print(f"   ⚠️  {class_name}: 没有有效的特征")
    
    print("="*60)
    print(f"✅ Memory Bank 构建完成")
    print(f"   类别数: {len(memory_bank)}")
    print(f"   总特征数: {sum(len(v['features']) for v in memory_bank.values())}")
    print("="*60)
    
    return memory_bank


# ============================================================
# 2. 像素级匹配函数
# ============================================================

def pixel_wise_similarity(query_features, support_features, support_mask, 
                          method='max', temperature=0.1):
    """
    计算查询特征与支持特征之间的像素级相似度
    
    Args:
        query_features: [C, H_q, W_q] 查询图像的特征图
        support_features: [C, H_s, W_s] 支持图像的特征图
        support_mask: [H_s, W_s] 支持图像的前景 mask
        method: 相似度聚合方法
            - 'max': 取与前景像素的最大相似度（推荐，更鲁棒）
            - 'mean': 取平均相似度
            - 'topk': 取 top-k 相似度的平均
            - 'weighted': 使用 softmax 加权
        temperature: softmax 温度参数（用于 'weighted' 方法）
    
    Returns:
        similarity_map: [H_q, W_q] 每个查询像素的相似度得分
    """
    C, H_q, W_q = query_features.shape
    C_s, H_s, W_s = support_features.shape
    
    # 展平特征
    query_flat = query_features.reshape(C, -1).T  # [H_q*W_q, C]
    support_flat = support_features.reshape(C_s, -1).T  # [H_s*W_s, C]
    
    # L2 归一化
    query_flat = F.normalize(query_flat, p=2, dim=1)
    support_flat = F.normalize(support_flat, p=2, dim=1)
    
    # 获取前景像素的特征
    mask_flat = support_mask.reshape(-1) > 0.5  # [H_s*W_s]
    fg_features = support_flat[mask_flat]  # [N_fg, C]
    
    if fg_features.shape[0] == 0:
        return torch.zeros(H_q, W_q)
    
    # 计算相似度矩阵: [H_q*W_q, N_fg]
    similarity = torch.mm(query_flat, fg_features.T)
    
    # 根据方法聚合相似度
    if method == 'max':
        # 每个查询像素取与所有前景像素的最大相似度
        pixel_sim, _ = similarity.max(dim=1)  # [H_q*W_q]
        
    elif method == 'mean':
        # 取平均相似度
        pixel_sim = similarity.mean(dim=1)  # [H_q*W_q]
        
    elif method == 'topk':
        # 取 top-k 相似度的平均
        k = min(5, fg_features.shape[0])
        topk_sim, _ = similarity.topk(k, dim=1)  # [H_q*W_q, k]
        pixel_sim = topk_sim.mean(dim=1)  # [H_q*W_q]
        
    elif method == 'weighted':
        # 使用 softmax 加权
        weights = F.softmax(similarity / temperature, dim=1)  # [H_q*W_q, N_fg]
        pixel_sim = (similarity * weights).sum(dim=1)  # [H_q*W_q]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 重塑为空间形状
    similarity_map = pixel_sim.reshape(H_q, W_q)
    
    return similarity_map


def compute_class_similarity(query_features, memory_bank, class_name, 
                             method='max', aggregation='max', device='cpu'):
    """
    计算查询特征与某个类别的相似度
    
    Args:
        query_features: [C, H, W] 查询图像的特征图
        memory_bank: 空间特征记忆库
        class_name: 类别名称
        method: 像素级匹配方法
        aggregation: 多个支持图像的聚合方式
            - 'max': 取所有支持图像的最大相似度
            - 'mean': 取平均
        device: 计算设备
    
    Returns:
        similarity_map: [H, W] 相似度图
        confidence: float 整体置信度得分
    """
    if class_name not in memory_bank:
        H, W = query_features.shape[1], query_features.shape[2]
        return torch.zeros(H, W), 0.0
    
    class_data = memory_bank[class_name]
    support_features_list = class_data['features']
    support_masks_list = class_data['masks']
    
    all_similarity_maps = []
    
    query_features = query_features.to(device)
    
    for sup_feat, sup_mask in zip(support_features_list, support_masks_list):
        sup_feat = sup_feat.to(device)
        sup_mask = sup_mask.to(device)
        
        sim_map = pixel_wise_similarity(
            query_features, sup_feat, sup_mask, method=method
        )
        all_similarity_maps.append(sim_map)
    
    # 聚合多个支持图像的相似度图
    stacked = torch.stack(all_similarity_maps, dim=0)  # [K, H, W]
    
    if aggregation == 'max':
        final_map, _ = stacked.max(dim=0)  # [H, W]
    elif aggregation == 'mean':
        final_map = stacked.mean(dim=0)  # [H, W]
    else:
        final_map = stacked.mean(dim=0)
    
    # 计算整体置信度（前景区域的平均相似度）
    confidence = final_map.mean().item()
    
    return final_map.cpu(), confidence


# ============================================================
# 3. 基于空间特征的分类和分割
# ============================================================

def classify_mask_spatial(query_features, query_mask, memory_bank, 
                          method='max', device='cpu'):
    """
    使用空间特征对单个 mask 进行分类
    
    Args:
        query_features: [C, H, W] 查询图像的特征图（与 mask 相同分辨率）
        query_mask: [H, W] 查询 mask（已 resize 到特征图大小）
        memory_bank: 空间特征记忆库
        method: 像素级匹配方法
        device: 计算设备
    
    Returns:
        best_class: str 最佳匹配类别
        best_score: float 最佳匹配得分
        all_scores: dict 所有类别的得分
    """
    query_features = query_features.to(device)
    query_mask_tensor = torch.from_numpy(query_mask).to(device) if isinstance(query_mask, np.ndarray) else query_mask.to(device)
    
    all_scores = {}
    
    for class_name in memory_bank.keys():
        # 计算与该类别的相似度图
        sim_map, _ = compute_class_similarity(
            query_features, memory_bank, class_name, method=method, device=device
        )
        sim_map = sim_map.to(device)
        
        # 只在 mask 区域内计算平均相似度
        mask_region = query_mask_tensor > 0.5
        if mask_region.sum() > 0:
            score = sim_map[mask_region].mean().item()
        else:
            score = 0.0
        
        all_scores[class_name] = score
    
    # 找到最佳匹配
    best_class = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_class]
    
    return best_class, best_score, all_scores


def extract_query_features(img_pil, dino_model, dino_transform, device='cpu'):
    """
    提取查询图像的空间特征
    
    Returns:
        feature_map: [C, H, W] 特征图
        (H, W): 特征图尺寸
    """
    img_tensor = dino_transform(img_pil)[None].to(device)
    
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        features = dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
        h = w = int(features.shape[1]**0.5)
        feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        feature_map = feature_map.float().squeeze(0)  # [C, H, W]
    
    return feature_map, (h, w)


# ============================================================
# 4. 完整的预测流程（替换原有的 predict_single_image）
# ============================================================

def predict_single_image_spatial(img_path, spatial_memory_bank, sam2_mask_generator,
                                  dino_model, dino_transform,
                                  min_area=500, max_area=None,
                                  iou_threshold=0.5, conf_threshold=0.5,
                                  matching_method='max',
                                  merge_overlapping=True, device='cpu'):
    """
    使用空间特征 Memory Bank 预测单张图像
    
    Args:
        img_path: 图像路径
        spatial_memory_bank: 空间特征记忆库（由 build_spatial_memory_bank 构建）
        sam2_mask_generator: SAM2 mask 生成器
        dino_model: DINO 模型
        dino_transform: 图像预处理
        min_area: 最小面积阈值
        max_area: 最大面积阈值
        iou_threshold: NMS IoU 阈值
        conf_threshold: 置信度阈值
        matching_method: 像素级匹配方法 ('max', 'mean', 'topk', 'weighted')
        merge_overlapping: 是否合并重叠 mask
        device: 计算设备
    
    Returns:
        img_pil: PIL 图像
        masks: SAM2 生成的所有 masks
        final_segmented_instances: [(mask, score, class_name, bbox), ...]
    """
    import torchvision
    
    # 1. 加载图像
    img_pil = Image.open(img_path).convert('RGB')
    
    # 2. 提取查询图像的空间特征
    print("📊 提取查询图像特征...")
    query_features, (feat_h, feat_w) = extract_query_features(
        img_pil, dino_model, dino_transform, device=device
    )
    
    # 3. 使用 SAM2 生成候选 mask
    print("🎯 生成候选 masks...")
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks = sam2_mask_generator.generate(np.array(img_pil))
    candidate_masks = [m['segmentation'] for m in masks]
    print(f"   生成了 {len(candidate_masks)} 个候选 mask")
    
    # 4. 过滤并获取 bounding box
    candidate_masks_with_boxes = []
    for mask in candidate_masks:
        bbox = mask_to_bbox_xyxy(mask, min_area, max_area)
        if bbox is not None:
            candidate_masks_with_boxes.append((mask, bbox))
    
    print(f"   过滤后剩余 {len(candidate_masks_with_boxes)} 个候选 mask")
    
    if len(candidate_masks_with_boxes) == 0:
        print(f"⚠️  没有检测到符合条件的候选 mask")
        return img_pil, masks, []
    
    # 5. 对每个候选 mask 进行分类
    print("🔍 使用像素级匹配进行分类...")
    predicted_masks_info = []
    
    for mask_np, bbox in tqdm(candidate_masks_with_boxes, desc="Classifying"):
        # 将 mask resize 到特征图大小
        resized_mask = cv2.resize(
            mask_np.astype(np.float32), 
            (feat_w, feat_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # 使用空间特征进行分类
        best_class, best_score, all_scores = classify_mask_spatial(
            query_features, resized_mask, spatial_memory_bank,
            method=matching_method, device=device
        )
        
        if best_score >= conf_threshold:
            predicted_masks_info.append((mask_np, best_score, best_class, bbox, all_scores))
    
    print(f"   通过置信度阈值的 mask: {len(predicted_masks_info)} 个")
    
    if len(predicted_masks_info) == 0:
        print(f"⚠️  没有置信度 >= {conf_threshold} 的检测结果")
        return img_pil, masks, []
    
    # 6. 按类别分组并进行 NMS
    masks_by_class = {}
    for mask_np, score, class_name, bbox, all_scores in predicted_masks_info:
        if class_name not in masks_by_class:
            masks_by_class[class_name] = []
        masks_by_class[class_name].append((mask_np, score, bbox))
    
    final_segmented_instances = []
    
    for cls_name, values in masks_by_class.items():
        current_masks = torch.stack([torch.tensor(v[0]) for v in values], dim=0)
        current_scores = torch.tensor([v[1] for v in values])
        current_boxes = torch.tensor([v[2] for v in values]).float()
        
        # NMS
        keep_indices = torchvision.ops.nms(current_boxes, current_scores, iou_threshold=iou_threshold)
        
        final_masks = current_masks[keep_indices]
        final_scores = current_scores[keep_indices]
        final_boxes = current_boxes[keep_indices]
        
        # 合并重叠 mask
        if merge_overlapping and len(final_masks) > 1:
            final_masks, final_scores, final_boxes = merge_overlapping_masks(
                final_masks, final_scores, final_boxes, iou_threshold=0.3
            )
        
        for mask, score, box in zip(final_masks, final_scores, final_boxes):
            final_segmented_instances.append([mask.numpy(), score.item(), cls_name, box.numpy()])
    
    print(f"✅ 最终检测到 {len(final_segmented_instances)} 个实例")
    
    return img_pil, masks, final_segmented_instances


# ============================================================
# 辅助函数
# ============================================================

def mask_to_bbox_xyxy(mask_np, min_area=None, max_area=None):
    """将 mask 转换为 bounding box"""
    if mask_np.sum() == 0:
        return None
    
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    area = w * h
    
    if min_area is not None and area < min_area:
        return None
    if max_area is not None and area > max_area:
        return None
    
    return [x, y, x + w, y + h]


def merge_overlapping_masks(masks, scores, boxes, iou_threshold=0.3):
    """合并高度重叠的 mask"""
    if len(masks) <= 1:
        return masks, scores, boxes
    
    def compute_mask_iou(mask1, mask2):
        intersection = (mask1 * mask2).sum()
        union = ((mask1 + mask2) > 0).sum()
        return intersection / (union + 1e-8)
    
    merged_indices = []
    used = set()
    
    for i in range(len(masks)):
        if i in used:
            continue
        
        group = [i]
        for j in range(i + 1, len(masks)):
            if j in used:
                continue
            
            iou = compute_mask_iou(masks[i].float(), masks[j].float())
            if iou > iou_threshold:
                group.append(j)
                used.add(j)
        
        merged_indices.append(group)
        used.add(i)
    
    new_masks, new_scores, new_boxes = [], [], []
    
    for group in merged_indices:
        if len(group) == 1:
            idx = group[0]
            new_masks.append(masks[idx])
            new_scores.append(scores[idx])
            new_boxes.append(boxes[idx])
        else:
            merged_mask = torch.zeros_like(masks[0]).float()
            for idx in group:
                merged_mask = torch.maximum(merged_mask, masks[idx].float())
            
            merged_score = torch.mean(torch.stack([scores[idx] for idx in group]))
            
            group_boxes = torch.stack([boxes[idx] for idx in group])
            merged_box = torch.tensor([
                group_boxes[:, 0].min(),
                group_boxes[:, 1].min(),
                group_boxes[:, 2].max(),
                group_boxes[:, 3].max()
            ])
            
            new_masks.append(merged_mask > 0.5)
            new_scores.append(merged_score)
            new_boxes.append(merged_box)
    
    return (torch.stack(new_masks) if new_masks else masks,
            torch.stack(new_scores) if new_scores else scores,
            torch.stack(new_boxes) if new_boxes else boxes)


# ============================================================
# 5. 可视化工具
# ============================================================

def visualize_similarity_maps(img_pil, query_features, spatial_memory_bank, 
                               method='max', device='cpu'):
    """
    可视化查询图像与每个类别的相似度图
    """
    class_names = list(spatial_memory_bank.keys())
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))
    
    # 原图
    axes[0, 0].imshow(img_pil)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # 每个类别的相似度图
    for i, class_name in enumerate(class_names):
        sim_map, confidence = compute_class_similarity(
            query_features, spatial_memory_bank, class_name,
            method=method, device=device
        )
        
        # 上采样到原图大小
        sim_map_resized = F.interpolate(
            sim_map.unsqueeze(0).unsqueeze(0),
            size=img_pil.size[::-1],
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # 相似度热力图
        axes[0, i + 1].imshow(img_pil)
        im = axes[0, i + 1].imshow(sim_map_resized, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        axes[0, i + 1].set_title(f'{class_name}\n(conf: {confidence:.3f})', fontweight='bold')
        axes[0, i + 1].axis('off')
        
        # 纯相似度图
        axes[1, i + 1].imshow(sim_map_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, i + 1].set_title(f'Similarity Map', fontsize=10)
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='Similarity')
    plt.show()


def visualize_spatial_memory_bank(spatial_memory_bank, figsize=(15, 10)):
    """
    可视化空间特征 Memory Bank 的内容
    """
    class_names = list(spatial_memory_bank.keys())
    
    # 计算总行数
    max_features = max(len(v['features']) for v in spatial_memory_bank.values())
    
    fig, axes = plt.subplots(len(class_names), max_features + 1, 
                              figsize=(3 * (max_features + 1), 3 * len(class_names)))
    
    if len(class_names) == 1:
        axes = axes.reshape(1, -1)
    
    for row, class_name in enumerate(class_names):
        data = spatial_memory_bank[class_name]
        features = data['features']
        masks = data['masks']
        
        # 类别标题
        axes[row, 0].text(0.5, 0.5, f'{class_name}\n({len(features)} samples)', 
                         ha='center', va='center', fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # 每个特征的 PCA 可视化
        for col, (feat, mask) in enumerate(zip(features, masks)):
            if col >= max_features:
                break
            
            # PCA 降维可视化
            C, H, W = feat.shape
            feat_flat = feat.reshape(C, -1).T.numpy()  # [H*W, C]
            
            pca = PCA(n_components=3)
            feat_pca = pca.fit_transform(feat_flat)
            feat_pca = feat_pca.reshape(H, W, 3)
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
            
            # 叠加 mask 轮廓
            mask_np = mask.numpy()
            
            axes[row, col + 1].imshow(feat_pca)
            axes[row, col + 1].contour(mask_np, colors='red', linewidths=2)
            axes[row, col + 1].set_title(f'Sample {col + 1}', fontsize=10)
            axes[row, col + 1].axis('off')
        
        # 隐藏多余的子图
        for col in range(len(features) + 1, max_features + 1):
            axes[row, col].axis('off')
    
    plt.suptitle('Spatial Memory Bank Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. 使用示例
# ============================================================

"""
使用示例：

# 1. 构建空间特征 Memory Bank
from spatial_memory_bank import build_spatial_memory_bank, predict_single_image_spatial

spatial_memory_bank = build_spatial_memory_bank(
    reference_data,
    dino_model,
    dino_transform,
    device='cuda',
    max_features_per_class=5  # 每个类别最多保留5个特征
)

# 2. 可视化 Memory Bank
from spatial_memory_bank import visualize_spatial_memory_bank
visualize_spatial_memory_bank(spatial_memory_bank)

# 3. 预测
img_pil, masks, instances = predict_single_image_spatial(
    test_img_path,
    spatial_memory_bank,
    sam2_mask_generator,
    dino_model,
    dino_transform,
    min_area=100,
    conf_threshold=0.5,
    matching_method='max',  # 推荐使用 'max'
    device='cuda'
)

# 4. 可视化相似度图
from spatial_memory_bank import visualize_similarity_maps, extract_query_features

query_features, _ = extract_query_features(img_pil, dino_model, dino_transform, device='cuda')
visualize_similarity_maps(img_pil, query_features, spatial_memory_bank, device='cuda')
"""

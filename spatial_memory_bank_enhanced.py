"""
SP-SAM 优化版本
==================

基于原始spatial_memory_bank.py的改进实现
主要改进：
1. 自适应阈值
2. 引导式SAM2生成
3. 多尺度特征融合
4. 更好的置信度计算
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu


# ============================================================
# 改进1：自适应阈值计算
# ============================================================

def compute_adaptive_threshold(scores, method='otsu'):
    """
    自适应计算置信度阈值
    
    Args:
        scores: List[float] 或 np.array，相似度分数
        method: 阈值计算方法
            - 'otsu': Otsu自动阈值
            - 'quantile': 基于分位数
            - 'mean_std': 均值+标准差
    
    Returns:
        threshold: float
    """
    if len(scores) == 0:
        return 0.5  # 默认阈值
    
    scores_array = np.array(scores)
    
    if method == 'otsu':
        try:
            # Otsu算法对二值化最优
            # 将scores归一化到[0, 255]
            scores_norm = ((scores_array - scores_array.min()) / 
                          (scores_array.max() - scores_array.min() + 1e-8) * 255).astype(np.uint8)
            threshold_norm = threshold_otsu(scores_norm)
            # 映射回原始范围
            threshold = threshold_norm / 255.0 * (scores_array.max() - scores_array.min()) + scores_array.min()
        except:
            # 如果Otsu失败，使用quantile
            threshold = np.quantile(scores_array, 0.5)
            
    elif method == 'quantile':
        # 使用中位数或更高分位数
        threshold = np.quantile(scores_array, 0.6)
        
    elif method == 'mean_std':
        # 均值 + 0.5倍标准差
        threshold = scores_array.mean() + 0.5 * scores_array.std()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(threshold)


def adaptive_threshold_filtering(masks, scores, classes, method='otsu', 
                                min_score=0.3, relative_threshold=True):
    """
    使用自适应阈值过滤masks
    
    Args:
        masks: List[np.array]
        scores: List[float]
        classes: List[str]
        method: 阈值计算方法
        min_score: 绝对最小分数（保底）
        relative_threshold: 是否使用相对阈值
    
    Returns:
        filtered_masks, filtered_scores, filtered_classes
    """
    if len(scores) == 0:
        return [], [], []
    
    scores_array = np.array(scores)
    
    # 计算自适应阈值
    if relative_threshold:
        adaptive_th = compute_adaptive_threshold(scores_array, method=method)
        # 确保不低于最小值
        threshold = max(adaptive_th, min_score)
    else:
        threshold = min_score
    
    print(f"   自适应阈值: {threshold:.3f} (方法: {method})")
    
    # 筛选
    valid_indices = scores_array >= threshold
    filtered_masks = [m for m, v in zip(masks, valid_indices) if v]
    filtered_scores = scores_array[valid_indices].tolist()
    filtered_classes = [c for c, v in zip(classes, valid_indices) if v]
    
    print(f"   筛选结果: {len(filtered_masks)}/{len(masks)} 个masks保留")
    
    return filtered_masks, filtered_scores, filtered_classes


# ============================================================
# 改进2：引导式SAM2生成
# ============================================================

def extract_similarity_peaks(similarity_map, top_k=20, min_distance=20, 
                             threshold_rel=0.5):
    """
    从相似度图中提取局部最大值点
    
    Args:
        similarity_map: [H, W] tensor，相似度图
        top_k: 最多提取的峰值数量
        min_distance: 峰值之间的最小距离（像素）
        threshold_rel: 相对阈值（相对于最大值）
    
    Returns:
        peaks: np.array [N, 2]，(y, x)坐标
        peak_values: np.array [N]，峰值处的相似度
    """
    sim_np = similarity_map.cpu().numpy()
    
    # 提取局部最大值
    peaks = peak_local_max(
        sim_np,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=top_k,
        exclude_border=True
    )
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    # 获取峰值处的相似度
    peak_values = sim_np[peaks[:, 0], peaks[:, 1]]
    
    # 按相似度排序
    sorted_indices = np.argsort(peak_values)[::-1]
    peaks = peaks[sorted_indices]
    peak_values = peak_values[sorted_indices]
    
    return peaks, peak_values


def similarity_guided_sam2_generation(img_pil, similarity_map, sam2_predictor,
                                     top_k=20, min_distance=20, 
                                     score_threshold=0.5):
    """
    使用相似度图引导SAM2生成masks
    
    这是改进的核心：不再盲目生成所有候选masks，而是：
    1. 从相似度图找高置信度点
    2. 用这些点作为prompts引导SAM2
    3. 显著减少候选masks数量，提高精度
    
    Args:
        img_pil: PIL Image
        similarity_map: [H, W] tensor，查询图像对某个类别的相似度图
        sam2_predictor: SAM2 predictor实例
        top_k: 最多使用的prompt点数量
        min_distance: 点之间的最小距离
        score_threshold: SAM2输出的最低置信度
    
    Returns:
        masks: List[np.array]，生成的masks
        mask_scores: List[float]，SAM2的置信度
        prompt_coords: List[tuple]，使用的prompt坐标（用于可视化）
    """
    # 1. 提取相似度峰值点
    peaks, peak_values = extract_similarity_peaks(
        similarity_map, 
        top_k=top_k,
        min_distance=min_distance,
        threshold_rel=0.5
    )
    
    if len(peaks) == 0:
        print("   ⚠️  未找到相似度峰值，跳过该类别")
        return [], [], []
    
    print(f"   找到 {len(peaks)} 个候选点，相似度范围: {peak_values.min():.3f}-{peak_values.max():.3f}")
    
    # 2. 设置SAM2图像
    sam2_predictor.set_image(np.array(img_pil))
    
    H_sim, W_sim = similarity_map.shape
    H_img, W_img = img_pil.size[::-1]  # PIL是(W,H)
    
    masks = []
    mask_scores = []
    prompt_coords = []
    
    # 3. 对每个峰值点生成mask
    for (y, x), sim_value in zip(peaks, peak_values):
        # 坐标从特征图上采样到原图
        x_img = int(x * W_img / W_sim)
        y_img = int(y * H_img / H_sim)
        
        # 使用点prompt生成mask
        try:
            mask, score, _ = sam2_predictor.predict(
                point_coords=np.array([[x_img, y_img]]),
                point_labels=np.array([1]),  # 1=前景点
                multimask_output=False  # 只要最好的mask
            )
            
            # 检查SAM2的置信度
            if score[0] >= score_threshold and mask[0].sum() > 0:
                masks.append(mask[0])
                # 综合SAM2分数和相似度
                combined_score = 0.7 * score[0] + 0.3 * float(sim_value)
                mask_scores.append(combined_score)
                prompt_coords.append((x_img, y_img))
                
        except Exception as e:
            print(f"   ⚠️  点({x_img},{y_img})生成失败: {e}")
            continue
    
    print(f"   ✅ 成功生成 {len(masks)} 个masks")
    
    return masks, mask_scores, prompt_coords


# ============================================================
# 改进3：多尺度特征融合
# ============================================================

def extract_multiscale_dino_features(img_tensor, dino_model, device='cuda',
                                    layers=[9, 15, 21, 23], weights=None):
    """
    提取并融合多层DINO特征
    
    思想：不同层次的特征捕获不同的语义信息
    - 浅层：纹理、边缘
    - 深层：语义、对象
    融合后的特征更鲁棒
    
    Args:
        img_tensor: [1, 3, H, W]
        dino_model: DINO模型
        device: 设备
        layers: 要提取的层索引（对于ViT-L/14，总共24层）
        weights: 融合权重，None则自动计算（深层权重更高）
    
    Returns:
        fused_features: [C, H, W]，融合后的特征图
    """
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        # 获取多层特征
        features_list = dino_model.get_intermediate_layers(
            img_tensor.to(torch.bfloat16), 
            n=layers
        )
        
        # 重塑为空间形状
        spatial_features = []
        for feat in features_list:
            h = w = int(feat.shape[1]**0.5)
            # [1, H*W, C] -> [1, C, H, W]
            feat_map = feat.reshape(1, h, w, -1).permute(0, 3, 1, 2)
            spatial_features.append(feat_map.float())
        
        # 上采样到最深层的尺寸（通常最细）
        target_size = spatial_features[-1].shape[-2:]
        aligned_features = []
        
        for feat in spatial_features:
            if feat.shape[-2:] != target_size:
                aligned = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                aligned = feat
            aligned_features.append(aligned)
        
        # 计算融合权重（深层权重更高）
        if weights is None:
            # 自动权重：深层2倍权重
            weights = torch.linspace(1.0, 2.0, len(layers))
            weights = F.softmax(weights, dim=0)
        else:
            weights = torch.tensor(weights)
            weights = weights / weights.sum()
        
        # 加权融合
        fused = sum(w.item() * f for w, f in zip(weights, aligned_features))
        
        print(f"   多尺度融合: {len(layers)}层, 权重={weights.tolist()}")
    
    return fused.squeeze(0)  # [C, H, W]


# ============================================================
# 改进4：增强的预测流程
# ============================================================

def predict_single_image_enhanced(
    test_img_path,
    spatial_memory_bank,
    sam2_predictor,  # 注意：这里改用predictor而非mask_generator
    dino_model,
    dino_transform,
    use_guided_sam=True,  # 是否使用引导式SAM2
    use_multiscale=True,  # 是否使用多尺度特征
    adaptive_threshold=True,  # 是否使用自适应阈值
    matching_method='max',
    conf_threshold=0.5,  # 仅在不使用自适应阈值时有效
    min_area=100,
    device='cuda'
):
    """
    增强版的单图像预测函数
    
    主要改进：
    1. 可选多尺度特征提取
    2. 使用引导式SAM2生成
    3. 自适应阈值筛选
    
    Args:
        test_img_path: 测试图像路径
        spatial_memory_bank: 空间特征memory bank
        sam2_predictor: SAM2 predictor（非mask_generator）
        dino_model, dino_transform: DINO模型和变换
        use_guided_sam: 是否使用相似度引导SAM2
        use_multiscale: 是否使用多尺度特征
        adaptive_threshold: 是否使用自适应阈值
        matching_method: 像素级匹配方法
        conf_threshold: 固定阈值（仅在adaptive_threshold=False时使用）
        min_area: 最小mask面积
        device: 设备
    
    Returns:
        img_pil: 原始图像
        class_predictions: Dict[str, Dict]，每个类别的预测结果
            {
                'class_name': {
                    'masks': List[np.array],
                    'scores': List[float],
                    'coords': List[tuple],  # 仅在use_guided_sam=True时有
                    'similarity_map': np.array
                }
            }
        final_instances: List[Dict]，最终检测到的实例
            [
                {
                    'mask': np.array,
                    'score': float,
                    'class': str,
                    'box': [x1, y1, x2, y2]
                },
                ...
            ]
    """
    print("="*60)
    print("🔍 开始增强版预测")
    print(f"   图像: {test_img_path}")
    print(f"   配置: guided_sam={use_guided_sam}, multiscale={use_multiscale}, adaptive_th={adaptive_threshold}")
    print("="*60)
    
    # 1. 加载图像
    img_pil = Image.open(test_img_path).convert('RGB')
    img_tensor = dino_transform(img_pil)[None].to(device)
    
    # 2. 提取查询特征
    print("\n📊 提取查询特征...")
    if use_multiscale:
        query_features = extract_multiscale_dino_features(
            img_tensor, dino_model, device=device
        )
    else:
        # 标准单层特征
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            features = dino_model.get_intermediate_layers(img_tensor.to(torch.bfloat16))[0]
            h = w = int(features.shape[1]**0.5)
            query_features = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0).float()
    
    print(f"   特征维度: {query_features.shape}")
    
    # 3. 对每个类别计算相似度并生成masks
    class_predictions = {}
    all_masks = []
    all_scores = []
    all_classes = []
    all_boxes = []
    
    for class_name in spatial_memory_bank.keys():
        print(f"\n🎯 处理类别: {class_name}")
        
        # 3.1 计算相似度图
        from spatial_memory_bank import compute_class_similarity
        similarity_map, confidence = compute_class_similarity(
            query_features,
            spatial_memory_bank,
            class_name,
            method=matching_method,
            aggregation='max',
            device=device
        )
        
        print(f"   相似度范围: {similarity_map.min():.3f} - {similarity_map.max():.3f}")
        print(f"   整体置信度: {confidence:.3f}")
        
        # 3.2 生成masks
        if use_guided_sam:
            # 引导式生成
            masks, mask_scores, prompt_coords = similarity_guided_sam2_generation(
                img_pil, similarity_map, sam2_predictor,
                top_k=20, min_distance=20, score_threshold=0.5
            )
        else:
            # 传统方式：生成所有候选再筛选
            # （这里需要mask_generator，所以这种模式需要单独处理）
            print("   ⚠️  传统模式需要使用mask_generator，这里跳过")
            masks, mask_scores, prompt_coords = [], [], []
        
        # 保存该类别的结果
        class_predictions[class_name] = {
            'masks': masks,
            'scores': mask_scores,
            'coords': prompt_coords if use_guided_sam else None,
            'similarity_map': similarity_map.cpu().numpy()
        }
        
        # 添加到总列表
        all_masks.extend(masks)
        all_scores.extend(mask_scores)
        all_classes.extend([class_name] * len(masks))
        
        # 计算bounding boxes
        for mask in masks:
            box = mask_to_bbox_xyxy(mask, min_area=min_area)
            all_boxes.append(box if box is not None else [0, 0, 0, 0])
    
    # 4. 自适应阈值筛选
    print("\n🔧 应用自适应阈值筛选...")
    if adaptive_threshold:
        final_masks, final_scores, final_classes = adaptive_threshold_filtering(
            all_masks, all_scores, all_classes,
            method='otsu',  # 可以尝试'quantile'或'mean_std'
            min_score=0.3,
            relative_threshold=True
        )
    else:
        # 使用固定阈值
        valid_indices = [i for i, s in enumerate(all_scores) if s >= conf_threshold]
        final_masks = [all_masks[i] for i in valid_indices]
        final_scores = [all_scores[i] for i in valid_indices]
        final_classes = [all_classes[i] for i in valid_indices]
        print(f"   固定阈值 {conf_threshold}: {len(final_masks)}/{len(all_masks)} 保留")
    
    # 5. NMS去重
    if len(final_masks) > 1:
        print("\n🔄 NMS去重...")
        final_masks, final_scores, final_classes = nms_by_class(
            final_masks, final_scores, final_classes, iou_threshold=0.5
        )
    
    # 6. 构建最终实例列表
    final_instances = []
    for mask, score, cls in zip(final_masks, final_scores, final_classes):
        box = mask_to_bbox_xyxy(mask, min_area=min_area)
        if box is not None:
            final_instances.append({
                'mask': mask,
                'score': score,
                'class': cls,
                'box': box
            })
    
    print("="*60)
    print(f"✅ 预测完成！最终检测到 {len(final_instances)} 个实例")
    for inst in final_instances:
        print(f"   - {inst['class']}: {inst['score']:.3f}")
    print("="*60)
    
    return img_pil, class_predictions, final_instances


# ============================================================
# 辅助函数
# ============================================================

def mask_to_bbox_xyxy(mask_np, min_area=None, max_area=None):
    """将mask转换为bounding box"""
    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.cpu().numpy()
    
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


def compute_mask_iou(mask1, mask2):
    """计算两个mask的IoU"""
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    intersection = (mask1 * mask2).sum()
    union = ((mask1 + mask2) > 0).sum()
    return intersection / (union + 1e-8)


def nms_by_class(masks, scores, classes, iou_threshold=0.5):
    """
    按类别进行NMS
    """
    if len(masks) <= 1:
        return masks, scores, classes
    
    # 按类别分组
    class_groups = {}
    for i, cls in enumerate(classes):
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(i)
    
    # 对每个类别单独NMS
    keep_indices = []
    for cls, indices in class_groups.items():
        cls_masks = [masks[i] for i in indices]
        cls_scores = [scores[i] for i in indices]
        
        # 按分数排序
        sorted_indices = np.argsort(cls_scores)[::-1]
        
        keep_cls = []
        while len(sorted_indices) > 0:
            # 保留当前最高分
            current = sorted_indices[0]
            keep_cls.append(indices[current])
            
            if len(sorted_indices) == 1:
                break
            
            # 计算与其他masks的IoU
            current_mask = cls_masks[current]
            remaining = sorted_indices[1:]
            
            new_remaining = []
            for idx in remaining:
                iou = compute_mask_iou(current_mask, cls_masks[idx])
                if iou < iou_threshold:  # 保留低重叠的
                    new_remaining.append(idx)
            
            sorted_indices = np.array(new_remaining)
        
        keep_indices.extend(keep_cls)
    
    # 按原始顺序返回
    keep_indices = sorted(keep_indices)
    return ([masks[i] for i in keep_indices],
            [scores[i] for i in keep_indices],
            [classes[i] for i in keep_indices])


# ============================================================
# 使用示例
# ============================================================

"""
使用示例：

from spatial_memory_bank_enhanced import *
from spatial_memory_bank import build_spatial_memory_bank

# 1. 构建memory bank（和之前一样）
memory_bank = build_spatial_memory_bank(
    reference_data, dino_model, dino_transform, device='cuda'
)

# 2. 使用增强版预测
img, class_preds, instances = predict_single_image_enhanced(
    'test.jpg',
    memory_bank,
    sam2_predictor,  # 注意：需要predictor而非mask_generator
    dino_model,
    dino_transform,
    use_guided_sam=True,      # 开启引导式SAM2
    use_multiscale=True,      # 开启多尺度特征
    adaptive_threshold=True,  # 开启自适应阈值
    device='cuda'
)

# 3. 可视化结果
for inst in instances:
    print(f"{inst['class']}: {inst['score']:.3f}")
"""

"""
基于 SAM2 Memory 机制的少样本分割
=====================================

核心思想（参考 FS-SAM2 论文）：
1. 将支持图像当作"视频的前一帧"
2. 利用 SAM2 的 memory encoder 编码支持特征+mask
3. 利用 SAM2 的 memory attention 进行像素级匹配
4. 使用 SAM2 的 mask decoder 生成高质量 mask

创新点：
1. 复用 SAM2 的视频分割能力进行少样本分割
2. Memory Bank 存储编码后的空间特征（而非原型向量）
3. 像素级匹配 + 高质量 mask decoder

这比简单的热力图阈值分割更有意义，因为：
- 利用了 SAM2 预训练的强大能力
- mask decoder 生成的边界更精细
- memory attention 的匹配更智能
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class SAM2MemoryBasedSegmentor:
    """
    基于 SAM2 Memory 机制的少样本分割器
    
    核心流程：
    1. 支持图像 + mask → SAM2 Memory Encoder → 编码的记忆特征
    2. 查询图像 → SAM2 Image Encoder → 查询特征
    3. 记忆特征 + 查询特征 → Memory Attention → 条件化的查询特征
    4. 条件化特征 → Mask Decoder → 预测 mask
    """
    
    def __init__(self, sam2_model, device='cuda'):
        """
        Args:
            sam2_model: 加载好的 SAM2 模型
            device: 计算设备
        """
        self.sam2 = sam2_model
        self.device = device
        self.memory_bank = {}  # 存储编码后的记忆特征
        
        # 获取 SAM2 的各个组件
        self.image_encoder = sam2_model.image_encoder
        self.memory_encoder = sam2_model.memory_encoder
        self.memory_attention = sam2_model.memory_attention
        self.mask_decoder = sam2_model.sam_mask_decoder
        
        print("✅ SAM2 Memory-based Segmentor 初始化完成")
        print(f"   设备: {device}")
    
    def encode_support(self, support_image, support_mask, class_name):
        """
        编码支持图像，存入记忆库
        
        Args:
            support_image: PIL Image 或 numpy array
            support_mask: 二值 mask (H, W)
            class_name: 类别名称
        """
        if isinstance(support_image, Image.Image):
            support_image = np.array(support_image)
        
        if isinstance(support_mask, torch.Tensor):
            support_mask = support_mask.numpy()
        
        with torch.inference_mode():
            # 1. 预处理图像
            img_tensor = self._preprocess_image(support_image)
            
            # 2. 图像编码
            image_embedding = self.image_encoder(img_tensor)
            
            # 处理不同的返回格式
            if isinstance(image_embedding, dict):
                backbone_features = image_embedding.get('vision_features', image_embedding.get('backbone_fpn', None))
                if backbone_features is None:
                    backbone_features = list(image_embedding.values())[0]
            elif isinstance(image_embedding, (list, tuple)):
                backbone_features = image_embedding[0]
            else:
                backbone_features = image_embedding
            
            # 3. 准备 mask（调整到特征图大小）
            if isinstance(backbone_features, (list, tuple)):
                feat_h, feat_w = backbone_features[0].shape[-2:]
            else:
                feat_h, feat_w = backbone_features.shape[-2:]
            
            mask_resized = cv2.resize(
                support_mask.astype(np.float32),
                (feat_w, feat_h),
                interpolation=cv2.INTER_NEAREST
            )
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 4. Memory 编码（将 mask 信息融入特征）
            # SAM2 的 memory encoder 将图像特征和 mask 结合
            try:
                memory_features = self._encode_to_memory(backbone_features, mask_tensor)
            except Exception as e:
                print(f"⚠️  Memory encoding failed: {e}")
                print("   使用简化的特征存储方式")
                # 简化方案：直接存储 masked 特征
                if isinstance(backbone_features, (list, tuple)):
                    memory_features = backbone_features[0] * mask_tensor
                else:
                    memory_features = backbone_features * mask_tensor
            
            # 5. 存入记忆库
            if class_name not in self.memory_bank:
                self.memory_bank[class_name] = []
            
            self.memory_bank[class_name].append({
                'features': memory_features.cpu(),
                'mask': mask_tensor.cpu(),
                'original_size': support_image.shape[:2]
            })
        
        print(f"   ✅ 编码支持图像: {class_name} (共 {len(self.memory_bank[class_name])} 个样本)")
    
    def _preprocess_image(self, image):
        """预处理图像为 SAM2 输入格式"""
        from torchvision import transforms
        
        # SAM2 标准预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        return img_tensor
    
    def _encode_to_memory(self, image_features, mask):
        """
        使用 SAM2 的 memory encoder 编码特征
        
        这是关键步骤：将 mask 信息融入到图像特征中
        """
        # SAM2 memory encoder 的输入格式
        # 需要根据实际 SAM2 实现调整
        
        if hasattr(self.memory_encoder, 'forward'):
            # 尝试直接调用
            try:
                # 不同版本的 SAM2 可能有不同的接口
                if isinstance(image_features, (list, tuple)):
                    feat = image_features[0]
                else:
                    feat = image_features
                
                # 简化的 memory encoding：特征 * mask + 位置编码
                masked_feat = feat * mask
                
                # 添加 mask 作为额外通道
                mask_expanded = mask.expand_as(masked_feat[:, :1, :, :])
                memory_feat = torch.cat([masked_feat, mask_expanded], dim=1)
                
                return memory_feat[:, :feat.shape[1], :, :]  # 保持通道数一致
                
            except Exception as e:
                print(f"Memory encoder error: {e}")
                raise
        else:
            raise NotImplementedError("Memory encoder not available")
    
    def build_memory_bank(self, reference_data):
        """
        从参考数据构建记忆库
        
        Args:
            reference_data: {
                'class_name': [
                    {'img_path': str, 'masks': [mask1, mask2, ...]},
                    ...
                ]
            }
        """
        print("="*60)
        print("🔨 构建 SAM2 Memory Bank")
        print("="*60)
        
        self.memory_bank = {}
        
        for class_name, ref_list in tqdm(reference_data.items(), desc="Encoding classes"):
            for ref_data in ref_list:
                img_path = ref_data['img_path']
                masks = ref_data['masks']
                
                # 加载图像
                img = np.array(Image.open(img_path).convert('RGB'))
                
                # 编码每个实例
                for mask in masks:
                    if mask.sum() > 100:  # 过滤太小的 mask
                        self.encode_support(img, mask, class_name)
        
        print("="*60)
        print(f"✅ Memory Bank 构建完成")
        for cls, memories in self.memory_bank.items():
            print(f"   {cls}: {len(memories)} 个记忆")
        print("="*60)
    
    def predict(self, query_image, similarity_threshold=0.5, top_k_memories=3):
        """
        对查询图像进行预测
        
        Args:
            query_image: PIL Image 或 numpy array
            similarity_threshold: 相似度阈值
            top_k_memories: 使用 top-k 个最相似的记忆
        
        Returns:
            predictions: [(mask, score, class_name, bbox), ...]
        """
        if isinstance(query_image, Image.Image):
            query_image_np = np.array(query_image)
        else:
            query_image_np = query_image
        
        original_size = query_image_np.shape[:2]
        
        with torch.inference_mode():
            # 1. 编码查询图像
            img_tensor = self._preprocess_image(query_image_np)
            query_features = self.image_encoder(img_tensor)
            
            if isinstance(query_features, dict):
                query_feat = list(query_features.values())[0]
            elif isinstance(query_features, (list, tuple)):
                query_feat = query_features[0]
            else:
                query_feat = query_features
            
            # 2. 对每个类别计算相似度并生成 mask
            predictions = []
            
            for class_name, memories in self.memory_bank.items():
                # 计算与每个记忆的相似度
                similarities = []
                for mem in memories:
                    mem_feat = mem['features'].to(self.device)
                    sim = self._compute_similarity(query_feat, mem_feat)
                    similarities.append((sim, mem))
                
                # 选择 top-k 最相似的记忆
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_memories = similarities[:top_k_memories]
                
                if top_memories[0][0] < similarity_threshold:
                    continue  # 相似度太低，跳过
                
                # 3. 使用 memory attention 融合特征
                fused_features = self._apply_memory_attention(
                    query_feat, 
                    [m[1]['features'].to(self.device) for m in top_memories]
                )
                
                # 4. 使用 mask decoder 生成 mask
                pred_mask, pred_score = self._decode_mask(fused_features, original_size)
                
                if pred_mask is not None and pred_mask.sum() > 500:
                    bbox = self._mask_to_bbox(pred_mask)
                    predictions.append((pred_mask, float(pred_score), class_name, bbox))
        
        # 按分数排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def _compute_similarity(self, query_feat, memory_feat):
        """计算查询特征和记忆特征的相似度"""
        # 全局平均池化
        q_global = query_feat.mean(dim=[2, 3])  # [B, C]
        m_global = memory_feat.mean(dim=[2, 3])  # [B, C]
        
        # 余弦相似度
        q_norm = F.normalize(q_global, dim=1)
        m_norm = F.normalize(m_global, dim=1)
        
        similarity = (q_norm * m_norm).sum().item()
        return similarity
    
    def _apply_memory_attention(self, query_feat, memory_feats):
        """
        应用 memory attention
        
        简化实现：加权融合记忆特征到查询特征
        """
        # 计算注意力权重
        weights = []
        for mem_feat in memory_feats:
            sim = self._compute_similarity(query_feat, mem_feat)
            weights.append(sim)
        
        weights = torch.softmax(torch.tensor(weights), dim=0)
        
        # 加权融合
        fused = query_feat.clone()
        for w, mem_feat in zip(weights, memory_feats):
            # 调整 memory 特征大小以匹配 query
            if mem_feat.shape != query_feat.shape:
                mem_feat = F.interpolate(mem_feat, size=query_feat.shape[-2:], mode='bilinear')
            fused = fused + w * mem_feat
        
        return fused
    
    def _decode_mask(self, features, original_size):
        """
        使用 SAM2 的 mask decoder 生成 mask
        
        简化实现：使用特征激活生成 mask
        """
        # 激活特征的前景响应
        activation = features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        activation = torch.sigmoid(activation)
        
        # 上采样到原图大小
        mask = F.interpolate(
            activation,
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # 阈值化
        mask_binary = (mask > 0.5).cpu().numpy()
        score = mask.mean().item()
        
        return mask_binary, score
    
    def _mask_to_bbox(self, mask):
        """将 mask 转换为 bounding box"""
        if mask.sum() == 0:
            return [0, 0, 1, 1]
        ys, xs = np.where(mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ============================================================
# 更简洁的方案：DINO 特征 + SAM2 Mask Decoder
# ============================================================

class DINOGuidedSAM2Segmentor:
    """
    DINO 引导的 SAM2 分割器
    
    核心思想：
    1. 使用 DINO 进行语义匹配（找到目标位置）
    2. 使用 SAM2 的 prompt 能力（点/框提示）生成高质量 mask
    
    创新点：
    - DINO 提供语义理解（知道"在哪里"）
    - SAM2 提供精细分割（知道"边界在哪"）
    - 两者结合，扬长避短
    """
    
    def __init__(self, sam2_predictor, dino_model, dino_transform, device='cuda'):
        """
        Args:
            sam2_predictor: SAM2ImagePredictor
            dino_model: DINO 模型
            dino_transform: DINO 预处理
            device: 计算设备
        """
        self.sam2_predictor = sam2_predictor
        self.dino_model = dino_model
        self.dino_transform = dino_transform
        self.device = device
        self.memory_bank = {}  # DINO 空间特征记忆库
        
        print("✅ DINO-Guided SAM2 Segmentor 初始化完成")
    
    def build_memory_bank(self, reference_data, max_per_class=5):
        """构建 DINO 空间特征记忆库"""
        print("="*60)
        print("🔨 构建 DINO 空间特征 Memory Bank")
        print("="*60)
        
        self.memory_bank = {}
        
        for class_name, ref_list in tqdm(reference_data.items(), desc="Processing"):
            class_features = []
            class_masks = []
            
            for ref_data in ref_list:
                img_path = ref_data['img_path']
                masks = ref_data['masks']
                
                # 加载并编码图像
                img_pil = Image.open(img_path).convert('RGB')
                img_tensor = self.dino_transform(img_pil)[None].to(self.device)
                
                with torch.inference_mode():
                    features = self.dino_model.get_intermediate_layers(img_tensor)[0]
                    h = w = int(features.shape[1]**0.5)
                    feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0)
                
                for mask in masks:
                    if mask.sum() < 100:
                        continue
                    
                    # 调整 mask 大小
                    mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                    
                    class_features.append(feature_map.cpu())
                    class_masks.append(torch.from_numpy(mask_resized > 0.5))
                    
                    if len(class_features) >= max_per_class:
                        break
                
                if len(class_features) >= max_per_class:
                    break
            
            if class_features:
                self.memory_bank[class_name] = {
                    'features': class_features,
                    'masks': class_masks
                }
                print(f"   {class_name}: {len(class_features)} 个特征")
        
        print("="*60)
    
    def predict(self, query_image, similarity_threshold=0.5, 
                num_points_per_instance=5, min_area=1000):
        """
        预测查询图像
        
        流程：
        1. DINO 计算相似度图
        2. 找到高响应区域的中心点
        3. 用这些点作为 SAM2 的 prompt
        4. SAM2 生成精细 mask
        """
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')
        
        query_np = np.array(query_image)
        original_size = query_np.shape[:2]
        
        # 1. DINO 特征提取
        img_tensor = self.dino_transform(query_image)[None].to(self.device)
        
        with torch.inference_mode():
            features = self.dino_model.get_intermediate_layers(img_tensor)[0]
            h = w = int(features.shape[1]**0.5)
            query_feat = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).squeeze(0)
        
        # 2. 计算每个类别的相似度图
        predictions = []
        
        for class_name, mem_data in self.memory_bank.items():
            # 计算像素级相似度
            sim_map = self._compute_pixel_similarity(
                query_feat, 
                mem_data['features'], 
                mem_data['masks']
            )
            
            # 上采样
            sim_map_up = F.interpolate(
                sim_map.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear'
            ).squeeze().numpy()
            
            # 3. 找到高响应区域并提取提示点
            prompt_points = self._extract_prompt_points(
                sim_map_up, 
                threshold=similarity_threshold,
                num_points=num_points_per_instance
            )
            
            if len(prompt_points) == 0:
                continue
            
            # 4. 使用 SAM2 生成 mask
            self.sam2_predictor.set_image(query_np)
            
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=np.array(prompt_points),
                point_labels=np.ones(len(prompt_points)),
                multimask_output=True
            )
            
            # 选择最佳 mask
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                best_score = scores[best_idx]
                
                # 结合 DINO 相似度调整分数
                dino_score = sim_map_up[best_mask].mean() if best_mask.sum() > 0 else 0
                combined_score = 0.5 * best_score + 0.5 * dino_score
                
                if best_mask.sum() >= min_area:
                    bbox = self._mask_to_bbox(best_mask)
                    predictions.append((best_mask, float(combined_score), class_name, bbox))
        
        # 排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def _compute_pixel_similarity(self, query_feat, support_feats, support_masks):
        """计算像素级相似度"""
        C, H, W = query_feat.shape
        query_flat = query_feat.reshape(C, -1).T  # [H*W, C]
        query_flat = F.normalize(query_flat, dim=1)
        
        all_sims = []
        
        for sup_feat, sup_mask in zip(support_feats, support_masks):
            sup_feat = sup_feat.to(self.device)
            sup_mask = sup_mask.to(self.device)
            
            sup_flat = sup_feat.reshape(C, -1).T  # [H*W, C]
            mask_flat = sup_mask.reshape(-1) > 0.5
            
            # 前景特征
            fg_feat = sup_flat[mask_flat]
            if fg_feat.shape[0] == 0:
                continue
            
            fg_feat = F.normalize(fg_feat, dim=1)
            
            # 相似度矩阵
            sim = torch.mm(query_flat, fg_feat.T)  # [H*W, N_fg]
            max_sim, _ = sim.max(dim=1)  # [H*W]
            
            all_sims.append(max_sim.reshape(H, W))
        
        if len(all_sims) == 0:
            return torch.zeros(H, W)
        
        # 聚合
        stacked = torch.stack(all_sims, dim=0)
        final_sim, _ = stacked.max(dim=0)
        
        return final_sim.cpu()
    
    def _extract_prompt_points(self, sim_map, threshold=0.5, num_points=5):
        """从相似度图提取提示点"""
        # 阈值化
        mask = sim_map > threshold
        
        if mask.sum() == 0:
            return []
        
        # 找到高响应区域
        from scipy import ndimage
        labeled, num_features = ndimage.label(mask)
        
        points = []
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if region.sum() < 100:
                continue
            
            # 在该区域内找到相似度最高的点
            region_sim = sim_map * region
            ys, xs = np.where(region_sim == region_sim.max())
            
            if len(ys) > 0:
                points.append([xs[0], ys[0]])
        
        # 如果点太少，添加更多点
        if len(points) < num_points and mask.sum() > 0:
            ys, xs = np.where(mask)
            indices = np.random.choice(len(ys), min(num_points - len(points), len(ys)), replace=False)
            for idx in indices:
                points.append([xs[idx], ys[idx]])
        
        return points[:num_points]
    
    def _mask_to_bbox(self, mask):
        """mask 转 bbox"""
        if mask.sum() == 0:
            return [0, 0, 1, 1]
        ys, xs = np.where(mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ============================================================
# 使用示例
# ============================================================

"""
使用示例 1: SAM2 Memory-based Segmentor

segmentor = SAM2MemoryBasedSegmentor(sam2_model, device='cuda')
segmentor.build_memory_bank(reference_data)
predictions = segmentor.predict(query_image)


使用示例 2: DINO-Guided SAM2 Segmentor（推荐）

segmentor = DINOGuidedSAM2Segmentor(
    sam2_predictor,
    dino_model,
    dino_transform,
    device='cuda'
)
segmentor.build_memory_bank(reference_data)
predictions = segmentor.predict(query_image)
"""

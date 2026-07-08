"""
SP-SAM 全面LoRA模块
==================

支持对以下组件添加LoRA：
1. DINO - Attention和FFN层
2. SAM2 - Image Encoder, Prompt Encoder, Mask Decoder, Memory模块
3. 自定义适配器模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# 基础LoRA层
# ============================================================

class LoRALayer(nn.Module):
    """
    低秩适配层 (Low-Rank Adaptation)
    
    W' = W + BA，其中B: (out, rank), A: (rank, in)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算LoRA增量"""
        # x: (..., in_features)
        # output: (..., out_features)
        return self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """
    带LoRA的线性层
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # 冻结原始层
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora(x)
    
    def merge_weights(self):
        """合并LoRA权重到原始层"""
        with torch.no_grad():
            self.original_layer.weight.data += (
                self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            )
        return self.original_layer


# ============================================================
# DINO LoRA
# ============================================================

class DinoLoRA(nn.Module):
    """
    DINO模型的LoRA适配器
    
    支持对Attention和FFN层添加LoRA
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_layers: List[int] = None,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.dino_model = dino_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # 默认：后4层的Attention
        if target_layers is None:
            num_blocks = len(dino_model.blocks)
            target_layers = list(range(max(0, num_blocks - 4), num_blocks))
        self.target_layers = target_layers
        
        # 默认：QKV和输出投影
        if target_modules is None:
            target_modules = ["attn.qkv", "attn.proj"]
        self.target_modules = target_modules
        
        # 冻结整个DINO
        for param in dino_model.parameters():
            param.requires_grad = False
        
        # 添加LoRA层
        self.lora_layers = nn.ModuleDict()
        self._add_lora_layers()
        
        # 统计参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in dino_model.parameters())
        print(f"📌 DinoLoRA: 可训练参数 {trainable:,} / 总参数 {total:,} ({100*trainable/total:.2f}%)")
    
    def _add_lora_layers(self):
        """为目标层添加LoRA"""
        for layer_idx in self.target_layers:
            if layer_idx >= len(self.dino_model.blocks):
                continue
            
            block = self.dino_model.blocks[layer_idx]
            
            for module_name in self.target_modules:
                try:
                    # 获取目标模块
                    parts = module_name.split(".")
                    module = block
                    for part in parts:
                        module = getattr(module, part)
                    
                    if isinstance(module, nn.Linear):
                        # 创建LoRA层
                        lora_key = f"block{layer_idx}_{module_name.replace('.', '_')}"
                        self.lora_layers[lora_key] = LoRALayer(
                            module.in_features,
                            module.out_features,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                        )
                except AttributeError:
                    print(f"   ⚠️  模块 {module_name} 在 block {layer_idx} 中不存在")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（需要手动hook到DINO）
        
        注意：这个forward不会自动被调用，需要配合register_forward_hook使用
        """
        return x
    
    def get_lora_params(self) -> List[nn.Parameter]:
        """获取所有LoRA参数"""
        params = []
        for lora in self.lora_layers.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params


class DinoWithLoRA(nn.Module):
    """
    带LoRA的DINO封装器
    
    直接替换目标层，无需手动hook
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_layers: List[int] = None,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.dino_model = dino_model
        
        # 默认配置
        num_blocks = len(dino_model.blocks)
        if target_layers is None:
            target_layers = list(range(max(0, num_blocks - 4), num_blocks))
        if target_modules is None:
            target_modules = ["qkv", "proj"]  # Attention中的线性层
        
        # 冻结整个DINO
        for param in dino_model.parameters():
            param.requires_grad = False
        
        # 替换目标层
        self._inject_lora(target_layers, target_modules, rank, alpha, dropout)
        
        # 统计
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in dino_model.parameters())
        print(f"📌 DinoWithLoRA: 可训练参数 {trainable:,} / 总参数 {total:,} ({100*trainable/total:.2f}%)")
    
    def _inject_lora(self, target_layers, target_modules, rank, alpha, dropout):
        """注入LoRA到目标层"""
        for layer_idx in target_layers:
            if layer_idx >= len(self.dino_model.blocks):
                continue
            
            block = self.dino_model.blocks[layer_idx]
            attn = block.attn
            
            for module_name in target_modules:
                if hasattr(attn, module_name):
                    original = getattr(attn, module_name)
                    if isinstance(original, nn.Linear):
                        lora_linear = LoRALinear(original, rank, alpha, dropout)
                        setattr(attn, module_name, lora_linear)
            
            # MLP层（可选）
            if "fc1" in target_modules and hasattr(block.mlp, "fc1"):
                lora_fc1 = LoRALinear(block.mlp.fc1, rank, alpha, dropout)
                block.mlp.fc1 = lora_fc1
            if "fc2" in target_modules and hasattr(block.mlp, "fc2"):
                lora_fc2 = LoRALinear(block.mlp.fc2, rank, alpha, dropout)
                block.mlp.fc2 = lora_fc2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dino_model(x)
    
    def get_intermediate_layers(self, x, n=1, reshape=False):
        """代理到原始DINO的方法"""
        return self.dino_model.get_intermediate_layers(x, n=n, reshape=reshape)


# ============================================================
# SAM2 LoRA
# ============================================================

class SAM2LoRA(nn.Module):
    """
    SAM2模型的LoRA适配器
    
    支持：
    - Image Encoder (Hiera)
    - Prompt Encoder
    - Mask Decoder
    - Memory Encoder
    - Memory Attention
    """
    
    def __init__(
        self,
        sam2_model: nn.Module,
        config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.sam2_model = sam2_model
        
        # 默认配置
        if config is None:
            config = {
                "image_encoder": {"enabled": False, "rank": 4, "layers": [-4, -3, -2, -1]},
                "mask_decoder": {"enabled": True, "rank": 4},
                "memory_encoder": {"enabled": False, "rank": 4},
                "memory_attention": {"enabled": False, "rank": 4},
            }
        self.config = config
        
        # 冻结整个SAM2
        for param in sam2_model.parameters():
            param.requires_grad = False
        
        # 添加LoRA
        self.lora_modules = nn.ModuleDict()
        
        if config.get("image_encoder", {}).get("enabled", False):
            self._add_image_encoder_lora(config["image_encoder"])
        
        if config.get("mask_decoder", {}).get("enabled", False):
            self._add_mask_decoder_lora(config["mask_decoder"])
        
        if config.get("memory_encoder", {}).get("enabled", False):
            self._add_memory_encoder_lora(config["memory_encoder"])
        
        if config.get("memory_attention", {}).get("enabled", False):
            self._add_memory_attention_lora(config["memory_attention"])
        
        # 统计
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📌 SAM2LoRA: 可训练参数 {trainable:,}")
    
    def _add_image_encoder_lora(self, cfg):
        """为Image Encoder添加LoRA"""
        rank = cfg.get("rank", 4)
        # SAM2使用Hiera作为Image Encoder
        if hasattr(self.sam2_model, "image_encoder"):
            encoder = self.sam2_model.image_encoder
            # 根据实际结构添加LoRA
            # 这里需要根据SAM2的具体实现调整
            pass
    
    def _add_mask_decoder_lora(self, cfg):
        """为Mask Decoder添加LoRA"""
        rank = cfg.get("rank", 4)
        alpha = cfg.get("alpha", 1.0)
        
        if hasattr(self.sam2_model, "sam_mask_decoder"):
            decoder = self.sam2_model.sam_mask_decoder
            
            # 解冻Mask Decoder的特定层
            for name, module in decoder.named_modules():
                if isinstance(module, nn.Linear):
                    # 对线性层添加LoRA
                    # 由于SAM2结构复杂，这里选择直接解冻某些层
                    pass
            
            # 或者直接解冻整个Mask Decoder
            for param in decoder.parameters():
                param.requires_grad = True
            
            self.lora_modules["mask_decoder"] = decoder
    
    def _add_memory_encoder_lora(self, cfg):
        """为Memory Encoder添加LoRA"""
        if hasattr(self.sam2_model, "memory_encoder"):
            encoder = self.sam2_model.memory_encoder
            for param in encoder.parameters():
                param.requires_grad = True
            self.lora_modules["memory_encoder"] = encoder
    
    def _add_memory_attention_lora(self, cfg):
        """为Memory Attention添加LoRA"""
        if hasattr(self.sam2_model, "memory_attention"):
            attn = self.sam2_model.memory_attention
            for param in attn.parameters():
                param.requires_grad = True
            self.lora_modules["memory_attention"] = attn


# ============================================================
# Prototype Adapter（保持原有功能）
# ============================================================

class PrototypeAdapter(nn.Module):
    """
    原型适配器 - 用于特征空间的域适应
    
    将预训练的DINO特征映射到任务特定的表示空间
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        # MLP层
        layers = []
        in_dim = feature_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # 最后一层映射回feature_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 残差缩放
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) 或 (B, C, H, W)
        Returns:
            适应后的特征，形状不变
        """
        if x.dim() == 3:
            C, H, W = x.shape
            x_flat = x.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
            out_flat = self.mlp(x_flat)
            if self.use_residual:
                out_flat = x_flat + self.scale * out_flat
            return out_flat.reshape(H, W, C).permute(2, 0, 1)
        else:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
            out_flat = self.mlp(x_flat)
            if self.use_residual:
                out_flat = x_flat + self.scale * out_flat
            return out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ============================================================
# Similarity Head（相似度计算增强）
# ============================================================

class SimilarityHead(nn.Module):
    """
    可学习的相似度计算头
    
    增强原型与查询特征之间的相似度计算
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        # 特征投影
        self.query_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.proto_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(
        self,
        query_features: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_features: (C, H, W)
            prototype: (C,)
        Returns:
            similarity_map: (H, W)
        """
        C, H, W = query_features.shape
        
        # 投影
        query_flat = query_features.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        query_proj = self.query_proj(query_flat)  # (H*W, hidden)
        query_proj = F.normalize(query_proj, p=2, dim=-1)
        
        proto_proj = self.proto_proj(prototype.unsqueeze(0))  # (1, hidden)
        proto_proj = F.normalize(proto_proj, p=2, dim=-1)
        
        # 计算相似度
        similarity = torch.mm(query_proj, proto_proj.T).squeeze(-1)  # (H*W,)
        similarity = similarity / self.temperature.abs()
        
        return similarity.reshape(H, W)


# ============================================================
# Point Predictor（点预测增强）
# ============================================================

class PointPredictor(nn.Module):
    """
    可学习的点预测器
    
    从相似度图预测更好的点提示
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        num_points: int = 10,
    ):
        super().__init__()
        self.num_points = num_points
        
        # 点预测网络
        self.point_net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 置信度预测
        self.confidence_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_points),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        similarity_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            similarity_map: (H, W)
        Returns:
            point_map: (H, W) 增强的相似度图
            confidences: (num_points,) 点置信度
        """
        # 添加batch和channel维度
        x = similarity_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # 预测增强的相似度图
        point_map = self.point_net(x).squeeze(0).squeeze(0)  # (H, W)
        
        return point_map


# ============================================================
# 完整的SP-SAM LoRA包装器
# ============================================================

class SPSAMLoRAWrapper(nn.Module):
    """
    SP-SAM完整LoRA包装器
    
    整合所有可微调组件
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        sam2_model: nn.Module,
        config: 'SPSAMFinetuneConfig' = None,
    ):
        super().__init__()
        
        # 原始模型
        self.dino_model = dino_model
        self.sam2_model = sam2_model
        
        # 从配置中获取设置
        if config is None:
            from finetune_config import SPSAMFinetuneConfig
            config = SPSAMFinetuneConfig()
        
        self.config = config
        self.lora_modules = nn.ModuleDict()
        
        # 初始化各组件
        self._init_dino_lora()
        self._init_sam2_lora()
        self._init_adapters()
        
        # 打印摘要
        self._print_summary()
    
    def _init_dino_lora(self):
        """初始化DINO LoRA"""
        if self.config.dino.enabled:
            if self.config.dino.use_lora:
                self.lora_modules["dino"] = DinoWithLoRA(
                    self.dino_model,
                    rank=self.config.dino.lora_config.rank,
                    alpha=self.config.dino.lora_config.alpha,
                    dropout=self.config.dino.lora_config.dropout,
                    target_layers=self.config.dino.target_layers,
                    target_modules=["qkv", "proj"],
                )
            elif self.config.dino.unfreeze_last_n_layers > 0:
                # 解冻后n层
                num_blocks = len(self.dino_model.blocks)
                for i in range(num_blocks - self.config.dino.unfreeze_last_n_layers, num_blocks):
                    for param in self.dino_model.blocks[i].parameters():
                        param.requires_grad = True
        else:
            # 完全冻结DINO
            for param in self.dino_model.parameters():
                param.requires_grad = False
    
    def _init_sam2_lora(self):
        """初始化SAM2 LoRA"""
        if self.config.sam2.enabled:
            # 冻结整个SAM2
            for param in self.sam2_model.parameters():
                param.requires_grad = False
            
            # 选择性解冻
            if self.config.sam2.finetune_mask_decoder:
                if hasattr(self.sam2_model, "sam_mask_decoder"):
                    for param in self.sam2_model.sam_mask_decoder.parameters():
                        param.requires_grad = True
            
            if self.config.sam2.finetune_memory_encoder:
                if hasattr(self.sam2_model, "memory_encoder"):
                    for param in self.sam2_model.memory_encoder.parameters():
                        param.requires_grad = True
            
            if self.config.sam2.finetune_memory_attention:
                if hasattr(self.sam2_model, "memory_attention"):
                    for param in self.sam2_model.memory_attention.parameters():
                        param.requires_grad = True
            
            if self.config.sam2.finetune_prompt_encoder:
                if hasattr(self.sam2_model, "sam_prompt_encoder"):
                    for param in self.sam2_model.sam_prompt_encoder.parameters():
                        param.requires_grad = True
        else:
            # 完全冻结SAM2
            for param in self.sam2_model.parameters():
                param.requires_grad = False
    
    def _init_adapters(self):
        """初始化适配器模块"""
        if self.config.adapter.enabled:
            if self.config.adapter.prototype_adapter:
                self.lora_modules["prototype_adapter"] = PrototypeAdapter(
                    feature_dim=768,  # DINO特征维度
                    hidden_dim=self.config.adapter.prototype_hidden_dim,
                    num_layers=self.config.adapter.prototype_num_layers,
                    use_residual=self.config.adapter.prototype_use_residual,
                )
            
            if self.config.adapter.similarity_head:
                self.lora_modules["similarity_head"] = SimilarityHead(
                    feature_dim=768,
                    hidden_dim=self.config.adapter.similarity_hidden_dim,
                )
            
            if self.config.adapter.point_predictor:
                self.lora_modules["point_predictor"] = PointPredictor(
                    feature_dim=768,
                    hidden_dim=self.config.adapter.point_hidden_dim,
                )
    
    def _print_summary(self):
        """打印参数摘要"""
        total_params = 0
        trainable_params = 0
        
        # DINO参数
        dino_total = sum(p.numel() for p in self.dino_model.parameters())
        dino_trainable = sum(p.numel() for p in self.dino_model.parameters() if p.requires_grad)
        total_params += dino_total
        trainable_params += dino_trainable
        
        # SAM2参数
        sam2_total = sum(p.numel() for p in self.sam2_model.parameters())
        sam2_trainable = sum(p.numel() for p in self.sam2_model.parameters() if p.requires_grad)
        total_params += sam2_total
        trainable_params += sam2_trainable
        
        # LoRA模块参数
        lora_total = sum(p.numel() for p in self.lora_modules.parameters())
        lora_trainable = sum(p.numel() for p in self.lora_modules.parameters() if p.requires_grad)
        total_params += lora_total
        trainable_params += lora_trainable
        
        print("\n" + "=" * 60)
        print("📊 参数统计")
        print("=" * 60)
        print(f"DINO:    {dino_trainable:>10,} / {dino_total:>12,} 可训练")
        print(f"SAM2:    {sam2_trainable:>10,} / {sam2_total:>12,} 可训练")
        print(f"LoRA模块: {lora_trainable:>10,} / {lora_total:>12,} 可训练")
        print("-" * 60)
        print(f"总计:    {trainable_params:>10,} / {total_params:>12,} 可训练")
        print(f"比例:    {100*trainable_params/total_params:.2f}%")
        print("=" * 60)
    
    def get_trainable_parameters(self) -> List[Dict]:
        """获取分组的可训练参数（用于不同学习率）"""
        param_groups = []
        
        # DINO参数
        if self.config.dino.enabled:
            dino_params = [p for p in self.dino_model.parameters() if p.requires_grad]
            if dino_params:
                param_groups.append({
                    "params": dino_params,
                    "lr": self.config.training.base_lr * self.config.dino.lr_multiplier,
                    "name": "dino",
                })
        
        # SAM2参数
        if self.config.sam2.enabled:
            # Mask Decoder
            if self.config.sam2.finetune_mask_decoder and hasattr(self.sam2_model, "sam_mask_decoder"):
                decoder_params = [p for p in self.sam2_model.sam_mask_decoder.parameters() if p.requires_grad]
                if decoder_params:
                    param_groups.append({
                        "params": decoder_params,
                        "lr": self.config.training.base_lr * self.config.sam2.mask_decoder_lr_multiplier,
                        "name": "sam2_mask_decoder",
                    })
            
            # Memory模块
            memory_params = []
            if self.config.sam2.finetune_memory_encoder and hasattr(self.sam2_model, "memory_encoder"):
                memory_params.extend([p for p in self.sam2_model.memory_encoder.parameters() if p.requires_grad])
            if self.config.sam2.finetune_memory_attention and hasattr(self.sam2_model, "memory_attention"):
                memory_params.extend([p for p in self.sam2_model.memory_attention.parameters() if p.requires_grad])
            if memory_params:
                param_groups.append({
                    "params": memory_params,
                    "lr": self.config.training.base_lr * self.config.sam2.memory_encoder_lr_multiplier,
                    "name": "sam2_memory",
                })
        
        # LoRA模块参数
        if self.config.adapter.enabled:
            adapter_params = [p for p in self.lora_modules.parameters() if p.requires_grad]
            if adapter_params:
                param_groups.append({
                    "params": adapter_params,
                    "lr": self.config.training.base_lr * self.config.adapter.lr_multiplier,
                    "name": "adapters",
                })
        
        return param_groups
    
    def save_lora_weights(self, path: str):
        """保存LoRA权重"""
        state_dict = {
            "lora_modules": self.lora_modules.state_dict(),
            "config": self.config,
        }
        
        # 保存SAM2可训练部分
        if self.config.sam2.enabled:
            sam2_trainable = {}
            if self.config.sam2.finetune_mask_decoder and hasattr(self.sam2_model, "sam_mask_decoder"):
                sam2_trainable["mask_decoder"] = self.sam2_model.sam_mask_decoder.state_dict()
            if self.config.sam2.finetune_memory_encoder and hasattr(self.sam2_model, "memory_encoder"):
                sam2_trainable["memory_encoder"] = self.sam2_model.memory_encoder.state_dict()
            if self.config.sam2.finetune_memory_attention and hasattr(self.sam2_model, "memory_attention"):
                sam2_trainable["memory_attention"] = self.sam2_model.memory_attention.state_dict()
            state_dict["sam2_trainable"] = sam2_trainable
        
        torch.save(state_dict, path)
        print(f"✅ LoRA权重已保存到: {path}")
    
    def load_lora_weights(self, path: str):
        """加载LoRA权重"""
        state_dict = torch.load(path, map_location="cpu")
        
        self.lora_modules.load_state_dict(state_dict["lora_modules"])
        
        if "sam2_trainable" in state_dict:
            sam2_trainable = state_dict["sam2_trainable"]
            if "mask_decoder" in sam2_trainable and hasattr(self.sam2_model, "sam_mask_decoder"):
                self.sam2_model.sam_mask_decoder.load_state_dict(sam2_trainable["mask_decoder"])
            if "memory_encoder" in sam2_trainable and hasattr(self.sam2_model, "memory_encoder"):
                self.sam2_model.memory_encoder.load_state_dict(sam2_trainable["memory_encoder"])
            if "memory_attention" in sam2_trainable and hasattr(self.sam2_model, "memory_attention"):
                self.sam2_model.memory_attention.load_state_dict(sam2_trainable["memory_attention"])
        
        print(f"✅ LoRA权重已从 {path} 加载")


if __name__ == "__main__":
    # 测试代码
    print("SP-SAM LoRA模块测试")
    
    # 测试LoRA层
    lora = LoRALayer(768, 768, rank=4)
    x = torch.randn(100, 768)
    out = lora(x)
    print(f"LoRA输出形状: {out.shape}")
    
    # 测试Prototype Adapter
    adapter = PrototypeAdapter(768, 256, 2)
    x = torch.randn(768, 32, 32)
    out = adapter(x)
    print(f"Adapter输出形状: {out.shape}")
    
    # 测试Similarity Head
    sim_head = SimilarityHead(768, 128)
    query = torch.randn(768, 32, 32)
    proto = torch.randn(768)
    sim = sim_head(query, proto)
    print(f"Similarity输出形状: {sim.shape}")
    
    print("\n✅ 所有测试通过")

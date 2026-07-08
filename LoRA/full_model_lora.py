"""
SP-SAM 全模型LoRA微调
======================

对整个模型的所有Transformer层添加LoRA：
1. DINO/DINOv3 - 所有Attention层的Q/K/V/Proj + MLP
2. SAM2 - Image Encoder + Mask Decoder + Memory模块

真正的端到端LoRA微调！
"""

import os
import sys
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# 路径设置
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LORA_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# 基础LoRA层
# ============================================================

class LoRALayer(nn.Module):
    """
    低秩适配层 (Low-Rank Adaptation)
    
    原理: W' = W + BA * (alpha/rank)
    - B: (out_features, rank) - 初始化为0
    - A: (rank, in_features) - Kaiming初始化
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
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算LoRA增量: dropout(x) @ A^T @ B^T * scaling
        """
        return self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
    
    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}'


class LoRALinear(nn.Module):
    """
    带LoRA的线性层包装器
    
    forward: original_output + lora_output
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
    
    # 代理原始Linear的属性
    @property
    def in_features(self):
        return self.original_linear.in_features
    
    @property
    def out_features(self):
        return self.original_linear.out_features
    
    @property
    def weight(self):
        return self.original_linear.weight
    
    @property
    def bias(self):
        return self.original_linear.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_linear(x) + self.lora(x)
    
    def merge_weights(self) -> nn.Linear:
        """合并LoRA权重到原始层（推理时使用）"""
        with torch.no_grad():
            delta_w = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            self.original_linear.weight.data += delta_w
        return self.original_linear


# ============================================================
# DINO/DINOv3 全模型LoRA
# ============================================================

class DinoFullLoRA(nn.Module):
    """
    DINO/DINOv3 全模型LoRA
    
    对所有Transformer Block添加LoRA：
    - Attention: qkv, proj
    - MLP: fc1, fc2
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: List[str] = None,
        freeze_patch_embed: bool = True,
        freeze_cls_token: bool = True,
    ):
        super().__init__()
        self.dino_model = dino_model
        self.rank = rank
        self.alpha = alpha
        
        # 默认目标模块
        if target_modules is None:
            target_modules = ["qkv", "proj", "fc1", "fc2"]
        self.target_modules = target_modules
        
        # 冻结整个模型
        for param in dino_model.parameters():
            param.requires_grad = False
        
        # 可选：解冻某些组件
        if not freeze_patch_embed and hasattr(dino_model, 'patch_embed'):
            for param in dino_model.patch_embed.parameters():
                param.requires_grad = True
        
        if not freeze_cls_token and hasattr(dino_model, 'cls_token'):
            dino_model.cls_token.requires_grad = True
        
        # 注入LoRA到所有blocks
        self.lora_count = 0
        self._inject_lora_to_all_blocks(rank, alpha, dropout)
        
        # 统计
        self._print_stats()
    
    def _inject_lora_to_all_blocks(self, rank, alpha, dropout):
        """为所有Transformer块注入LoRA"""
        blocks = self.dino_model.blocks
        
        for block_idx, block in enumerate(blocks):
            # Attention模块
            attn = block.attn
            
            # QKV (通常是一个合并的线性层)
            if "qkv" in self.target_modules and hasattr(attn, 'qkv'):
                if isinstance(attn.qkv, nn.Linear):
                    device = attn.qkv.weight.device
                    lora_linear = LoRALinear(attn.qkv, rank, alpha, dropout)
                    lora_linear.to(device)
                    attn.qkv = lora_linear
                    self.lora_count += 1
            
            # Projection
            if "proj" in self.target_modules and hasattr(attn, 'proj'):
                if isinstance(attn.proj, nn.Linear):
                    device = attn.proj.weight.device
                    lora_linear = LoRALinear(attn.proj, rank, alpha, dropout)
                    lora_linear.to(device)
                    attn.proj = lora_linear
                    self.lora_count += 1
            
            # MLP模块
            mlp = block.mlp
            
            if "fc1" in self.target_modules and hasattr(mlp, 'fc1'):
                if isinstance(mlp.fc1, nn.Linear):
                    device = mlp.fc1.weight.device
                    lora_linear = LoRALinear(mlp.fc1, rank, alpha, dropout)
                    lora_linear.to(device)
                    mlp.fc1 = lora_linear
                    self.lora_count += 1
            
            if "fc2" in self.target_modules and hasattr(mlp, 'fc2'):
                if isinstance(mlp.fc2, nn.Linear):
                    device = mlp.fc2.weight.device
                    lora_linear = LoRALinear(mlp.fc2, rank, alpha, dropout)
                    lora_linear.to(device)
                    mlp.fc2 = lora_linear
                    self.lora_count += 1
    
    def _print_stats(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.dino_model.parameters())
        trainable_params = sum(p.numel() for p in self.dino_model.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"📌 DINO 全模型LoRA")
        print(f"{'='*60}")
        print(f"  LoRA层数量: {self.lora_count}")
        print(f"  LoRA Rank: {self.rank}")
        print(f"  目标模块: {self.target_modules}")
        print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"{'='*60}\n")
    
    def forward(self, x: torch.Tensor):
        """直接使用原始模型的forward"""
        return self.dino_model(x)
    
    def get_intermediate_layers(self, x, n=1, reshape=False):
        """代理到原始模型"""
        return self.dino_model.get_intermediate_layers(x, n=n, reshape=reshape)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取所有LoRA参数"""
        params = []
        for module in self.dino_model.modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora.lora_A, module.lora.lora_B])
        return params
    
    def save_lora_weights(self, path: str):
        """保存LoRA权重"""
        lora_state_dict = {}
        for name, module in self.dino_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'target_modules': self.target_modules,
        }, path)
        print(f"✅ DINO LoRA权重已保存: {path}")
    
    def load_lora_weights(self, path: str):
        """加载LoRA权重"""
        checkpoint = torch.load(path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        for name, module in self.dino_model.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state_dict:
                    module.lora.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                    module.lora.lora_B.data = lora_state_dict[f"{name}.lora_B"]
        
        print(f"✅ DINO LoRA权重已加载: {path}")


# ============================================================
# SAM2 全模型LoRA
# ============================================================

class SAM2FullLoRA(nn.Module):
    """
    SAM2 全模型LoRA
    
    对SAM2的多个组件添加LoRA：
    - Image Encoder (Hiera backbone)
    - Mask Decoder
    - Memory Encoder
    - Memory Attention
    """
    
    def __init__(
        self,
        sam2_model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        finetune_image_encoder: bool = True,
        finetune_mask_decoder: bool = True,
        finetune_prompt_encoder: bool = False,
        finetune_memory_encoder: bool = True,
        finetune_memory_attention: bool = True,
    ):
        super().__init__()
        self.sam2_model = sam2_model
        self.rank = rank
        self.alpha = alpha
        
        # 配置
        self.config = {
            'image_encoder': finetune_image_encoder,
            'mask_decoder': finetune_mask_decoder,
            'prompt_encoder': finetune_prompt_encoder,
            'memory_encoder': finetune_memory_encoder,
            'memory_attention': finetune_memory_attention,
        }
        
        # 冻结整个模型
        for param in sam2_model.parameters():
            param.requires_grad = False
        
        # 注入LoRA
        self.lora_count = 0
        
        if finetune_image_encoder:
            self._inject_lora_to_image_encoder(rank, alpha, dropout)
        
        if finetune_mask_decoder:
            self._inject_lora_to_mask_decoder(rank, alpha, dropout)
        
        if finetune_prompt_encoder:
            self._inject_lora_to_prompt_encoder(rank, alpha, dropout)
        
        if finetune_memory_encoder:
            self._inject_lora_to_memory_encoder(rank, alpha, dropout)
        
        if finetune_memory_attention:
            self._inject_lora_to_memory_attention(rank, alpha, dropout)
        
        # 打印统计
        self._print_stats()
    
    def _inject_lora_to_linear(self, module: nn.Module, rank, alpha, dropout, prefix=""):
        """递归地为模块中的所有Linear层添加LoRA"""
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # 获取原始设备
                device = child.weight.device
                # 创建LoRA包装
                lora_linear = LoRALinear(child, rank, alpha, dropout)
                lora_linear.to(device)
                setattr(module, name, lora_linear)
                self.lora_count += 1
            else:
                # 递归处理子模块
                self._inject_lora_to_linear(child, rank, alpha, dropout, full_name)
    
    def _inject_lora_to_image_encoder(self, rank, alpha, dropout):
        """为Image Encoder添加LoRA"""
        if hasattr(self.sam2_model, 'image_encoder'):
            encoder = self.sam2_model.image_encoder
            self._inject_lora_to_linear(encoder, rank, alpha, dropout, "image_encoder")
            print(f"  ✓ Image Encoder LoRA已注入")
    
    def _inject_lora_to_mask_decoder(self, rank, alpha, dropout):
        """为Mask Decoder添加LoRA"""
        if hasattr(self.sam2_model, 'sam_mask_decoder'):
            decoder = self.sam2_model.sam_mask_decoder
            self._inject_lora_to_linear(decoder, rank, alpha, dropout, "mask_decoder")
            print(f"  ✓ Mask Decoder LoRA已注入")
    
    def _inject_lora_to_prompt_encoder(self, rank, alpha, dropout):
        """为Prompt Encoder添加LoRA"""
        if hasattr(self.sam2_model, 'sam_prompt_encoder'):
            encoder = self.sam2_model.sam_prompt_encoder
            self._inject_lora_to_linear(encoder, rank, alpha, dropout, "prompt_encoder")
            print(f"  ✓ Prompt Encoder LoRA已注入")
    
    def _inject_lora_to_memory_encoder(self, rank, alpha, dropout):
        """为Memory Encoder添加LoRA"""
        if hasattr(self.sam2_model, 'memory_encoder'):
            encoder = self.sam2_model.memory_encoder
            self._inject_lora_to_linear(encoder, rank, alpha, dropout, "memory_encoder")
            print(f"  ✓ Memory Encoder LoRA已注入")
    
    def _inject_lora_to_memory_attention(self, rank, alpha, dropout):
        """为Memory Attention添加LoRA"""
        if hasattr(self.sam2_model, 'memory_attention'):
            attn = self.sam2_model.memory_attention
            self._inject_lora_to_linear(attn, rank, alpha, dropout, "memory_attention")
            print(f"  ✓ Memory Attention LoRA已注入")
    
    def _print_stats(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.sam2_model.parameters())
        trainable_params = sum(p.numel() for p in self.sam2_model.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"📌 SAM2 全模型LoRA")
        print(f"{'='*60}")
        print(f"  LoRA层数量: {self.lora_count}")
        print(f"  LoRA Rank: {self.rank}")
        print(f"  微调组件: {[k for k,v in self.config.items() if v]}")
        print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"{'='*60}\n")
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取所有LoRA参数"""
        params = []
        for module in self.sam2_model.modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora.lora_A, module.lora.lora_B])
        return params
    
    def save_lora_weights(self, path: str):
        """保存LoRA权重"""
        lora_state_dict = {}
        for name, module in self.sam2_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'config': self.config,
        }, path)
        print(f"✅ SAM2 LoRA权重已保存: {path}")
    
    def load_lora_weights(self, path: str):
        """加载LoRA权重"""
        checkpoint = torch.load(path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        for name, module in self.sam2_model.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state_dict:
                    module.lora.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                    module.lora.lora_B.data = lora_state_dict[f"{name}.lora_B"]
        
        print(f"✅ SAM2 LoRA权重已加载: {path}")


# ============================================================
# Prototype Adapter (额外的适配层)
# ============================================================

class PrototypeAdapter(nn.Module):
    """
    原型适配器 - 用于特征空间的域适应
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
        self.use_residual = use_residual
        
        # MLP
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
        layers.append(nn.Linear(in_dim, feature_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
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
        """
        if x.dim() == 3:
            C, H, W = x.shape
            x_flat = x.permute(1, 2, 0).reshape(-1, C)
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
# SP-SAM 完整LoRA包装器
# ============================================================

class SPSAMFullLoRA(nn.Module):
    """
    SP-SAM 完整模型LoRA包装器
    
    整合DINO LoRA + SAM2 LoRA + Prototype Adapter
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        dino_transform,
        sam2_model: nn.Module,
        sam2_predictor,
        device: str = 'cuda',
        # DINO LoRA配置
        dino_lora_rank: int = 4,
        dino_lora_alpha: float = 1.0,
        dino_target_modules: List[str] = None,
        # SAM2 LoRA配置  
        sam2_lora_rank: int = 4,
        sam2_lora_alpha: float = 1.0,
        sam2_finetune_image_encoder: bool = True,
        sam2_finetune_mask_decoder: bool = True,
        sam2_finetune_memory_encoder: bool = True,
        sam2_finetune_memory_attention: bool = True,
        # Adapter配置
        use_prototype_adapter: bool = True,
        adapter_hidden_dim: int = 256,
    ):
        super().__init__()
        self.device = device
        self.dino_transform = dino_transform
        self.sam2_predictor = sam2_predictor
        
        print("\n" + "="*60)
        print("🚀 SP-SAM 全模型LoRA初始化")
        print("="*60)
        
        # 1. DINO LoRA
        print("\n📦 初始化DINO LoRA...")
        self.dino_lora = DinoFullLoRA(
            dino_model=dino_model,
            rank=dino_lora_rank,
            alpha=dino_lora_alpha,
            target_modules=dino_target_modules or ["qkv", "proj", "fc1", "fc2"],
        )
        
        # 2. SAM2 LoRA
        print("\n📦 初始化SAM2 LoRA...")
        self.sam2_lora = SAM2FullLoRA(
            sam2_model=sam2_model,
            rank=sam2_lora_rank,
            alpha=sam2_lora_alpha,
            finetune_image_encoder=sam2_finetune_image_encoder,
            finetune_mask_decoder=sam2_finetune_mask_decoder,
            finetune_memory_encoder=sam2_finetune_memory_encoder,
            finetune_memory_attention=sam2_finetune_memory_attention,
        )
        
        # 3. Prototype Adapter
        if use_prototype_adapter:
            # 获取DINO特征维度
            if hasattr(dino_model, 'embed_dim'):
                feature_dim = dino_model.embed_dim
            else:
                feature_dim = 768  # 默认
            
            print(f"\n📦 初始化Prototype Adapter (dim={feature_dim})...")
            self.prototype_adapter = PrototypeAdapter(
                feature_dim=feature_dim,
                hidden_dim=adapter_hidden_dim,
            )
        else:
            self.prototype_adapter = None
        
        # 移动所有模块到设备
        self.dino_lora.dino_model.to(device)
        self.sam2_lora.sam2_model.to(device)
        if self.prototype_adapter is not None:
            self.prototype_adapter.to(device)
        
        # 打印总体统计
        self._print_total_stats()
    
    def _print_total_stats(self):
        """打印总体统计"""
        total_params = 0
        trainable_params = 0
        
        # DINO
        for p in self.dino_lora.dino_model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        
        # SAM2
        for p in self.sam2_lora.sam2_model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        
        # Adapter
        if self.prototype_adapter is not None:
            for p in self.prototype_adapter.parameters():
                total_params += p.numel()
                trainable_params += p.numel()
        
        print("\n" + "="*60)
        print("📊 总体参数统计")
        print("="*60)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  训练比例: {100*trainable_params/total_params:.2f}%")
        print(f"  DINO LoRA数: {self.dino_lora.lora_count}")
        print(f"  SAM2 LoRA数: {self.sam2_lora.lora_count}")
        print("="*60 + "\n")
    
    def get_all_trainable_parameters(self) -> List[Dict]:
        """获取所有可训练参数（分组）"""
        param_groups = []
        
        # DINO LoRA参数
        dino_params = self.dino_lora.get_lora_parameters()
        if dino_params:
            param_groups.append({
                'params': dino_params,
                'name': 'dino_lora',
            })
        
        # SAM2 LoRA参数
        sam2_params = self.sam2_lora.get_lora_parameters()
        if sam2_params:
            param_groups.append({
                'params': sam2_params,
                'name': 'sam2_lora',
            })
        
        # Adapter参数
        if self.prototype_adapter is not None:
            param_groups.append({
                'params': list(self.prototype_adapter.parameters()),
                'name': 'prototype_adapter',
            })
        
        return param_groups
    
    def extract_features(self, img_pil) -> torch.Tensor:
        """提取DINO特征"""
        img_tensor = self.dino_transform(img_pil)[None].to(self.device)
        
        # 兼容不同PyTorch版本
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            features = self.dino_lora.get_intermediate_layers(
                img_tensor.to(torch.bfloat16)
            )[0]
            h = w = int(features.shape[1] ** 0.5)
            feature_map = features.reshape(1, h, w, -1).permute(0, 3, 1, 2).float()
        
        # 应用Prototype Adapter
        if self.prototype_adapter is not None:
            feature_map = self.prototype_adapter(feature_map)
        
        return feature_map.squeeze(0)
    
    def save_all_lora_weights(self, path: str):
        """保存所有LoRA权重"""
        state_dict = {
            'dino_lora': {},
            'sam2_lora': {},
            'prototype_adapter': None,
        }
        
        # DINO LoRA
        for name, module in self.dino_lora.dino_model.named_modules():
            if isinstance(module, LoRALinear):
                state_dict['dino_lora'][f"{name}.lora_A"] = module.lora.lora_A.data.cpu()
                state_dict['dino_lora'][f"{name}.lora_B"] = module.lora.lora_B.data.cpu()
        
        # SAM2 LoRA
        for name, module in self.sam2_lora.sam2_model.named_modules():
            if isinstance(module, LoRALinear):
                state_dict['sam2_lora'][f"{name}.lora_A"] = module.lora.lora_A.data.cpu()
                state_dict['sam2_lora'][f"{name}.lora_B"] = module.lora.lora_B.data.cpu()
        
        # Prototype Adapter
        if self.prototype_adapter is not None:
            state_dict['prototype_adapter'] = self.prototype_adapter.state_dict()
        
        # 配置信息
        state_dict['config'] = {
            'dino_lora_rank': self.dino_lora.rank,
            'dino_lora_alpha': self.dino_lora.alpha,
            'sam2_lora_rank': self.sam2_lora.rank,
            'sam2_lora_alpha': self.sam2_lora.alpha,
        }
        
        torch.save(state_dict, path)
        print(f"✅ 所有LoRA权重已保存: {path}")
    
    def load_all_lora_weights(self, path: str):
        """加载所有LoRA权重"""
        state_dict = torch.load(path, map_location='cpu')
        
        # DINO LoRA
        for name, module in self.dino_lora.dino_model.named_modules():
            if isinstance(module, LoRALinear):
                key_a = f"{name}.lora_A"
                key_b = f"{name}.lora_B"
                if key_a in state_dict['dino_lora']:
                    module.lora.lora_A.data = state_dict['dino_lora'][key_a].to(self.device)
                    module.lora.lora_B.data = state_dict['dino_lora'][key_b].to(self.device)
        
        # SAM2 LoRA
        for name, module in self.sam2_lora.sam2_model.named_modules():
            if isinstance(module, LoRALinear):
                key_a = f"{name}.lora_A"
                key_b = f"{name}.lora_B"
                if key_a in state_dict['sam2_lora']:
                    module.lora.lora_A.data = state_dict['sam2_lora'][key_a].to(self.device)
                    module.lora.lora_B.data = state_dict['sam2_lora'][key_b].to(self.device)
        
        # Prototype Adapter
        if self.prototype_adapter is not None and state_dict['prototype_adapter'] is not None:
            self.prototype_adapter.load_state_dict(state_dict['prototype_adapter'])
        
        print(f"✅ 所有LoRA权重已加载: {path}")


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SP-SAM 全模型LoRA 测试")
    print("=" * 60)
    
    # 测试LoRA层
    print("\n测试 LoRALayer...")
    lora = LoRALayer(768, 768, rank=4)
    x = torch.randn(100, 768)
    out = lora(x)
    print(f"  输入: {x.shape}, 输出: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in lora.parameters()):,}")
    
    # 测试LoRALinear
    print("\n测试 LoRALinear...")
    linear = nn.Linear(768, 768)
    lora_linear = LoRALinear(linear, rank=4)
    out = lora_linear(x)
    print(f"  输入: {x.shape}, 输出: {out.shape}")
    print(f"  可训练参数: {sum(p.numel() for p in lora_linear.parameters() if p.requires_grad):,}")
    
    # 测试Prototype Adapter
    print("\n测试 PrototypeAdapter...")
    adapter = PrototypeAdapter(768, 256, 2)
    x = torch.randn(768, 32, 32)
    out = adapter(x)
    print(f"  输入: {x.shape}, 输出: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in adapter.parameters()):,}")
    
    print("\n✅ 所有测试通过!")
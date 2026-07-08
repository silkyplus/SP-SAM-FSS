"""
SP-SAM 全面微调配置
====================

支持微调的组件：
1. DINO - Vision Encoder (特征提取)
   - LoRA for Attention layers
   - LoRA for FFN layers
   - 解冻后几层

2. SAM2 - Segmentation Model
   - Image Encoder LoRA
   - Prompt Encoder 微调
   - Mask Decoder 微调
   - Memory Encoder 微调
   - Memory Attention 微调

3. Prototype Adapter (新增模块)
   - 轻量级特征适配

4. 混合策略
   - 渐进式解冻
   - 多阶段训练
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class FinetuneTarget(Enum):
    """可微调的目标组件"""
    # DINO组件
    DINO_ATTENTION = "dino_attention"          # DINO Attention层
    DINO_FFN = "dino_ffn"                      # DINO FFN层
    DINO_LAST_LAYERS = "dino_last_layers"      # DINO后几层完全解冻
    
    # SAM2组件
    SAM2_IMAGE_ENCODER = "sam2_image_encoder"  # SAM2图像编码器
    SAM2_PROMPT_ENCODER = "sam2_prompt_encoder" # SAM2提示编码器
    SAM2_MASK_DECODER = "sam2_mask_decoder"    # SAM2掩码解码器
    SAM2_MEMORY_ENCODER = "sam2_memory_encoder" # SAM2记忆编码器
    SAM2_MEMORY_ATTENTION = "sam2_memory_attention" # SAM2记忆注意力
    
    # 新增适配器
    PROTOTYPE_ADAPTER = "prototype_adapter"    # 原型适配器
    SIMILARITY_HEAD = "similarity_head"        # 相似度计算头
    POINT_PREDICTOR = "point_predictor"        # 点预测器


@dataclass
class LoRAConfig:
    """LoRA配置"""
    rank: int = 4                    # LoRA秩
    alpha: float = 1.0               # 缩放因子
    dropout: float = 0.0             # Dropout
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class DinoFinetuneConfig:
    """DINO微调配置"""
    enabled: bool = False
    
    # LoRA配置
    use_lora: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    # 目标层
    target_layers: List[int] = field(default_factory=lambda: [20, 21, 22, 23])  # 后4层
    target_modules: List[str] = field(default_factory=lambda: ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    
    # 完全解冻选项
    unfreeze_last_n_layers: int = 0  # 解冻后n层（0表示不解冻）
    
    # 训练参数
    lr_multiplier: float = 0.1       # 相对于基础学习率的倍数


@dataclass
class SAM2FinetuneConfig:
    """SAM2微调配置"""
    enabled: bool = False
    
    # 各组件开关
    finetune_image_encoder: bool = False
    finetune_prompt_encoder: bool = False
    finetune_mask_decoder: bool = True   # 默认微调Mask Decoder
    finetune_memory_encoder: bool = False
    finetune_memory_attention: bool = False
    
    # Image Encoder LoRA配置
    image_encoder_lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        rank=4, target_modules=["attn.qkv", "attn.proj"]
    ))
    image_encoder_target_layers: List[int] = field(default_factory=lambda: [28, 29, 30, 31])
    
    # Mask Decoder配置
    mask_decoder_lr_multiplier: float = 1.0
    
    # Memory模块配置
    memory_encoder_lr_multiplier: float = 0.5
    memory_attention_lr_multiplier: float = 0.5


@dataclass
class AdapterConfig:
    """适配器配置"""
    enabled: bool = True  # 默认启用
    
    # Prototype Adapter
    prototype_adapter: bool = True
    prototype_hidden_dim: int = 256
    prototype_num_layers: int = 2
    prototype_use_residual: bool = True
    
    # Similarity Head (相似度计算增强)
    similarity_head: bool = False
    similarity_hidden_dim: int = 128
    
    # Point Predictor (点预测增强)
    point_predictor: bool = False
    point_hidden_dim: int = 128
    
    # 学习率
    lr_multiplier: float = 1.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    epochs: int = 100
    batch_size: int = 1
    base_lr: float = 1e-4
    weight_decay: float = 0.01
    
    # 优化器
    optimizer: str = "adamw"         # adamw, sgd, adam
    
    # 学习率调度
    scheduler: str = "cosine"        # cosine, step, onecycle, constant
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 梯度
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # 混合精度
    use_amp: bool = True
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # 早停
    early_stopping_patience: int = 20
    
    # 保存
    save_every: int = 10
    save_best_only: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = "jiguangdatasets"
    
    # Few-shot设置
    k_shot: int = 1
    
    # 数据增强
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotate_prob: float = 0.3
    rotate_range: int = 15
    color_jitter: bool = True
    
    # 采样
    episodes_per_epoch: int = 100
    
    # 图像大小
    image_size: int = 518  # DINO输入大小


@dataclass 
class LossConfig:
    """损失函数配置"""
    # 主损失
    use_bce: bool = True
    use_dice: bool = True
    use_focal: bool = True
    
    # 权重
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    
    # Focal Loss参数
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 辅助损失
    use_contrastive: bool = True
    contrastive_weight: float = 0.1
    contrastive_margin: float = 0.5
    
    # 边界损失
    use_boundary: bool = False
    boundary_weight: float = 0.2


@dataclass
class SPSAMFinetuneConfig:
    """SP-SAM完整微调配置"""
    # 组件配置
    dino: DinoFinetuneConfig = field(default_factory=DinoFinetuneConfig)
    sam2: SAM2FinetuneConfig = field(default_factory=SAM2FinetuneConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # 设备
    device: str = "cuda"
    
    # 输出
    output_dir: str = "outputs"
    experiment_name: str = "finetune"
    
    def get_trainable_params_summary(self) -> Dict[str, Any]:
        """获取可训练参数摘要"""
        summary = {
            "dino": {
                "enabled": self.dino.enabled,
                "use_lora": self.dino.use_lora if self.dino.enabled else False,
                "target_layers": self.dino.target_layers if self.dino.enabled else [],
            },
            "sam2": {
                "enabled": self.sam2.enabled,
                "image_encoder": self.sam2.finetune_image_encoder,
                "prompt_encoder": self.sam2.finetune_prompt_encoder,
                "mask_decoder": self.sam2.finetune_mask_decoder,
                "memory_encoder": self.sam2.finetune_memory_encoder,
                "memory_attention": self.sam2.finetune_memory_attention,
            },
            "adapter": {
                "enabled": self.adapter.enabled,
                "prototype_adapter": self.adapter.prototype_adapter,
                "similarity_head": self.adapter.similarity_head,
                "point_predictor": self.adapter.point_predictor,
            }
        }
        return summary


# ============================================================
# 预设配置
# ============================================================

def get_preset_config(preset: str) -> SPSAMFinetuneConfig:
    """获取预设配置"""
    
    if preset == "adapter_only":
        # 最轻量：只训练Prototype Adapter
        config = SPSAMFinetuneConfig()
        config.dino.enabled = False
        config.sam2.enabled = False
        config.adapter.enabled = True
        config.adapter.prototype_adapter = True
        config.training.epochs = 100
        config.training.base_lr = 1e-4
        return config
    
    elif preset == "adapter_plus_decoder":
        # 中等：Adapter + SAM2 Mask Decoder
        config = SPSAMFinetuneConfig()
        config.dino.enabled = False
        config.sam2.enabled = True
        config.sam2.finetune_mask_decoder = True
        config.sam2.finetune_image_encoder = False
        config.adapter.enabled = True
        config.training.epochs = 150
        config.training.base_lr = 5e-5
        return config
    
    elif preset == "dino_lora":
        # DINO LoRA + Adapter
        config = SPSAMFinetuneConfig()
        config.dino.enabled = True
        config.dino.use_lora = True
        config.dino.lora_config.rank = 4
        config.dino.target_layers = [20, 21, 22, 23]
        config.sam2.enabled = False
        config.adapter.enabled = True
        config.training.epochs = 100
        config.training.base_lr = 1e-4
        return config
    
    elif preset == "dino_lora_plus_decoder":
        # DINO LoRA + Adapter + SAM2 Decoder
        config = SPSAMFinetuneConfig()
        config.dino.enabled = True
        config.dino.use_lora = True
        config.dino.lora_config.rank = 4
        config.dino.target_layers = [20, 21, 22, 23]
        config.sam2.enabled = True
        config.sam2.finetune_mask_decoder = True
        config.adapter.enabled = True
        config.training.epochs = 150
        config.training.base_lr = 5e-5
        return config
    
    elif preset == "full_finetune":
        # 全面微调（需要大数据集）
        config = SPSAMFinetuneConfig()
        config.dino.enabled = True
        config.dino.use_lora = True
        config.dino.lora_config.rank = 8
        config.dino.target_layers = [16, 17, 18, 19, 20, 21, 22, 23]
        config.sam2.enabled = True
        config.sam2.finetune_image_encoder = True
        config.sam2.finetune_mask_decoder = True
        config.sam2.finetune_memory_attention = True
        config.adapter.enabled = True
        config.adapter.similarity_head = True
        config.training.epochs = 200
        config.training.base_lr = 2e-5
        config.training.weight_decay = 0.05
        return config
    
    elif preset == "sam2_memory":
        # 专注Memory模块
        config = SPSAMFinetuneConfig()
        config.dino.enabled = False
        config.sam2.enabled = True
        config.sam2.finetune_memory_encoder = True
        config.sam2.finetune_memory_attention = True
        config.sam2.finetune_mask_decoder = True
        config.adapter.enabled = True
        config.training.epochs = 100
        config.training.base_lr = 5e-5
        return config
    
    elif preset == "progressive":
        # 渐进式训练配置（多阶段）
        # 注：需要配合渐进式训练脚本使用
        config = SPSAMFinetuneConfig()
        config.adapter.enabled = True
        config.training.epochs = 50  # 每阶段
        config.training.base_lr = 1e-4
        return config
    
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: adapter_only, adapter_plus_decoder, "
                        f"dino_lora, dino_lora_plus_decoder, full_finetune, sam2_memory, progressive")


def print_config(config: SPSAMFinetuneConfig):
    """打印配置信息"""
    print("=" * 60)
    print("SP-SAM 微调配置")
    print("=" * 60)
    
    print("\n📌 DINO配置:")
    print(f"   启用: {config.dino.enabled}")
    if config.dino.enabled:
        print(f"   使用LoRA: {config.dino.use_lora}")
        if config.dino.use_lora:
            print(f"   LoRA秩: {config.dino.lora_config.rank}")
        print(f"   目标层: {config.dino.target_layers}")
        print(f"   解冻后n层: {config.dino.unfreeze_last_n_layers}")
    
    print("\n📌 SAM2配置:")
    print(f"   启用: {config.sam2.enabled}")
    if config.sam2.enabled:
        print(f"   Image Encoder: {config.sam2.finetune_image_encoder}")
        print(f"   Prompt Encoder: {config.sam2.finetune_prompt_encoder}")
        print(f"   Mask Decoder: {config.sam2.finetune_mask_decoder}")
        print(f"   Memory Encoder: {config.sam2.finetune_memory_encoder}")
        print(f"   Memory Attention: {config.sam2.finetune_memory_attention}")
    
    print("\n📌 适配器配置:")
    print(f"   启用: {config.adapter.enabled}")
    if config.adapter.enabled:
        print(f"   Prototype Adapter: {config.adapter.prototype_adapter}")
        print(f"   Similarity Head: {config.adapter.similarity_head}")
        print(f"   Point Predictor: {config.adapter.point_predictor}")
    
    print("\n📌 训练配置:")
    print(f"   Epochs: {config.training.epochs}")
    print(f"   学习率: {config.training.base_lr}")
    print(f"   优化器: {config.training.optimizer}")
    print(f"   调度器: {config.training.scheduler}")
    
    print("\n📌 损失配置:")
    losses = []
    if config.loss.use_bce:
        losses.append(f"BCE({config.loss.bce_weight})")
    if config.loss.use_dice:
        losses.append(f"Dice({config.loss.dice_weight})")
    if config.loss.use_focal:
        losses.append(f"Focal({config.loss.focal_weight})")
    if config.loss.use_contrastive:
        losses.append(f"Contrastive({config.loss.contrastive_weight})")
    print(f"   损失函数: {' + '.join(losses)}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 测试预设配置
    for preset in ["adapter_only", "adapter_plus_decoder", "dino_lora", 
                   "dino_lora_plus_decoder", "full_finetune", "sam2_memory"]:
        print(f"\n\n{'='*60}")
        print(f"预设: {preset}")
        config = get_preset_config(preset)
        print_config(config)

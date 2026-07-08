"""
SP-SAM LoRA 训练配置
====================

提供多种训练策略和配置选项。
"""

import os
import sys

# 添加父目录到路径
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(LORA_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class LoRAConfig:
    """LoRA配置"""
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])


@dataclass
class PrototypeAdapterConfig:
    """Prototype Adapter配置"""
    hidden_dim: int = 256
    num_layers: int = 2
    use_residual: bool = True
    init_scale: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 4
    k_shot: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    contrastive_weight: float = 0.1
    use_augmentation: bool = True
    random_flip: bool = True
    random_rotate: bool = True
    color_jitter: bool = True
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    save_freq: int = 10
    eval_freq: int = 10
    log_freq: int = 1


@dataclass
class ModelConfig:
    """模型配置"""
    dino_model: str = 'dinov3_vitb16'
    sam2_model: str = 'large'
    freeze_dino: bool = True
    freeze_sam: bool = True
    freeze_dino_layers: List[int] = field(default_factory=lambda: list(range(8)))


@dataclass
class FullConfig:
    """完整配置"""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    adapter: PrototypeAdapterConfig = field(default_factory=PrototypeAdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data_root: str = 'jiguangdatasets'
    output_dir: str = 'LoRA/outputs'
    device: str = 'cuda'
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lora': self.lora.__dict__,
            'adapter': self.adapter.__dict__,
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'data_root': self.data_root,
            'output_dir': self.output_dir,
            'device': self.device,
            'seed': self.seed
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FullConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        config.lora = LoRAConfig(**data.get('lora', {}))
        config.adapter = PrototypeAdapterConfig(**data.get('adapter', {}))
        config.training = TrainingConfig(**data.get('training', {}))
        config.model = ModelConfig(**data.get('model', {}))
        config.data_root = data.get('data_root', 'jiguangdatasets')
        config.output_dir = data.get('output_dir', 'LoRA/outputs')
        config.device = data.get('device', 'cuda')
        config.seed = data.get('seed', 42)
        return config


def get_baseline_config() -> FullConfig:
    """基础配置"""
    config = FullConfig()
    config.training.epochs = 50
    config.training.lr = 1e-4
    config.lora.rank = 4
    return config


def get_small_data_config() -> FullConfig:
    """小数据量配置 (<50样本)"""
    config = FullConfig()
    config.training.epochs = 200
    config.training.lr = 5e-5
    config.training.weight_decay = 1e-3
    config.lora.rank = 8
    config.lora.dropout = 0.1
    config.adapter.hidden_dim = 128
    config.adapter.num_layers = 1
    return config


def get_medium_data_config() -> FullConfig:
    """中等数据量配置 (50-200样本)"""
    config = FullConfig()
    config.training.epochs = 100
    config.training.lr = 1e-4
    config.training.warmup_epochs = 10
    config.lora.rank = 4
    config.adapter.hidden_dim = 256
    config.adapter.num_layers = 2
    return config


def get_large_data_config() -> FullConfig:
    """大数据量配置 (>200样本)"""
    config = FullConfig()
    config.training.epochs = 50
    config.training.lr = 2e-4
    config.training.batch_size = 8
    config.lora.rank = 4
    config.adapter.hidden_dim = 256
    config.adapter.num_layers = 3
    config.model.freeze_dino_layers = list(range(6))
    return config


def get_high_quality_config() -> FullConfig:
    """高质量配置"""
    config = FullConfig()
    config.training.epochs = 200
    config.training.lr = 1e-4
    config.training.warmup_epochs = 20
    config.training.ema_decay = 0.9999
    config.training.use_augmentation = True
    config.lora.rank = 8
    config.lora.alpha = 2.0
    config.adapter.hidden_dim = 384
    config.adapter.num_layers = 3
    config.training.contrastive_weight = 0.2
    return config


class DataAugmentation:
    """数据增强"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def __call__(self, image, mask):
        import numpy as np
        from PIL import Image
        import random
        
        if self.config.random_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()
        
        if self.config.random_rotate and random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)
            mask = np.array(mask_pil)
        
        return image, mask


class EMAModel:
    """指数移动平均模型"""
    
    def __init__(self, model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_scheduler(optimizer, config: TrainingConfig, num_training_steps: int):
    """获取学习率调度器"""
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR
    
    if config.lr_scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.lr_scheduler == 'step':
        return StepLR(optimizer, step_size=config.epochs // 3, gamma=0.1)
    elif config.lr_scheduler == 'onecycle':
        return OneCycleLR(optimizer, max_lr=config.lr, total_steps=num_training_steps)
    else:
        return CosineAnnealingLR(optimizer, T_max=config.epochs)


def print_config(config: FullConfig):
    """打印配置"""
    print(f"\n{'='*60}")
    print("配置信息")
    print(f"{'='*60}")
    print(f"\n📌 LoRA: rank={config.lora.rank}, alpha={config.lora.alpha}")
    print(f"📌 Adapter: hidden={config.adapter.hidden_dim}, layers={config.adapter.num_layers}")
    print(f"📌 训练: epochs={config.training.epochs}, lr={config.training.lr}, k_shot={config.training.k_shot}")
    print(f"📌 模型: DINO={config.model.dino_model}, SAM2={config.model.sam2_model}")
    print(f"📁 数据: {config.data_root}")
    print(f"📁 输出: {config.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    config = get_high_quality_config()
    print_config(config)

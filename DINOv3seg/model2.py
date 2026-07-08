"""
model2.py — DINOv3 (LoRA) + FPN + Converse2D 上采分割头

相对 model.py 的变化:
1. FPN decoder 的 4 个上采阶段 (up1~up4) 从 ConvTranspose2d
   换成基于 FFT 反卷积的 Converse2D 上采块 (ConverseUpBlock)，
   对小目标 (antenna) 的细节恢复更友好。
2. 在 smooth 之后、up1 之前加一个 scale=1 的 Converse2D 残差精炼
   (refine)，做频域细节增强；走残差以避免破坏 FPN 融合后的特征。
3. 其余结构 (backbone, LoRA, lateral, smooth, head) 与 model.py 一致。

注意:
- Converse2D 强制 in_channels == out_channels (depthwise)，
  所以每段都拆成 "Converse2D 同通道上采 → 1x1 Conv 降通道" 两步。
- decoder 全程 fp32 (FFT 在 bf16 下精度差)；backbone 仍用 bf16。
- padding_mode 改成 reflect (卫星图边缘没有循环结构)。
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DINOV3_REPO, DINOV3_WEIGHTS, DINOV3_ARCH,
    EMBED_DIM, PATCH_SIZE, FPN_LAYERS, NUM_CLASSES,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
)
from Converse2D import Converse2D

sys.path.insert(0, DINOV3_REPO)


# ============================================================
# 1. 加载 DINOv3 backbone (与 model.py 完全一致)
# ============================================================
def load_dinov3_backbone(device='cuda', dtype=torch.bfloat16):
    """从本地仓库和权重加载 DINOv3 ViT-L/16"""
    print(f"[Backbone] Loading {DINOV3_ARCH} from {DINOV3_REPO}")
    model = torch.hub.load(
        repo_or_dir=DINOV3_REPO,
        model=DINOV3_ARCH,
        source='local',
        pretrained=False,
    )
    print(f"[Backbone] Loading weights from {DINOV3_WEIGHTS}")
    sd = torch.load(DINOV3_WEIGHTS, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing {len(missing)} keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected {len(unexpected)} keys, e.g. {unexpected[:3]}")

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model = model.to(device=device, dtype=dtype)
    print(f"[Backbone] Ready. dtype={dtype}, device={device}")
    return model


# ============================================================
# 2. 给 DINOv3 backbone 打 LoRA (与 model.py 完全一致)
# ============================================================
def apply_lora(backbone: nn.Module):
    """用 peft 给 backbone 的 qkv / proj 插 LoRA 适配器"""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias='none',
        target_modules=LORA_TARGET_MODULES,
        task_type=None,
    )
    try:
        from peft import inject_adapter_in_model
        backbone = inject_adapter_in_model(lora_cfg, backbone, adapter_name='default')
    except ImportError:
        backbone = get_peft_model(backbone, lora_cfg)

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"[LoRA] trainable={trainable:,} / total={total:,}  "
          f"({100.0*trainable/total:.3f}%)")
    return backbone


# ============================================================
# 3. Converse2D 上采块
# ============================================================
class ConverseUpBlock(nn.Module):
    """
    Converse2D (depthwise, scale=2) → 1x1 Conv (降通道) → GN → GELU
    
    用法等价于 ConvTranspose2d(in_ch -> out_ch, stride=2)，
    但用 FFT 反卷积，对纹理 / 细线状目标更友好。
    """
    def __init__(self, in_ch, out_ch, scale=2,
                 kernel_size=5, padding=4, padding_mode='reflect'):
        super().__init__()
        self.up = Converse2D(
            in_channels=in_ch,
            out_channels=in_ch,            # 必须相等 (depthwise)
            kernel_size=kernel_size,
            scale=scale,
            padding=padding,
            padding_mode=padding_mode,
        )
        groups = min(32, out_ch)           # out_ch 都是 32 的倍数
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.up(x)        # (B, in_ch, H*scale, W*scale)
        x = self.proj(x)      # (B, out_ch, H*scale, W*scale)
        return x


# ============================================================
# 4. FPN 解码器 (Converse2D 版)
# ============================================================
class FPNDecoderConverse(nn.Module):
    """
    与原 FPNDecoder 接口一致，但:
      - up1~up4 用 ConverseUpBlock
      - smooth 之后加 scale=1 的 Converse2D 做频域残差精炼
    """
    def __init__(self, in_dim=EMBED_DIM, mid=256, num_classes=NUM_CLASSES,
                 use_refine=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_refine = use_refine
        n_feats = len(FPN_LAYERS)

        # 对每层 token 做 1x1 降维 (1024 -> mid=256)
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_dim, mid, 1) for _ in range(n_feats)
        ])

        # 多层 concat 后的细化
        self.smooth = nn.Sequential(
            nn.Conv2d(mid * n_feats, mid, 3, padding=1),
            nn.GroupNorm(32, mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(32, mid),
            nn.GELU(),
        )

        # scale=1 的 Converse2D 残差精炼 (可选)
        if use_refine:
            self.refine = Converse2D(
                in_channels=mid, out_channels=mid,
                kernel_size=5, scale=1,
                padding=4, padding_mode='reflect',
            )
            self.refine_norm = nn.GroupNorm(32, mid)

        # 4 段 2x 上采，从 H/16 -> H
        self.up1 = ConverseUpBlock(mid,        mid // 2,  scale=2)   # H/16 -> H/8
        self.up2 = ConverseUpBlock(mid // 2,   mid // 4,  scale=2)   # H/8  -> H/4
        self.up3 = ConverseUpBlock(mid // 4,   mid // 4,  scale=2)   # H/4  -> H/2
        self.up4 = ConverseUpBlock(mid // 4,   mid // 8,  scale=2)   # H/2  -> H

        self.head = nn.Conv2d(mid // 8, num_classes, 1)

    def forward(self, features_list, out_hw):
        H, W = out_hw
        H16, W16 = H // PATCH_SIZE, W // PATCH_SIZE

        feats_2d = []
        for i, tokens in enumerate(features_list):
            B, N, C = tokens.shape
            assert N == H16 * W16, f"token count {N} != {H16*W16}"
            x = tokens.transpose(1, 2).reshape(B, C, H16, W16)
            x = self.lateral[i](x)
            feats_2d.append(x)

        x = torch.cat(feats_2d, dim=1)            # (B, mid*4, H/16, W/16)
        x = self.smooth(x)                         # (B, mid,   H/16, W/16)

        # 频域残差精炼
        if self.use_refine:
            r = self.refine(x)
            r = self.refine_norm(r)
            x = x + r

        x = self.up1(x)                            # H/8
        x = self.up2(x)                            # H/4
        x = self.up3(x)                            # H/2
        x = self.up4(x)                            # H
        x = self.head(x)                           # (B, num_classes, H, W)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


# ============================================================
# 5. 整合：Seg 模型 (Converse2D 版)
# ============================================================
class DinoV3SegLoRAConverse(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, device='cuda',
                 dtype=torch.bfloat16, use_refine=True):
        super().__init__()
        backbone = load_dinov3_backbone(device=device, dtype=dtype)
        backbone = apply_lora(backbone)
        self.backbone = backbone
        self.dtype_bb = dtype

        # decoder 全程 fp32 (FFT 数值稳定性)
        self.decoder = FPNDecoderConverse(
            in_dim=EMBED_DIM,
            mid=256,
            num_classes=num_classes,
            use_refine=use_refine,
        ).to(device=device)

        assert hasattr(backbone, 'get_intermediate_layers'), \
            "backbone 没有 get_intermediate_layers 方法"

    def forward(self, x):
        """
        x: (B, 3, H, W) 已 normalize, fp32
        return: (B, num_classes, H, W) fp32
        """
        B, _, H, W = x.shape
        assert H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0, \
            f"输入尺寸 {H}x{W} 必须是 {PATCH_SIZE} 的倍数"

        # ------- backbone forward (bf16) -------
        x_bb = x.to(self.dtype_bb)
        feats = self.backbone.get_intermediate_layers(
            x_bb,
            n=FPN_LAYERS,
            reshape=False,
            norm=True,
        )
        feats = [f.float() for f in feats]

        # ------- decoder (fp32) -------
        logits = self.decoder(feats, out_hw=(H, W))
        return logits


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {device}")

    model = DinoV3SegLoRAConverse(device=device)
    model.eval()

    # 统计 decoder 可训练参数
    dec_trainable = sum(p.numel() for p in model.decoder.parameters()
                        if p.requires_grad)
    print(f"[Decoder] trainable = {dec_trainable:,}")

    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output dtype: {y.dtype}")
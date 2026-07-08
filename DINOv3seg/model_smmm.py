"""
model_smmm.py — DINOv3 + FPN/Converse2D + SMMM(B) 分组融合模型

用途：
- 按“方案 B：高低层分组融合”测试 SMMM 是否能替代原始 concat + smooth 融合。
- 支持 LoRA / noLoRA。
- 支持标准 FPN decoder / Converse2D decoder。

四个模型类：
1. DinoV3SegLoRASMMM
   DINOv3-FPN-SMMM + LoRA

2. DinoV3SegNoLoRASMMM
   DINOv3-FPN-SMMM noLoRA

3. DinoV3SegLoRAConverseSMMM
   DINOv3-FPN-Converse2D-SMMM + LoRA

4. DinoV3SegNoLoRAConverseSMMM
   DINOv3-FPN-Converse2D-SMMM noLoRA

SMMM(B) 融合方式：
    low  = SMMM(f_block5,  f_block11)
    high = SMMM(f_block17, f_block23)
    x    = SMMM(low, high)

其中每个 f 都已经由：
    patch tokens (B,N,C)
    -> reshape (B,C,H/16,W/16)
    -> 1×1 lateral conv, 1024 -> 256
得到。

说明：
- 你上传的 SMMM 原始实现使用 BatchNorm2d。
- 这里默认改为 GroupNorm，以适配你当前 batch size 较小的分割训练。
- 若想严格使用原始 BatchNorm，可以把 ConvGN / DWConvGN 改回 Conv / DWConv。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    EMBED_DIM, PATCH_SIZE, FPN_LAYERS, NUM_CLASSES,
)
from model import load_dinov3_backbone, apply_lora
from model2 import Converse2D, ConverseUpBlock


# ============================================================
# 1. SMMM 模块：使用 GroupNorm 的稳定版本
# ============================================================

def autopad(k, p=None, d=1):
    """Pad to same shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def valid_gn_groups(channels: int, max_groups: int = 8) -> int:
    """选择一个能整除 channels 的 GroupNorm groups。"""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ConvGN(nn.Module):
    """
    Conv + GroupNorm + SiLU。
    这是对上传 SMMM 中 Conv(BatchNorm) 的小 batch 稳定版替换。
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.norm = nn.GroupNorm(valid_gn_groups(c2, gn_groups), c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DWConvGN(ConvGN):
    """Depth-wise convolution + GN + activation."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, gn_groups=8):
        super().__init__(
            c1, c2, k, s,
            g=math.gcd(c1, c2),
            d=d,
            act=act,
            gn_groups=gn_groups,
        )


class MultiScaleExtractor(nn.Module):
    """
    SMMM 的多尺度提取器：
      - 第一阶段：3×3 DWConv 与 5×5 DWConv 并行
      - 第二阶段：对 concat 后特征再次做 3×3 / 5×5 DWConv
      - 1×1 conv 融合回 channels
    """
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.dwconv3_1 = DWConvGN(channels, channels, 3, gn_groups=gn_groups)
        self.dwconv5_1 = DWConvGN(channels, channels, 5, gn_groups=gn_groups)

        self.dwconv3_2 = DWConvGN(2 * channels, channels, 3, gn_groups=gn_groups)
        self.dwconv5_2 = DWConvGN(2 * channels, channels, 5, gn_groups=gn_groups)

        self.fuse_conv = nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B,C,H,W) -> LayerNorm over C
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)

        x3_1 = self.dwconv3_1(x)
        x5_1 = self.dwconv5_1(x)
        cat1 = torch.cat([x3_1, x5_1], dim=1)

        x3_2 = self.dwconv3_2(cat1)
        x5_2 = self.dwconv5_2(cat1)
        cat2 = torch.cat([x3_2, x5_2], dim=1)

        fused = self.fuse_conv(cat2)
        return fused


class MaskLayer(nn.Module):
    """
    SMMM 的结构感知 mask：
      - 1×1 / 3×3 / 5×5 depth-wise conv
      - channel-wise softmax
      - x * mask
    """
    def __init__(self, dim):
        super().__init__()
        self.cg1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim)
        self.cg2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.cg3 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        m1 = self.cg1(x)
        m2 = self.cg2(x)
        m3 = self.cg3(x)
        m = self.softmax(m1 + m2 + m3)
        return x * m


class SMMM(nn.Module):
    """
    Structure-aware Multi-scale Masked feature fusion Module.

    输入：
      x_enc: (B,C,H,W)
      x_dec: (B,C,H,W)

    输出：
      out:   (B,C,H,W)
    """
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.encoder_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.decoder_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.extractor = MultiScaleExtractor(channels, gn_groups=gn_groups)
        self.mask = MaskLayer(channels)

        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(valid_gn_groups(channels, 4), channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_enc, x_dec):
        x1 = self.encoder_proj(x_enc)
        x2 = self.decoder_proj(x_dec)

        f1 = self.extractor(x1)
        f2 = self.extractor(x2)

        merged = self.mask(f1) + self.mask(f2)

        out = self.dilated_conv(merged)
        out = self.norm(out)
        out = self.out_conv(out)
        return out


class SMMMGroupFusionB(nn.Module):
    """
    方案 B：高低层分组融合。

    输入 feats_2d:
      feats_2d[0] = Block 5
      feats_2d[1] = Block 11
      feats_2d[2] = Block 17
      feats_2d[3] = Block 23

    融合：
      low  = SMMM(Block5,  Block11)
      high = SMMM(Block17, Block23)
      out  = SMMM(low, high)
    """
    def __init__(self, channels=256, gn_groups=8, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.low_fuse = SMMM(channels, gn_groups=gn_groups)
        self.high_fuse = SMMM(channels, gn_groups=gn_groups)
        self.cross_fuse = SMMM(channels, gn_groups=gn_groups)

        # 轻量输出平滑，稳定训练
        self.out_norm = nn.GroupNorm(valid_gn_groups(channels, 8), channels)
        self.out_act = nn.GELU()

    def forward(self, feats_2d):
        assert len(feats_2d) == 4, f"SMMMGroupFusionB 需要 4 层特征，当前 len={len(feats_2d)}"

        low = self.low_fuse(feats_2d[0], feats_2d[1])
        high = self.high_fuse(feats_2d[2], feats_2d[3])
        x = self.cross_fuse(low, high)

        if self.use_residual:
            # 给一个简单残差基线，避免 SMMM 初期训练不稳定。
            # 这里用四层特征均值作为 residual，而不是 concat。
            residual = torch.stack(feats_2d, dim=0).mean(dim=0)
            x = x + residual

        x = self.out_act(self.out_norm(x))
        return x


# ============================================================
# 2. 标准 FPN 上采样 decoder + SMMM(B) 融合
# ============================================================

class FPNDecoderSMMM(nn.Module):
    """
    替代原始 FPNDecoder 中的 concat + smooth：
      原始：cat([f5,f11,f17,f23]) -> smooth
      新版：SMMM(f5,f11), SMMM(f17,f23), SMMM(low,high)

    后续仍使用 ConvTranspose2d 四级上采样。
    """
    def __init__(self, in_dim=EMBED_DIM, mid=256, num_classes=NUM_CLASSES,
                 gn_groups=8, fusion_residual=True):
        super().__init__()
        self.num_classes = num_classes
        n_feats = len(FPN_LAYERS)

        self.lateral = nn.ModuleList([
            nn.Conv2d(in_dim, mid, 1) for _ in range(n_feats)
        ])

        self.smmm_fusion = SMMMGroupFusionB(
            channels=mid,
            gn_groups=gn_groups,
            use_residual=fusion_residual,
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(mid, mid // 2, 4, stride=2, padding=1),
            nn.GroupNorm(valid_gn_groups(mid // 2, 16), mid // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid // 2, mid // 4, 4, stride=2, padding=1),
            nn.GroupNorm(valid_gn_groups(mid // 4, 8), mid // 4),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(mid // 4, mid // 4, 4, stride=2, padding=1),
            nn.GroupNorm(valid_gn_groups(mid // 4, 8), mid // 4),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(mid // 4, mid // 8, 4, stride=2, padding=1),
            nn.GroupNorm(valid_gn_groups(mid // 8, 8), mid // 8),
            nn.GELU(),
        )
        self.head = nn.Conv2d(mid // 8, num_classes, 1)

    def _tokens_to_lateral_feats(self, features_list, out_hw):
        H, W = out_hw
        H16, W16 = H // PATCH_SIZE, W // PATCH_SIZE

        feats_2d = []
        for i, tokens in enumerate(features_list):
            B, N, C = tokens.shape
            assert N == H16 * W16, f"token count {N} != {H16 * W16}"
            x = tokens.transpose(1, 2).reshape(B, C, H16, W16)
            x = self.lateral[i](x)
            feats_2d.append(x)
        return feats_2d

    def forward(self, features_list, out_hw):
        H, W = out_hw
        feats_2d = self._tokens_to_lateral_feats(features_list, out_hw)

        x = self.smmm_fusion(feats_2d)     # (B,256,H/16,W/16)

        x = self.up1(x)                    # H/8
        x = self.up2(x)                    # H/4
        x = self.up3(x)                    # H/2
        x = self.up4(x)                    # H
        x = self.head(x)                   # (B,num_classes,H,W)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


# ============================================================
# 3. Converse2D 上采样 decoder + SMMM(B) 融合
# ============================================================

class FPNDecoderConverseSMMM(nn.Module):
    """
    SMMM(B) 融合 + Converse2D decoder：
      - 四层 DINO token -> lateral conv
      - SMMM(B) 分组融合
      - optional Converse2D(scale=1) residual refine
      - ConverseUpBlock ×4
    """
    def __init__(self, in_dim=EMBED_DIM, mid=256, num_classes=NUM_CLASSES,
                 use_refine=True, gn_groups=8, fusion_residual=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_refine = use_refine
        n_feats = len(FPN_LAYERS)

        self.lateral = nn.ModuleList([
            nn.Conv2d(in_dim, mid, 1) for _ in range(n_feats)
        ])

        self.smmm_fusion = SMMMGroupFusionB(
            channels=mid,
            gn_groups=gn_groups,
            use_residual=fusion_residual,
        )

        if use_refine:
            self.refine = Converse2D(
                in_channels=mid, out_channels=mid,
                kernel_size=5, scale=1,
                padding=4, padding_mode="reflect",
            )
            self.refine_norm = nn.GroupNorm(valid_gn_groups(mid, 32), mid)

        self.up1 = ConverseUpBlock(mid,        mid // 2, scale=2)  # H/16 -> H/8
        self.up2 = ConverseUpBlock(mid // 2,   mid // 4, scale=2)  # H/8  -> H/4
        self.up3 = ConverseUpBlock(mid // 4,   mid // 4, scale=2)  # H/4  -> H/2
        self.up4 = ConverseUpBlock(mid // 4,   mid // 8, scale=2)  # H/2  -> H

        self.head = nn.Conv2d(mid // 8, num_classes, 1)

    def _tokens_to_lateral_feats(self, features_list, out_hw):
        H, W = out_hw
        H16, W16 = H // PATCH_SIZE, W // PATCH_SIZE

        feats_2d = []
        for i, tokens in enumerate(features_list):
            B, N, C = tokens.shape
            assert N == H16 * W16, f"token count {N} != {H16 * W16}"
            x = tokens.transpose(1, 2).reshape(B, C, H16, W16)
            x = self.lateral[i](x)
            feats_2d.append(x)
        return feats_2d

    def forward(self, features_list, out_hw):
        H, W = out_hw
        feats_2d = self._tokens_to_lateral_feats(features_list, out_hw)

        x = self.smmm_fusion(feats_2d)     # (B,256,H/16,W/16)

        if self.use_refine:
            r = self.refine(x)
            r = self.refine_norm(r)
            x = x + r

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.head(x)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


# ============================================================
# 4. Segmentation wrapper：LoRA / noLoRA
# ============================================================

class _DinoV3SegBaseSMMM(nn.Module):
    def _forward_backbone(self, x):
        B, _, H, W = x.shape
        assert H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0, \
            f"输入尺寸 {H}x{W} 必须是 {PATCH_SIZE} 的倍数"

        x_bb = x.to(self.dtype_bb)
        feats = self.backbone.get_intermediate_layers(
            x_bb,
            n=FPN_LAYERS,
            reshape=False,
            norm=True,
        )
        feats = [f.float() for f in feats]
        return feats, (H, W)

    def forward(self, x):
        feats, out_hw = self._forward_backbone(x)
        return self.decoder(feats, out_hw=out_hw)


class DinoV3SegLoRASMMM(_DinoV3SegBaseSMMM):
    """DINOv3-FPN-SMMM + LoRA。"""
    def __init__(self, num_classes=NUM_CLASSES, device="cuda",
                 dtype=torch.bfloat16, gn_groups=8, fusion_residual=True):
        super().__init__()
        backbone = load_dinov3_backbone(device=device, dtype=dtype)
        backbone = apply_lora(backbone)
        self.backbone = backbone
        self.dtype_bb = dtype
        self.decoder = FPNDecoderSMMM(
            in_dim=EMBED_DIM,
            mid=256,
            num_classes=num_classes,
            gn_groups=gn_groups,
            fusion_residual=fusion_residual,
        ).to(device=device)


class DinoV3SegNoLoRASMMM(_DinoV3SegBaseSMMM):
    """DINOv3-FPN-SMMM noLoRA。"""
    def __init__(self, num_classes=NUM_CLASSES, device="cuda",
                 dtype=torch.bfloat16, gn_groups=8, fusion_residual=True):
        super().__init__()
        self.backbone = load_dinov3_backbone(device=device, dtype=dtype)
        self.dtype_bb = dtype
        self.decoder = FPNDecoderSMMM(
            in_dim=EMBED_DIM,
            mid=256,
            num_classes=num_classes,
            gn_groups=gn_groups,
            fusion_residual=fusion_residual,
        ).to(device=device)


class DinoV3SegLoRAConverseSMMM(_DinoV3SegBaseSMMM):
    """DINOv3-FPN-Converse2D-SMMM + LoRA。"""
    def __init__(self, num_classes=NUM_CLASSES, device="cuda",
                 dtype=torch.bfloat16, use_refine=True,
                 gn_groups=8, fusion_residual=True):
        super().__init__()
        backbone = load_dinov3_backbone(device=device, dtype=dtype)
        backbone = apply_lora(backbone)
        self.backbone = backbone
        self.dtype_bb = dtype
        self.decoder = FPNDecoderConverseSMMM(
            in_dim=EMBED_DIM,
            mid=256,
            num_classes=num_classes,
            use_refine=use_refine,
            gn_groups=gn_groups,
            fusion_residual=fusion_residual,
        ).to(device=device)


class DinoV3SegNoLoRAConverseSMMM(_DinoV3SegBaseSMMM):
    """DINOv3-FPN-Converse2D-SMMM noLoRA。"""
    def __init__(self, num_classes=NUM_CLASSES, device="cuda",
                 dtype=torch.bfloat16, use_refine=True,
                 gn_groups=8, fusion_residual=True):
        super().__init__()
        self.backbone = load_dinov3_backbone(device=device, dtype=dtype)
        self.dtype_bb = dtype
        self.decoder = FPNDecoderConverseSMMM(
            in_dim=EMBED_DIM,
            mid=256,
            num_classes=num_classes,
            use_refine=use_refine,
            gn_groups=gn_groups,
            fusion_residual=fusion_residual,
        ).to(device=device)


# ============================================================
# 5. 自测
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 默认自测 noLoRA，避免要求 PEFT LoRA 环境。
    model = DinoV3SegNoLoRASMMM(device=device, dtype=dtype)
    model.eval()

    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")

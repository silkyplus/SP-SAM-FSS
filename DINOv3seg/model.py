"""
model.py — DINOv3 (LoRA) + FPN 分割头

设计:
- Backbone: DINOv3 ViT-L/16, 权重 = sat493m
- 用 get_intermediate_layers 一次取多层 patch tokens
- 4 层特征 -> FPN decoder -> 4 类 logits
- LoRA 插在 attention 的 qkv 和 proj 上（其他参数冻结）
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

sys.path.insert(0, DINOV3_REPO)


# ============================================================
# 1. 加载 DINOv3 backbone
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

    # 冻结全部
    for p in model.parameters():
        p.requires_grad = False

    model.eval()  # 让 LayerNorm 等用 eval 模式（但 LoRA 会重新激活梯度）
    model = model.to(device=device, dtype=dtype)
    print(f"[Backbone] Ready. dtype={dtype}, device={device}")
    return model


# ============================================================
# 2. 给 DINOv3 backbone 打 LoRA
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
        # 非 PEFT 认识的任务类型，设 None 表示纯模块替换
        task_type=None,
    )
    # 注意：peft 的 get_peft_model 会把原模型封一层
    # 我们希望 forward 仍然能调 backbone 的 get_intermediate_layers，
    # 所以用 inject_adapter_in_model（不包 wrapper）
    try:
        from peft import inject_adapter_in_model
        backbone = inject_adapter_in_model(lora_cfg, backbone, adapter_name='default')
    except ImportError:
        # 老版本 peft 没有 inject_adapter_in_model，退回 get_peft_model
        backbone = get_peft_model(backbone, lora_cfg)

    # 统计可训练参数
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"[LoRA] trainable={trainable:,} / total={total:,}  "
          f"({100.0*trainable/total:.3f}%)")

    return backbone


# ============================================================
# 3. FPN 解码器
# ============================================================
class FPNDecoder(nn.Module):
    """
    输入 4 层的 (B, N, C) tokens（N = (H/16)*(W/16)）
    reshape 成 (B, C, H/16, W/16)，做 1x1 conv 降维后多层融合，
    再上采样到 (B, num_classes, H, W)
    """
    def __init__(self, in_dim=EMBED_DIM, mid=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        n_feats = len(FPN_LAYERS)
        # 对每层 token 做 1x1 降维
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_dim, mid, 1) for _ in range(n_feats)
        ])
        # fusion 后的细化
        self.smooth = nn.Sequential(
            nn.Conv2d(mid * n_feats, mid, 3, padding=1),
            nn.GroupNorm(32, mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(32, mid),
            nn.GELU(),
        )
        # 上采样到原分辨率（patch=16，所以 4x 上采两次 = 16x）
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(mid, mid // 2, 4, stride=2, padding=1),
            nn.GroupNorm(16, mid // 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid // 2, mid // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, mid // 4),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(mid // 4, mid // 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, mid // 4),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(mid // 4, mid // 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, mid // 8),
            nn.GELU(),
        )
        self.head = nn.Conv2d(mid // 8, num_classes, 1)

    def forward(self, features_list, out_hw):
        """
        features_list: list of (B, N, C) tensors
        out_hw: (H, W) 原图尺寸
        """
        H, W = out_hw
        H16, W16 = H // PATCH_SIZE, W // PATCH_SIZE

        feats_2d = []
        for i, tokens in enumerate(features_list):
            # tokens: (B, N, C)。如果 DINOv3 含 cls/reg token，get_intermediate_layers
            # 的 norm=True 已经处理，只返回 patch tokens（我们在调用时确保）
            B, N, C = tokens.shape
            assert N == H16 * W16, f"token count {N} != {H16*W16}"
            x = tokens.transpose(1, 2).reshape(B, C, H16, W16)
            x = self.lateral[i](x)
            feats_2d.append(x)

        x = torch.cat(feats_2d, dim=1)           # (B, mid*4, H16, W16)
        x = self.smooth(x)                        # (B, mid, H16, W16)
        x = self.up1(x)                           # 2x  -> H/8
        x = self.up2(x)                           # 2x  -> H/4
        x = self.up3(x)                           # 2x  -> H/2
        x = self.up4(x)                           # 2x  -> H
        x = self.head(x)                          # (B, num_classes, H, W)
        # 保险：确保尺寸一致
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


# ============================================================
# 4. 整合：Seg 模型
# ============================================================
class DinoV3SegLoRA(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        backbone = load_dinov3_backbone(device=device, dtype=dtype)
        backbone = apply_lora(backbone)
        self.backbone = backbone
        self.dtype_bb = dtype

        # FPN 用 fp32（head 参数量小，fp32 更稳）
        self.decoder = FPNDecoder(
            in_dim=EMBED_DIM,
            mid=256,
            num_classes=num_classes,
        ).to(device=device)

        # 检查 DINOv3 是否暴露 get_intermediate_layers
        assert hasattr(backbone, 'get_intermediate_layers'), \
            "backbone 没有 get_intermediate_layers 方法"

    def forward(self, x):
        """
        x: (B, 3, H, W) 已 normalize，fp32
        return: (B, num_classes, H, W) fp32
        """
        B, _, H, W = x.shape
        # DINOv3 ViT 通常要求 H,W 是 patch_size 的倍数
        assert H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0, \
            f"输入尺寸 {H}x{W} 必须是 {PATCH_SIZE} 的倍数"

        # ------- backbone forward (bf16) -------
        # get_intermediate_layers(x, n, norm=True) 返回 tuple of (B, N, C)
        # n 可以是 int（最后 n 层）或 List[int]（指定层 index）
        x_bb = x.to(self.dtype_bb)
        feats = self.backbone.get_intermediate_layers(
            x_bb,
            n=FPN_LAYERS,
            reshape=False,   # 保持 (B, N, C)
            norm=True,
        )
        # feats: tuple of tensors，长度 = len(FPN_LAYERS)
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

    model = DinoV3SegLoRA(device=device)
    model.eval()

    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output dtype: {y.dtype}")

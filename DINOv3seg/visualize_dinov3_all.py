"""
python visualize_dinov3_all.py --model fpn_nolora --ckpt "D:/pycharm_projects/NOTRAING/NOTRAING/DINOv3seg/runs/run_nolora_fpn/best.pth" --images "D:/data/final_dataset/final_dataset/images/val/img_resize_423.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_430.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_506.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_609.png" --out "D:/data/final_dataset/feature_vis"
python visualize_dinov3_all_win.py --model converse_nolora --ckpt "D:/pycharm_projects/NOTRAING/NOTRAING/DINOv3seg/runs/run_nolora_converse_20260512_234002/best.pth" --images "D:/data/final_dataset/final_dataset/images/val/img_resize_423.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_430.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_506.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_609.png" --out "D:/data/final_dataset/feature_vis/converse_nolora"
python visualize_dinov3_all.py --model converse_nolora --ckpt "D:/pycharm_projects/NOTRAING/NOTRAING/DINOv3seg/runs/run_nolora_converse_20260512_234002/best.pth" --images "D:/data/final_dataset/final_dataset/images/val/img_resize_423.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_430.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_506.png" "D:/data/final_dataset/final_dataset/images/val/img_resize_609.png" --out "D:/data/final_dataset/feature_vis/converse_nolora"
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from config import (
    PATCH_SIZE, FPN_LAYERS,
    NUM_CLASSES, CLASS_NAMES,
    IMAGENET_MEAN, IMAGENET_STD,
    RGB_TO_CLASS,
)
from model import DinoV3SegLoRA
from model2 import DinoV3SegLoRAConverse
from model_nolora import DinoV3SegNoLoRA, DinoV3SegNoLoRAConverse


# ==================== 全局配置 ====================

DEFAULT_IMAGE = r"D:\data\final_dataset\final_dataset\images\val\img_resize_423.png"

CLASS_COLORS_RGB = {v: k for k, v in RGB_TO_CLASS.items()}
STAGE_NAMES = [f"stage{i+1}_block{b}" for i, b in enumerate(FPN_LAYERS)]

MEAN_A = np.array(IMAGENET_MEAN, dtype=np.float32)
STD_A  = np.array(IMAGENET_STD,  dtype=np.float32)


# ==================== 模型构建与权重加载 ====================

def normalize_model_name(name: str) -> str:
    aliases = {
        "fpn": "fpn_lora",
        "lora_fpn": "fpn_lora",
        "dino_fpn": "fpn_lora",
        "converse": "converse_lora",
        "conv2d": "converse_lora",
        "improved": "converse_lora",
        "lora_converse": "converse_lora",
        "dino_converse": "converse_lora",
        "nolora_fpn": "fpn_nolora",
        "no_lora_fpn": "fpn_nolora",
        "fpn_no_lora": "fpn_nolora",
        "nolora_converse": "converse_nolora",
        "no_lora_converse": "converse_nolora",
        "converse_no_lora": "converse_nolora",
    }
    return aliases.get(name, name)


def build_model(model_name: str, device: str, dtype: torch.dtype):
    name = normalize_model_name(model_name)

    if name == "fpn_lora":
        model = DinoV3SegLoRA(num_classes=NUM_CLASSES, device=device, dtype=dtype)
        label = "DINOv3-FPN + LoRA"
    elif name == "converse_lora":
        model = DinoV3SegLoRAConverse(num_classes=NUM_CLASSES, device=device, dtype=dtype, use_refine=True)
        label = "DINOv3-FPN-Converse2D + LoRA"
    elif name == "fpn_nolora":
        model = DinoV3SegNoLoRA(num_classes=NUM_CLASSES, device=device, dtype=dtype)
        label = "DINOv3-FPN noLoRA"
    elif name == "converse_nolora":
        model = DinoV3SegNoLoRAConverse(num_classes=NUM_CLASSES, device=device, dtype=dtype, use_refine=True)
        label = "DINOv3-FPN-Converse2D noLoRA"
    else:
        raise ValueError(
            f"未知模型: {model_name}. 可选：fpn_lora / converse_lora / fpn_nolora / converse_nolora"
        )

    return model, label, name


def extract_state_dict(obj):
    """兼容 {'state_dict': ...} / {'model': ...} / {'model_state_dict': ...} / 纯 state_dict。"""
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def load_compatible_weights(model, ckpt_path: str, device: str):
    """
    兼容：
      - 你的 LoRA ckpt：通常只保存 LoRA + decoder 权重；
      - noLoRA 训练 ckpt：通常只保存 decoder 权重；
      - 完整 state_dict。
    只加载 key 存在且 shape 一致的权重。
    """
    raw = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(raw)
    model_state = model.state_dict()

    compatible = {}
    ignored = []
    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            compatible[k] = v
        else:
            ignored.append(k)

    missing, unexpected = model.load_state_dict(compatible, strict=False)

    print(f"[ckpt] {ckpt_path}")
    print(f"  compatible keys : {len(compatible)}")
    print(f"  ignored keys    : {len(ignored)}")
    print(f"  missing keys    : {len(missing)}")
    print(f"  unexpected keys : {len(unexpected)}")
    if ignored:
        print(f"  ignored examples: {ignored[:5]}")

    meta = {}
    if isinstance(raw, dict):
        for k in ["epoch", "miou_fg", "miou_all", "model_type"]:
            if k in raw:
                meta[k] = raw[k]
    return model, meta


# ==================== 图像预处理与颜色辅助 ====================

def preprocess_image(img_path: str, size: int):
    """读图 → resize → normalize → (1,3,H,W) fp32 tensor，同时返回 RGB uint8。"""
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"图像不存在或无法读取: {img_path}")

    img_bgr = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f = (img_rgb.astype(np.float32) / 255.0 - MEAN_A) / STD_A
    t = torch.from_numpy(img_f.transpose(2, 0, 1)).float().unsqueeze(0)
    return t, img_rgb


def render_pred_as_gt_style(pred_label, size_hw):
    """把预测渲染成和 GT mask 一致的纯色格式：黑底 + 实色类别。"""
    H, W = size_hw
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS_RGB.items():
        if cls_id == 0:
            continue
        out[pred_label == cls_id] = rgb
    return out


def overlay_seg_on_image(img_rgb, pred_label, alpha=0.45):
    """原图上叠加预测类别颜色。"""
    overlay = img_rgb.copy().astype(np.float32)
    for cls_id, rgb in CLASS_COLORS_RGB.items():
        if cls_id == 0:
            continue
        mask = pred_label == cls_id
        overlay[mask] = overlay[mask] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha
    return overlay.astype(np.uint8)


def blend(img_rgb, heatmap, alpha=0.55):
    """原图 + 热力图混合。"""
    return (img_rgb.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha).astype(np.uint8)


def legend_patches():
    return [
        mpatches.Patch(
            color=tuple(c / 255 for c in CLASS_COLORS_RGB[i]),
            label=CLASS_NAMES[i],
        )
        for i in range(len(CLASS_NAMES))
    ]


# ==================== 特征与 GradCAM ====================

def tokens_to_spatial(tokens, img_H, img_W, patch=PATCH_SIZE):
    """
    tokens: (B, N, C)
    取末尾 img_H/patch * img_W/patch 个 patch tokens，reshape 为 (B,C,h,w)。
    """
    B, N, C = tokens.shape
    expected = (img_H // patch) * (img_W // patch)
    patch_tokens = tokens[:, -expected:, :]
    grid_h, grid_w = img_H // patch, img_W // patch
    return patch_tokens.transpose(1, 2).reshape(B, C, grid_h, grid_w)


def feature_to_heatmap(feat_chw, target_size):
    """(C,h,w) tensor → 通道平均 → ReLU → norm → JET RGB。"""
    act = feat_chw.float().mean(dim=0).detach().cpu().numpy()
    act = np.maximum(act, 0)
    if act.max() > act.min():
        act = (act - act.min()) / (act.max() - act.min() + 1e-8)

    H, W = target_size
    rsz = cv2.resize(act, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_bgr = cv2.applyColorMap((rsz * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)


class MultiStageHook:
    """
    在 DINOv3 的 FPN_LAYERS 对应 block 上注册 forward/backward hook。
    兼容：
      - 原始 DINOv3 backbone
      - PEFT LoRA 注入后的 backbone
      - noLoRA backbone
    """
    def __init__(self, seg_model, stage_block_indices=FPN_LAYERS):
        self.stage_block_indices = stage_block_indices
        self.activations = {}
        self._refs = {}
        self.gradients = {}
        self._hooks = []

        blocks = self._find_blocks(seg_model.backbone)
        assert blocks is not None and len(blocks) > max(stage_block_indices), (
            f"找不到 transformer blocks，或 block 数量不足，需要 > {max(stage_block_indices)}"
        )

        for stage_idx, block_idx in enumerate(stage_block_indices):
            self._register_block(blocks[block_idx], stage_idx)

    @staticmethod
    def _find_blocks(backbone):
        # 原始 DINOv3
        b = getattr(backbone, "blocks", None)
        if b is not None:
            return b

        # PEFT 可能包在 base_model / base_model.model 下
        bm = getattr(backbone, "base_model", None)
        if bm is not None:
            b = getattr(bm, "blocks", None)
            if b is not None:
                return b

            inner = getattr(bm, "model", None)
            if inner is not None:
                b = getattr(inner, "blocks", None)
                if b is not None:
                    return b

        # 其他包装层兜底
        inner = getattr(backbone, "model", None)
        if inner is not None:
            b = getattr(inner, "blocks", None)
            if b is not None:
                return b

        return None

    def _register_block(self, block_module, stage_idx):
        def fwd_hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            self._refs[stage_idx] = t
            self.activations[stage_idx] = t.detach()
            if t.requires_grad:
                t.retain_grad()

        def bwd_hook(module, gin, gout):
            g = gout[0] if gout and gout[0] is not None else None
            if g is not None:
                self.gradients[stage_idx] = g.detach()

        self._hooks.append(block_module.register_forward_hook(fwd_hook))
        self._hooks.append(block_module.register_full_backward_hook(bwd_hook))

    def clear_grads(self):
        self.gradients.clear()

    def fallback_collect_grads_from_refs(self):
        for k, ref in self._refs.items():
            if k not in self.gradients:
                g = getattr(ref, "grad", None)
                if g is not None:
                    self.gradients[k] = g.detach()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


@torch.no_grad()
def forward_and_extract_features(model, hook, img_tensor, device):
    model.eval()
    out = model(img_tensor.to(device))
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
    return out, pred


def compute_gradcam_for_stage(acts_2d, grads_2d, target_size):
    """
    acts_2d, grads_2d: (C,h,w)
    return: RGB heatmap, or None
    """
    weights = grads_2d.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * acts_2d).sum(dim=0)).detach().cpu().numpy()

    if cam.max() <= cam.min():
        return None

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    H, W = target_size
    rsz = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_bgr = cv2.applyColorMap((rsz * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)


def compute_all_gradcams(model, hook, img_tensor, device, num_classes, img_H, img_W):
    """
    每个类别 backward 一次，一次 backward 获取所有 stage 的梯度。
    对 noLoRA / frozen backbone：
      - 输入必须 requires_grad=True，否则冻结 backbone 的中间激活不会进入 autograd 图。
    """
    gradcams = {s: {c: None for c in range(num_classes)} for s in range(len(FPN_LAYERS))}
    target_size = (img_H, img_W)
    model.eval()

    for c in range(num_classes):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        hook.clear_grads()

        with torch.enable_grad():
            inp = img_tensor.to(device).detach().requires_grad_(True)
            out = model(inp)
            score = out[:, c].mean()
            score.backward()

        hook.fallback_collect_grads_from_refs()

        for s in range(len(FPN_LAYERS)):
            act_tokens = hook.activations.get(s)
            grad_tokens = hook.gradients.get(s)
            if act_tokens is None or grad_tokens is None:
                continue

            acts_2d = tokens_to_spatial(act_tokens.float(), img_H, img_W).squeeze(0)
            grads_2d = tokens_to_spatial(grad_tokens.float(), img_H, img_W).squeeze(0)
            cam = compute_gradcam_for_stage(acts_2d, grads_2d, target_size)
            gradcams[s][c] = cam

    return gradcams


# ==================== 保存 ====================

def _save_single(img_arr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(img_arr.shape[1] / 100, img_arr.shape[0] / 100))
    ax.imshow(img_arr)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_comparison_grid(img_rgb, pred_mask, pred_overlay, heatmaps, gradcams,
                         num_classes, out_path: Path, model_label="", title_suffix=""):
    n_stages = len(FPN_LAYERS)
    has_gradcam = gradcams is not None and any(
        any(cam is not None for cam in gradcams[s].values()) for s in gradcams
    )

    n_rows = 2 + (num_classes if has_gradcam else 0)
    n_cols = max(4, n_stages)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r in range(n_rows):
        for col in range(n_cols):
            axes[r, col].axis("off")

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Input", fontsize=11)

    axes[0, 1].imshow(pred_mask)
    axes[0, 1].set_title("Pred Mask", fontsize=11)

    axes[0, 2].imshow(pred_overlay)
    axes[0, 2].set_title("Overlay", fontsize=11)
    axes[0, 2].legend(handles=legend_patches(), fontsize=7, loc="upper right", framealpha=0.8)

    for s in range(n_stages):
        if s in heatmaps:
            axes[1, s].imshow(heatmaps[s])
            axes[1, s].set_title(f"Feature {STAGE_NAMES[s]}", fontsize=10)

    if has_gradcam:
        for c in range(num_classes):
            row = 2 + c
            for s in range(n_stages):
                cam = gradcams[s].get(c)
                if cam is not None:
                    axes[row, s].imshow(cam)
                    axes[row, s].set_title(
                        f"GradCAM {CLASS_NAMES[c]} @ {STAGE_NAMES[s]}",
                        fontsize=8,
                    )

    title = model_label
    if title_suffix:
        title += f"  {title_suffix}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def visualize_one_image(model, hook, img_path: Path, out_dir: Path, args,
                        device: str, model_label: str, ckpt_meta: dict):
    img_stem = img_path.stem

    print(f"\n[Image] {img_path}")
    img_tensor, img_rgb = preprocess_image(str(img_path), args.size)

    print("  [1] forward + feature extraction")
    _, pred_label = forward_and_extract_features(model, hook, img_tensor, device)

    pred_mask = render_pred_as_gt_style(pred_label, (args.size, args.size))
    pred_overlay = overlay_seg_on_image(img_rgb, pred_label, alpha=args.overlay_alpha)

    H = W = args.size
    heatmaps = {}
    for s in range(len(FPN_LAYERS)):
        act_tokens = hook.activations.get(s)
        if act_tokens is None:
            print(f"    [warn] no activation for {STAGE_NAMES[s]}")
            continue
        feat_2d = tokens_to_spatial(act_tokens.float(), H, W).squeeze(0)
        hm = feature_to_heatmap(feat_2d, (H, W))
        heatmaps[s] = blend(img_rgb, hm, alpha=args.heat_alpha)

    gradcams = None
    if not args.no_gradcam:
        print(f"  [2] GradCAM: {len(FPN_LAYERS)} stages × {NUM_CLASSES} classes")
        raw_gradcams = compute_all_gradcams(
            model, hook, img_tensor, device,
            num_classes=NUM_CLASSES,
            img_H=H,
            img_W=W,
        )
        gradcams = {
            s: {
                c: (blend(img_rgb, raw_gradcams[s][c], alpha=args.heat_alpha)
                    if raw_gradcams[s][c] is not None else None)
                for c in raw_gradcams[s]
            }
            for s in raw_gradcams
        }

    print("  [3] save files")
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_single(img_rgb, out_dir / f"{img_stem}_00_original.png")
    _save_single(pred_mask, out_dir / f"{img_stem}_01_pred_mask.png")
    _save_single(pred_overlay, out_dir / f"{img_stem}_02_pred_overlay.png")

    for s, hm in heatmaps.items():
        _save_single(hm, out_dir / f"{img_stem}_10_feature_{STAGE_NAMES[s]}.png")

    if gradcams is not None:
        for s in gradcams:
            for c in range(NUM_CLASSES):
                cam = gradcams[s].get(c)
                if cam is None:
                    continue
                _save_single(cam, out_dir / f"{img_stem}_20_gradcam_{STAGE_NAMES[s]}_{CLASS_NAMES[c]}.png")

    meta_tag = ""
    if ckpt_meta:
        parts = []
        if "epoch" in ckpt_meta:
            parts.append(f"epoch={ckpt_meta['epoch']}")
        if "miou_fg" in ckpt_meta:
            try:
                parts.append(f"mIoU_fg={float(ckpt_meta['miou_fg']):.4f}")
            except Exception:
                parts.append(f"mIoU_fg={ckpt_meta['miou_fg']}")
        meta_tag = "(" + ", ".join(parts) + ")" if parts else ""

    save_comparison_grid(
        img_rgb=img_rgb,
        pred_mask=pred_mask,
        pred_overlay=pred_overlay,
        heatmaps=heatmaps,
        gradcams=gradcams,
        num_classes=NUM_CLASSES,
        out_path=out_dir / f"{img_stem}_99_overview.png",
        model_label=model_label,
        title_suffix=meta_tag,
    )

    n_files = len(list(out_dir.glob(f"{img_stem}_*")))
    print(f"  ✓ saved {n_files} files -> {out_dir}")


# ==================== CLI ====================

def collect_images(args):
    images = []
    if args.image:
        images.append(Path(args.image))
    if args.images:
        images.extend(Path(p) for p in args.images)

    if not images:
        images = [Path(DEFAULT_IMAGE)]

    # 去重但保留顺序
    seen = set()
    uniq = []
    for p in images:
        p = p.resolve()
        if str(p).lower() not in seen:
            uniq.append(p)
            seen.add(str(p).lower())

    for p in uniq:
        if not p.exists():
            raise FileNotFoundError(f"指定图像不存在: {p}")

    return uniq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "fpn_lora", "converse_lora", "fpn_nolora", "converse_nolora",
            "fpn", "converse", "conv2d", "improved",
            "nolora_fpn", "nolora_converse",
            "no_lora_fpn", "no_lora_converse",
        ],
        help="选择模型类型",
    )
    parser.add_argument("--ckpt", required=True, help="best.pth / last.pth 路径")
    parser.add_argument("--image", default=None, help="单张图像路径")
    parser.add_argument("--images", nargs="*", default=None, help="多张图像路径")
    parser.add_argument("--out", default=None, help="输出根目录；默认 ckpt 同级 vis_<model>/")
    parser.add_argument("--size", type=int, default=512, help="输入尺寸，必须是 PATCH_SIZE 的倍数")
    parser.add_argument("--no_gradcam", action="store_true", help="只保存预测和 feature map，不计算 GradCAM")
    parser.add_argument("--fp32", action="store_true", help="backbone 使用 fp32；默认 cuda 下使用 bf16")
    parser.add_argument("--heat_alpha", type=float, default=0.55, help="heatmap 与原图混合系数")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="预测 mask overlay 透明度")
    args = parser.parse_args()

    assert args.size % PATCH_SIZE == 0, f"size={args.size} 必须是 {PATCH_SIZE} 的倍数"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.fp32 else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model_key = normalize_model_name(args.model)

    images = collect_images(args)

    if args.out is None:
        base_out = Path(args.ckpt).parent / f"vis_{model_key}"
    else:
        base_out = Path(args.out)

    print(f"device     = {device}")
    print(f"dtype      = {dtype}")
    print(f"model      = {model_key}")
    print(f"ckpt       = {args.ckpt}")
    print(f"size       = {args.size}")
    print(f"FPN layers = {FPN_LAYERS} -> {STAGE_NAMES}")
    print(f"images     = {len(images)}")
    print(f"out        = {base_out}")

    print("\n[0] loading model ...")
    model, model_label, model_key = build_model(model_key, device=device, dtype=dtype)
    model, ckpt_meta = load_compatible_weights(model, args.ckpt, device=device)
    model.eval()

    hook = MultiStageHook(model, stage_block_indices=FPN_LAYERS)

    try:
        for img_path in images:
            if len(images) == 1:
                out_dir = base_out
            else:
                out_dir = base_out / img_path.stem
            visualize_one_image(
                model=model,
                hook=hook,
                img_path=img_path,
                out_dir=out_dir,
                args=args,
                device=device,
                model_label=model_label,
                ckpt_meta=ckpt_meta,
            )
    finally:
        hook.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 保存本次配置
    base_out.mkdir(parents=True, exist_ok=True)
    with open(base_out / "vis_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_key,
            "ckpt": str(args.ckpt),
            "images": [str(p) for p in images],
            "size": args.size,
            "no_gradcam": bool(args.no_gradcam),
            "fpn_layers": FPN_LAYERS,
            "stage_names": STAGE_NAMES,
            "dtype": str(dtype),
        }, f, indent=2, ensure_ascii=False)

    print("\n全部完成。")


if __name__ == "__main__":
    main()

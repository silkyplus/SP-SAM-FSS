"""
eval2.py — 滑窗推理 + mIoU 评估 (Converse2D 版模型)

相对 eval.py 的唯一变化:
- main() 里加载的模型类换成 model2.DinoV3SegLoRAConverse
- run_validation 函数本身和模型架构无关，逻辑完全一致
python .\eval2.py --ckpt "D:\pycharm_projects\NOTRAING\NOTRAING\DINOv3seg\runs\run_converse_20260429_134655\best.pth" --save_json "D:\data\final_dataset\dino_converse_lora_miou.json"
输出口径:
  (A) dataset-wise (micro IoU): 所有图共享一个混淆矩阵 (分割论文标准)
  (B) image-wise: 与 eval_miou2.py 对齐
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from config import (
    VAL_WINDOW, VAL_STRIDE, PATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    NUM_CLASSES, CLASS_NAMES,
)
from dataset import SatelliteSegVal


# ------------------------------------------------------------
def _normalize_u8_to_tensor(img_u8):
    x = img_u8.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(x.device)
    std  = torch.tensor(IMAGENET_STD ).view(3, 1, 1).to(x.device)
    return ((x - mean) / std).unsqueeze(0)


@torch.no_grad()
def sliding_window_predict(model, image_u8, device='cuda',
                           window=VAL_WINDOW, stride=VAL_STRIDE):
    """image_u8: (3,H,W) uint8. 返回 (num_classes, H, W) fp32 logits"""
    C, H, W = image_u8.shape
    num_cls = NUM_CLASSES

    pad_h = max(0, window - H)
    pad_w = max(0, window - W)
    img = F.pad(image_u8.unsqueeze(0).float(), (0, pad_w, 0, pad_h),
                mode='reflect').squeeze(0).to(torch.uint8)
    Hp, Wp = img.shape[-2:]
    assert window % PATCH_SIZE == 0

    accum_logits = torch.zeros((num_cls, Hp, Wp), dtype=torch.float32, device=device)
    accum_weight = torch.zeros((1, Hp, Wp), dtype=torch.float32, device=device)

    ys = list(range(0, max(1, Hp - window + 1), stride))
    xs = list(range(0, max(1, Wp - window + 1), stride))
    if ys[-1] + window < Hp:
        ys.append(Hp - window)
    if xs[-1] + window < Wp:
        xs.append(Wp - window)

    g = _gaussian_window(window, device)

    model.eval()
    for y in ys:
        for x in xs:
            patch_u8 = img[:, y:y+window, x:x+window].to(device)
            x_in = _normalize_u8_to_tensor(patch_u8)
            logits = model(x_in)[0]
            accum_logits[:, y:y+window, x:x+window] += logits * g
            accum_weight[:, y:y+window, x:x+window] += g

    accum_logits = accum_logits / accum_weight.clamp_min(1e-6)
    return accum_logits[:, :H, :W]


def _gaussian_window(win, device, sigma_scale=0.125):
    coords = torch.arange(win, dtype=torch.float32, device=device) - win / 2 + 0.5
    sigma = win * sigma_scale
    g1 = torch.exp(-(coords**2) / (2 * sigma**2))
    return (g1.view(1, -1) * g1.view(-1, 1)).unsqueeze(0)


# ------------------------------------------------------------
# 混淆矩阵 / IoU
# ------------------------------------------------------------
def update_confusion(conf, pred, gt, num_classes=NUM_CLASSES):
    mask = (gt >= 0) & (gt < num_classes)
    k = gt[mask].astype(np.int64) * num_classes + pred[mask].astype(np.int64)
    binc = np.bincount(k, minlength=num_classes * num_classes)
    conf += binc.reshape(num_classes, num_classes)


def iou_from_confusion(conf):
    """dataset-wise IoU: 从全局混淆矩阵一次算出 per-class IoU"""
    tp = np.diag(conf).astype(np.float64)
    gt = conf.sum(axis=1).astype(np.float64)
    pd = conf.sum(axis=0).astype(np.float64)
    union = gt + pd - tp
    iou = np.where(union > 0, tp / union, np.nan)
    return iou


def image_wise_iou(pred, gt, num_classes=NUM_CLASSES):
    """
    image-wise IoU (对齐 eval_miou2.py):
    - 某类 GT 和 pred 都空 -> NaN (调用方应跳过)
    - 某类只有一方 -> 0.0
    """
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        gt_m = (gt == c)
        pd_m = (pred == c)
        gt_has = gt_m.any()
        pd_has = pd_m.any()
        if (not gt_has) and (not pd_has):
            out[c] = np.nan
        elif gt_has != pd_has:
            out[c] = 0.0
        else:
            inter = np.logical_and(gt_m, pd_m).sum()
            union = np.logical_or(gt_m, pd_m).sum()
            out[c] = inter / max(1, union)
    return out


# ------------------------------------------------------------
@torch.no_grad()
def run_validation(model, device='cuda', dtype=torch.bfloat16,
                   max_images=None, verbose=False):
    val_ds = SatelliteSegVal()
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=lambda b: b[0])

    # dataset-wise
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    # image-wise
    per_img_ious = [[] for _ in range(NUM_CLASSES)]

    n_done = 0
    for item in loader:
        image_u8 = item['image_u8']
        gt = item['mask'].numpy()

        logits = sliding_window_predict(model, image_u8, device=device)
        pred = logits.argmax(dim=0).cpu().numpy()

        update_confusion(conf, pred, gt)
        iou_this = image_wise_iou(pred, gt)
        for c in range(NUM_CLASSES):
            if not np.isnan(iou_this[c]):
                per_img_ious[c].append(iou_this[c])

        n_done += 1
        if verbose and n_done % 50 == 0:
            print(f"  val {n_done}/{len(val_ds)}")
        if max_images is not None and n_done >= max_images:
            break

    # ============ dataset-wise ============
    iou_ds = iou_from_confusion(conf)
    per_class_ds = {name: float(iou_ds[i]) if not np.isnan(iou_ds[i]) else 0.0
                    for i, name in enumerate(CLASS_NAMES)}
    fg_ds = [per_class_ds[n] for n in CLASS_NAMES if n != 'background']
    miou_fg_ds  = float(np.mean(fg_ds))
    miou_all_ds = float(np.mean([per_class_ds[n] for n in CLASS_NAMES]))

    # ============ image-wise ============
    per_class_iw = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class_iw[name] = float(np.mean(per_img_ious[i])) if per_img_ious[i] else 0.0
    fg_iw = [per_class_iw[n] for n in CLASS_NAMES if n != 'background']
    miou_fg_iw  = float(np.mean(fg_iw))
    miou_all_iw = float(np.mean([per_class_iw[n] for n in CLASS_NAMES]))

    return {
        'per_class_iou': per_class_ds,
        'miou_fg':       miou_fg_ds,
        'miou_all':      miou_all_ds,

        'per_class_iou_datasetwise': per_class_ds,
        'miou_fg_datasetwise':       miou_fg_ds,
        'miou_all_datasetwise':      miou_all_ds,

        'per_class_iou_imagewise':   per_class_iw,
        'miou_fg_imagewise':         miou_fg_iw,
        'miou_all_imagewise':        miou_all_iw,

        'per_class_sample_count_imagewise': {n: len(per_img_ious[i])
                                              for i, n in enumerate(CLASS_NAMES)},
        'confusion': conf.tolist(),
        'n_images':  n_done,
    }


def _print_report(res):
    print("=" * 60)
    print(f"样本数: {res['n_images']}")
    print("-" * 60)
    print("【Dataset-wise (micro IoU, 分割论文标准)】")
    for k, v in res['per_class_iou_datasetwise'].items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"  mIoU (前景3类) = {res['miou_fg_datasetwise']:.4f}")
    print(f"  mIoU (含背景4) = {res['miou_all_datasetwise']:.4f}")
    print("=" * 60)


def main(ckpt_path=None):
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=ckpt_path)
    parser.add_argument('--save_json', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载 Converse2D 版模型
    from model2 import DinoV3SegLoRAConverse
    dtype = torch.bfloat16
    model = DinoV3SegLoRAConverse(device=device, dtype=dtype)

    if args.ckpt:
        print(f"Loading ckpt: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
        print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")

    print("Running validation ...")
    res = run_validation(model, device=device, dtype=dtype, verbose=True)
    _print_report(res)

    if args.save_json:
        out = {k: v for k, v in res.items() if k != 'confusion'}
        with open(args.save_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved JSON: {args.save_json}")


if __name__ == '__main__':
    main()
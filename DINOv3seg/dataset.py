"""
dataset.py — 卫星图分割数据集

- 彩色 GT mask (RGB) -> 类别 index (int64)
- 训练：随机裁剪 512x512、随机翻转、90°旋转、颜色抖动
- 验证：返回完整图像（eval.py 自己做滑窗）
- WeightedRandomSampler：含 antenna 的图抽样概率 x ANTENNA_OVERSAMPLE
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms.functional as TF
import random

from config import (
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
    RGB_TO_CLASS, TRAIN_CROP,
    IMAGENET_MEAN, IMAGENET_STD,
    ANTENNA_OVERSAMPLE,
)


# ------------------------------------------------------------
# 工具：RGB mask -> class index map
# ------------------------------------------------------------
def rgb_mask_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """
    mask_rgb: (H,W,3) uint8
    返回 (H,W) int64，元素为 0..3
    未匹配的像素 -> 0 (background)
    """
    H, W, _ = mask_rgb.shape
    out = np.zeros((H, W), dtype=np.int64)
    for rgb, cls in RGB_TO_CLASS.items():
        if cls == 0:
            continue  # background 默认就是 0，不用写
        r, g, b = rgb
        match = (mask_rgb[..., 0] == r) & (mask_rgb[..., 1] == g) & (mask_rgb[..., 2] == b)
        out[match] = cls
    return out


def class_to_rgb_mask(cls_map: np.ndarray) -> np.ndarray:
    """反向：class map -> RGB 图（可视化用）"""
    H, W = cls_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for rgb, cls in RGB_TO_CLASS.items():
        if cls == 0:
            continue
        out[cls_map == cls] = rgb
    return out


# ------------------------------------------------------------
# Train Dataset
# ------------------------------------------------------------
class SatelliteSegTrain(Dataset):
    def __init__(self, img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR,
                 crop=TRAIN_CROP, mask_suffix='_mask.png'):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.crop = crop
        self.mask_suffix = mask_suffix

        imgs = sorted(list(self.img_dir.glob('*.png')) + list(self.img_dir.glob('*.jpg')))
        # 只保留有对应 mask 的
        self.items = []
        for p in imgs:
            mp = self.mask_dir / f"{p.stem}{mask_suffix}"
            if mp.exists():
                self.items.append((p, mp))
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")
        print(f"[Train Dataset] {len(self.items)} pairs")

        # 预扫：哪些图含 antenna？用于 oversampling weights
        self._antenna_flags = None

    def __len__(self):
        return len(self.items)

    # 供 sampler 调用
    def get_sample_weights(self):
        """抽 antenna 图的概率 = ANTENNA_OVERSAMPLE 倍其他图"""
        if self._antenna_flags is None:
            print("[Train Dataset] Scanning masks for antenna presence "
                  "(一次性，为了 oversampling)...")
            flags = np.zeros(len(self.items), dtype=bool)
            antenna_rgb = np.array([0, 0, 255], dtype=np.uint8)  # antenna
            for i, (_, mp) in enumerate(self.items):
                arr = np.asarray(Image.open(mp).convert('RGB'))
                has = np.any(np.all(arr == antenna_rgb, axis=-1))
                flags[i] = has
                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{len(self.items)}")
            self._antenna_flags = flags
            print(f"  含 antenna 的图: {flags.sum()}/{len(flags)} "
                  f"({100.0*flags.sum()/len(flags):.1f}%)")

        w = np.ones(len(self.items), dtype=np.float32)
        w[self._antenna_flags] = ANTENNA_OVERSAMPLE
        return w

    # ------------------ 主逻辑 ------------------
    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        img = Image.open(img_path).convert('RGB')   # 1280x720
        msk = Image.open(mask_path).convert('RGB')

        img_np = np.asarray(img)                      # (H,W,3) uint8
        msk_np = np.asarray(msk)                      # (H,W,3) uint8
        cls_map = rgb_mask_to_class(msk_np)           # (H,W) int64

        # ============ 增强 ============
        img_np, cls_map = self._augment(img_np, cls_map)

        # to tensor + normalize
        # img_t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        img_t = torch.from_numpy(img_np.copy()).float().permute(2, 0, 1) / 255.0
        img_t = TF.normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # msk_t = torch.from_numpy(cls_map).long()
        msk_t = torch.from_numpy(cls_map.copy()).long()

        return img_t, msk_t

    # ------------------ 增强 ------------------
    def _augment(self, img: np.ndarray, cls_map: np.ndarray):
        H, W = img.shape[:2]
        crop = self.crop

        # --- 1. 随机裁剪（偏向含 antenna 的区域）---
        # 30% 概率做 "antenna-centered" 裁剪（如果这张有 antenna）
        antenna_ys, antenna_xs = np.where(cls_map == 3)
        if len(antenna_ys) > 0 and random.random() < 0.4:
            # 随机选一个 antenna 像素作中心
            i = random.randint(0, len(antenna_ys) - 1)
            cy, cx = int(antenna_ys[i]), int(antenna_xs[i])
            # 加入一点抖动
            cy += random.randint(-crop // 4, crop // 4)
            cx += random.randint(-crop // 4, crop // 4)
            y0 = np.clip(cy - crop // 2, 0, max(0, H - crop))
            x0 = np.clip(cx - crop // 2, 0, max(0, W - crop))
        else:
            # 纯随机裁剪
            y0 = random.randint(0, max(0, H - crop))
            x0 = random.randint(0, max(0, W - crop))

        img = img[y0:y0 + crop, x0:x0 + crop]
        cls_map = cls_map[y0:y0 + crop, x0:x0 + crop]

        # 若裁出来尺寸不足（不应该发生，720 < 512 不会，但 720 刚好够），补 0
        if img.shape[0] != crop or img.shape[1] != crop:
            pad_h = crop - img.shape[0]
            pad_w = crop - img.shape[1]
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            cls_map = np.pad(cls_map, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # --- 2. 翻转 ---
        if random.random() < 0.5:  # h flip
            img = np.ascontiguousarray(img[:, ::-1])
            cls_map = np.ascontiguousarray(cls_map[:, ::-1])
        if random.random() < 0.5:  # v flip（卫星图无方向语义）
            img = np.ascontiguousarray(img[::-1, :])
            cls_map = np.ascontiguousarray(cls_map[::-1, :])

        # --- 3. 90° 旋转 ---
        k = random.randint(0, 3)
        if k > 0:
            img = np.ascontiguousarray(np.rot90(img, k))
            cls_map = np.ascontiguousarray(np.rot90(cls_map, k))

        # --- 4. 轻颜色抖动（只动亮度和对比度，不动色调）---
        if random.random() < 0.5:
            # brightness
            b = 1.0 + random.uniform(-0.15, 0.15)
            img = np.clip(img.astype(np.float32) * b, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            # contrast
            c = 1.0 + random.uniform(-0.15, 0.15)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * c + mean, 0, 255).astype(np.uint8)

        return img, cls_map


# ------------------------------------------------------------
# Val Dataset（不做增强，返回整图，由 eval.py 做滑窗）
# ------------------------------------------------------------
class SatelliteSegVal(Dataset):
    def __init__(self, img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR,
                 mask_suffix='_mask.png'):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        imgs = sorted(list(self.img_dir.glob('*.png')) + list(self.img_dir.glob('*.jpg')))
        self.items = []
        for p in imgs:
            mp = self.mask_dir / f"{p.stem}{mask_suffix}"
            if mp.exists():
                self.items.append((p, mp))
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")
        print(f"[Val Dataset] {len(self.items)} pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        img = np.asarray(Image.open(img_path).convert('RGB'))
        msk = np.asarray(Image.open(mask_path).convert('RGB'))
        cls_map = rgb_mask_to_class(msk)

        # 不 normalize，eval 脚本切 patch 后再做
        # img_t = torch.from_numpy(img).permute(2, 0, 1)  # uint8 C,H,W
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1)

        # msk_t = torch.from_numpy(cls_map).long()
        msk_t = torch.from_numpy(cls_map.copy()).long()


        return {
            'image_u8': img_t,             # uint8, C,H,W
            'mask':     msk_t,             # H,W
            'stem':     img_path.stem,
            'path':     str(img_path),
        }


# ------------------------------------------------------------
# Sampler helper
# ------------------------------------------------------------
def make_train_sampler(dataset: SatelliteSegTrain, num_samples=None):
    w = dataset.get_sample_weights()
    n = num_samples if num_samples is not None else len(dataset)
    return WeightedRandomSampler(weights=torch.from_numpy(w).double(),
                                  num_samples=n, replacement=True)


# ------------------------------------------------------------
# 自测
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Train dataset 自测")
    print("=" * 50)
    ds = SatelliteSegTrain()
    img, msk = ds[0]
    print(f"img: {img.shape} {img.dtype}   mean={img.mean():.3f}")
    print(f"msk: {msk.shape} {msk.dtype}   unique={torch.unique(msk).tolist()}")

    print("\n" + "=" * 50)
    print("Val dataset 自测")
    print("=" * 50)
    vds = SatelliteSegVal()
    item = vds[0]
    print(f"image_u8: {item['image_u8'].shape} {item['image_u8'].dtype}")
    print(f"mask    : {item['mask'].shape}")
    print(f"stem    : {item['stem']}")

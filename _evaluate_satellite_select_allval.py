"""
SP-SAM 在卫星数据集 (final_dataset) 上的评估脚本
===================================================

指标计算方式：
  前景类 IoU：每个类别独立的 2×2 二值混淆矩阵，跨全部 query 图像累积后计算
    IoU_body        = TP / (TP + FP + FN)   (body 的二值分割)
    IoU_solar_panel = TP / (TP + FP + FN)   (solar_panel 的二值分割)
    IoU_antenna     = TP / (TP + FP + FN)   (antenna 的二值分割)

  背景 IoU：将三个类别的预测和GT各自合并为"有无前景"的二值图，
    bg_pred = 三个类别都预测为背景的像素
    bg_gt   = 三个类别GT都为背景的像素（即不属于任何前景类）
    IoU_bg  = 在统一背景定义下的 IoU

  汇总指标：
    mIoU(fg)  = mean(IoU_body, IoU_solar_panel, IoU_antenna)
    mIoU(all) = mean(IoU_bg, IoU_body, IoU_solar_panel, IoU_antenna)

  注意：SP-SAM 对每个类别独立预测二值 mask，不同类别预测之间可能重叠。
  前景类 IoU 在各自独立的二值语境下计算，不受其他类别预测的影响。
  背景 IoU 使用合并后的统一定义，与 ablation_seg.py 的背景含义对齐。

Support 选择方式（二选一，通过 --support_json 控制）：
  - 提供 --support_json：从 support_selector 生成的排序结果中取 top-k
  - 不提供 --support_json：随机选 k 个（用于对比实验或未跑 selector 时）

全监督评估：对所有val集图像进行评估，无论是否包含目标类别

使用方法：
    # 使用最优 support
    python evaluate_satellite_full.py --dataset_root D:/data/final_dataset/final_dataset --support_json satellite_support_ranking.json --mode cmrs_memory --k_shot 5

    # 随机 support
    python evaluate_satellite_full.py \\
        --dataset_root D:/data/final_dataset/final_dataset \\
        --mode cmrs_memory --k_shot 5
"""

import argparse
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ================================================================
# 数据集加载器
# ================================================================

class SatelliteDataset:

    COLOR_MAP = {
        1: (0, 255, 0),    # body        -> 绿色
        2: (255, 0, 0),    # solar_panel -> 红色
        3: (0, 0, 255),    # antenna     -> 蓝色
    }
    CATEGORIES = {
        1: 'body',
        2: 'solar_panel',
        3: 'antenna',
    }

    def __init__(self, root: str, split: str = 'val'):
        self.root     = Path(root)
        self.split    = split
        self.img_dir  = self.root / 'images' / split
        self.mask_dir = self.root / 'masks'  / split

        ann_file = self.root / 'annotations' / f'instances_{split}_satellite.json'
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        self.id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        self.categories     = {cat['id']: cat['name'] for cat in coco['categories']}
        self.images         = coco['images']  # 保存所有图像信息

        self.img_to_anns = defaultdict(list)
        for ann in coco['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)

        self.cat_to_imgs = defaultdict(set)
        for ann in coco['annotations']:
            self.cat_to_imgs[ann['category_id']].add(ann['image_id'])

        print(f"[Dataset] {split} set: {len(self.id_to_filename)} images")
        for cat_id, cat_name in self.categories.items():
            n = len(self.cat_to_imgs.get(cat_id, []))
            print(f"   category {cat_id} ({cat_name}): {n} images with this class")

    def get_sample(self, image_id: int, category_id: int):
        """获取图像和指定类别的mask"""
        filename = self.id_to_filename[image_id]
        img = Image.open(self.img_dir / filename).convert('RGB')
        stem = Path(filename).stem
        
        # 尝试加载mask
        mask_path = self.mask_dir / f"{stem}_mask.png"
        if mask_path.exists():
            mask_rgb = np.array(Image.open(mask_path))
            # 查找指定类别的mask
            mask = np.all(mask_rgb == self.COLOR_MAP[category_id], axis=2).astype(np.uint8)
        else:
            # 如果没有mask文件，创建全零mask
            img_array = np.array(img)
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            
        return img, mask

    def get_all_image_ids(self, max_samples: int = None, random_seed: int = 42):
        """获取所有图像ID（全监督评估）"""
        img_ids = list(self.id_to_filename.keys())
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(img_ids)
        if max_samples is not None:
            img_ids = img_ids[:max_samples]
        return img_ids

    def get_samples_for_category(self, category_id: int,
                                  max_samples: int = None,
                                  random_seed: int = 42):
        """获取包含指定类别的图像ID（原few-shot评估用）"""
        img_ids = list(self.cat_to_imgs.get(category_id, []))
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(img_ids)
        if max_samples is not None:
            img_ids = img_ids[:max_samples]
        return img_ids


# ================================================================
# Support 加载工具
# ================================================================

def load_best_supports(ranking_json: str,
                        train_dataset: SatelliteDataset,
                        category_id: int,
                        k_shot: int) -> tuple:
    with open(ranking_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ranked = data['results'].get(str(category_id), [])
    if not ranked:
        raise ValueError(f"category_id={category_id} not found in {ranking_json}")

    top_k_entries = ranked[:k_shot]
    support_images, support_masks = [], []

    for entry in top_k_entries:
        img_id = entry['image_id']
        try:
            img, mask = train_dataset.get_sample(img_id, category_id)
            if mask.sum() > 0:
                support_images.append(img)
                support_masks.append(mask)
        except Exception as e:
            print(f"  [WARN] load support id={img_id} failed: {e}")

    return support_images, support_masks


def load_random_supports(train_dataset: SatelliteDataset,
                          category_id: int,
                          k_shot: int,
                          random_seed: int = 42) -> tuple:
    img_ids = train_dataset.get_samples_for_category(
        category_id, max_samples=k_shot*2, random_seed=random_seed)
    
    support_images, support_masks = [], []
    for img_id in img_ids:
        try:
            img, mask = train_dataset.get_sample(img_id, category_id)
            if mask.sum() > 0:
                support_images.append(img)
                support_masks.append(mask)
                if len(support_images) >= k_shot:
                    break
        except Exception as e:
            print(f"  [WARN] load support id={img_id} failed: {e}")

    if len(support_images) < k_shot:
        all_img_ids = list(train_dataset.id_to_filename.keys())
        random.shuffle(all_img_ids)
        for img_id in all_img_ids:
            if img_id not in img_ids:
                try:
                    img, mask = train_dataset.get_sample(img_id, category_id)
                    if mask.sum() > 0:
                        support_images.append(img)
                        support_masks.append(mask)
                        if len(support_images) >= k_shot:
                            break
                except Exception as e:
                    print(f"  [WARN] load support id={img_id} failed: {e}")

    return support_images, support_masks


# ================================================================
# 指标计算
# ================================================================

class BinarySegMetrics:
    """
    单类别二值分割的混淆矩阵累积器。
    用于计算每个前景类别的独立 IoU。

    混淆矩阵布局（行=GT，列=Pred）：
        gt\\pred | 0(bg) | 1(fg)
        --------+-------+------
        0 (bg)  |  TN   |  FP
        1 (fg)  |  FN   |  TP

    IoU_fg = TP / (TP + FP + FN)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.confusion = np.zeros((2, 2), dtype=np.int64)

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        pred = (pred_mask > 0).astype(np.int64).ravel()
        gt   = (gt_mask   > 0).astype(np.int64).ravel()
        np.add.at(self.confusion.ravel(), gt * 2 + pred, 1)

    def iou_fg(self):
        tp = self.confusion[1, 1]
        denom = self.confusion[1].sum() + self.confusion[:, 1].sum() - tp
        return float(tp) / float(denom) if denom > 0 else float('nan')

    def iou_bg(self):
        tn = self.confusion[0, 0]
        denom = self.confusion[0].sum() + self.confusion[:, 0].sum() - tn
        return float(tn) / float(denom) if denom > 0 else float('nan')

    def pixel_acc(self):
        total = self.confusion.sum()
        return float(np.diag(self.confusion).sum()) / float(total) if total > 0 else 0.0

    def summary(self) -> dict:
        fg = self.iou_fg()
        bg = self.iou_bg()
        return {
            'iou_fg':    fg if not np.isnan(fg) else 0.0,
            'iou_bg':    bg if not np.isnan(bg) else 0.0,
            'pixel_acc': self.pixel_acc(),
            'TP': int(self.confusion[1, 1]),
            'FP': int(self.confusion[0, 1]),
            'FN': int(self.confusion[1, 0]),
            'TN': int(self.confusion[0, 0]),
        }


class UnifiedBackgroundMetrics:
    """
    统一背景 IoU 计算器。

    背景定义：一个像素在 GT 中不属于任何前景类，且在预测中也不属于任何前景类，
    才算背景的 TP (TN in binary sense)。

    对每张图像，收集所有类别的 GT 和预测：
      bg_gt   = NOT(body_gt OR solar_gt OR antenna_gt)
      bg_pred = NOT(body_pred OR solar_pred OR antenna_pred)

    然后用 BinarySegMetrics 的方式累积。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 用 2x2 矩阵，class 1 = background, class 0 = any foreground
        # 这样 iou_fg() 就是背景的 IoU
        self.confusion = np.zeros((2, 2), dtype=np.int64)

    def update(self, bg_pred: np.ndarray, bg_gt: np.ndarray):
        """bg_pred/bg_gt: 1=background, 0=foreground"""
        pred = bg_pred.astype(np.int64).ravel()
        gt   = bg_gt.astype(np.int64).ravel()
        np.add.at(self.confusion.ravel(), gt * 2 + pred, 1)

    def iou_bg(self):
        """背景的 IoU"""
        tp = self.confusion[1, 1]  # bg predicted as bg
        denom = self.confusion[1].sum() + self.confusion[:, 1].sum() - tp
        return float(tp) / float(denom) if denom > 0 else float('nan')

    def summary(self) -> dict:
        bg = self.iou_bg()
        return {
            'iou_bg': bg if not np.isnan(bg) else 0.0,
            'bg_TP': int(self.confusion[1, 1]),  # bg correct
            'bg_FP': int(self.confusion[0, 1]),  # fg predicted as bg (miss)
            'bg_FN': int(self.confusion[1, 0]),  # bg predicted as fg (false alarm)
            'bg_TN': int(self.confusion[0, 0]),  # fg correct
        }


def compute_per_image_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """单张图 IoU，仅用于可视化排序，不作为最终指标。"""
    pred_b = (pred > 0).astype(bool)
    gt_b   = (gt   > 0).astype(bool)
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return float(intersection / union) if union > 0 else float('nan')


# ================================================================
# 可视化工具
# ================================================================

VIS_COLORS = {
    1: (0,   255,   0, 120),
    2: (255,   0,   0, 120),
    3: (0,     0, 255, 120),
}
CAT_NAMES = {1: 'body', 2: 'solar_panel', 3: 'antenna'}


def _overlay(ax, base_img_np, masks_dict):
    ax.imshow(base_img_np)
    H, W = base_img_np.shape[:2]
    for cat_id, mask in masks_dict.items():
        if mask is None or mask.sum() == 0:
            continue
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[mask > 0] = VIS_COLORS[cat_id]
        ax.imshow(rgba)


def _legend(ax, cat_ids):
    patches = [mpatches.Patch(
        color=[v/255 for v in VIS_COLORS[c][:3]],
        label=CAT_NAMES.get(c, str(c))
    ) for c in cat_ids]
    ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.7)


def make_per_image_figure(query_img, gt_masks, pred_masks,
                           support_images, support_masks_dict,
                           ious, image_id):
    cat_ids  = sorted(gt_masks.keys())
    n_cats   = len(cat_ids)
    query_np = np.array(query_img)
    n_cols   = max(3 + n_cats * 2, n_cats * 3)
    fig      = plt.figure(figsize=(max(n_cols * 3, 18), 10))
    gs       = GridSpec(2, n_cols, figure=fig, hspace=0.35, wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(query_np)
    ax.set_title(f"Query  id={image_id}", fontsize=9, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    _overlay(ax, query_np, gt_masks)
    _legend(ax, cat_ids)
    ax.set_title("GT (all classes)", fontsize=9, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    _overlay(ax, query_np, pred_masks)
    _legend(ax, cat_ids)
    # 只用有GT前景的类别算mean IoU用于显示
    valid_ious = [v for k, v in ious.items() if not np.isnan(v)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    ax.set_title(f"Pred  mean IoU(fg)={mean_iou:.3f}", fontsize=9, fontweight='bold')
    ax.axis('off')

    for i, cat_id in enumerate(cat_ids):
        cat_name = CAT_NAMES.get(cat_id, f'cat{cat_id}')
        iou_val  = ious.get(cat_id, 0.0)
        col_base = 3 + i * 2

        ax_gt = fig.add_subplot(gs[0, col_base])
        _overlay(ax_gt, query_np, {cat_id: gt_masks[cat_id]})
        ax_gt.set_title(f"{cat_name}\nGT", fontsize=8)
        ax_gt.axis('off')

        ax_pr = fig.add_subplot(gs[0, col_base + 1])
        _overlay(ax_pr, query_np, {cat_id: pred_masks.get(cat_id)})
        iou_str = f"IoU={iou_val:.3f}" if not np.isnan(iou_val) else "no GT"
        ax_pr.set_title(f"{cat_name}\n{iou_str}", fontsize=8,
                        color='green' if (not np.isnan(iou_val) and iou_val >= 0.5) else 'red')
        ax_pr.axis('off')

    for i, cat_id in enumerate(cat_ids):
        cat_name = CAT_NAMES.get(cat_id, f'cat{cat_id}')
        sup_imgs = support_images.get(cat_id, [])
        sup_msks = support_masks_dict.get(cat_id, [])
        for j in range(3):
            col = i * 3 + j
            ax  = fig.add_subplot(gs[1, col])
            if j < len(sup_imgs):
                ax.imshow(np.array(sup_imgs[j]))
                if j < len(sup_msks):
                    rgba = np.zeros((*sup_msks[j].shape, 4), dtype=np.uint8)
                    rgba[sup_msks[j] > 0] = VIS_COLORS[cat_id]
                    ax.imshow(rgba)
                ax.set_title(f"{cat_name} Sup{j+1}" if j == 0 else f"Sup{j+1}", fontsize=8)
            else:
                ax.set_visible(False)
            ax.axis('off')

    fig.suptitle(f"Image {image_id}  |  classes: {[CAT_NAMES[c] for c in cat_ids]}",
                 fontsize=11, fontweight='bold')
    return fig


# ================================================================
# 主评估器
# ================================================================

class SatelliteEvaluator:

    def __init__(self, sp_sam, output_dir: str):
        self.sp_sam     = sp_sam
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_image_results = defaultdict(dict)

    # ----------------------------------------------------------
    def print_support_info(self, support_images: list,
                            support_masks: list,
                            category_id: int,
                            support_source: str):
        cat_name = CAT_NAMES.get(category_id, f'cat_{category_id}')
        k = len(support_images)

        print(f"\n{'='*60}")
        print(f"Category: {cat_name} (id={category_id})  |  source: {support_source}")
        print(f"Using {k} support images:")

        fig, axes = plt.subplots(2, k, figsize=(5 * k, 10))
        if k == 1:
            axes = axes.reshape(2, 1)

        for i, (img, mask) in enumerate(zip(support_images, support_masks)):
            fg_ratio = mask.sum() / mask.size * 100
            print(f"  Support {i+1}: fg_ratio={fg_ratio:.2f}%, fg_pixels={mask.sum():,}")

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Support {i+1}", fontsize=9)
            axes[0, i].axis('off')

            axes[1, i].imshow(img)
            ov = np.zeros((*mask.shape, 4), dtype=np.float32)
            ov[mask > 0] = [0, 1, 0, 0.5]
            axes[1, i].imshow(ov)
            axes[1, i].set_title(f"fg={fg_ratio:.1f}%", fontsize=9)
            axes[1, i].axis('off')

        plt.suptitle(f"{cat_name} - {k}-shot Support  [{support_source}]",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        tag = 'best' if 'ranking' in support_source.lower() else 'random'
        save_path = self.output_dir / f"support_vis_{cat_name}_{tag}.png"
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    def _parse_mode(self, mode):
        if mode == 'rough_only':
            return dict(use_cmrs=True, use_memory_refinement=False,
                        use_predictor_refinement=False), True
        elif mode == 'cmrs_predictor':
            return dict(use_cmrs=True, use_memory_refinement=False,
                        use_predictor_refinement=True), False
        elif mode == 'cmrs_memory':
            return dict(use_cmrs=True, use_memory_refinement=True,
                        use_predictor_refinement=False), False
        elif mode == 'memory_only':
            return dict(use_cmrs=False, use_memory_refinement=True,
                        use_predictor_refinement=False), False
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ----------------------------------------------------------
    def evaluate_category(self, train_dataset, val_dataset,
                           category_id, k_shot=5, mode='cmrs_memory',
                           max_query=None, random_seed=42,
                           support_json: str = None,
                           query_ids: list = None):
        """
        对单个类别运行预测，累积该类别的 BinarySegMetrics。
        同时将每张图的 pred_mask / gt_mask 存入 self.per_image_results
        供后续统一背景 IoU 计算使用。

        参数 query_ids: 如果提供，使用指定的图像ID列表（确保所有类别用同一批图像）
        """

        cat_name = train_dataset.categories.get(category_id, f'cat_{category_id}')
        predict_kwargs, rough_only = self._parse_mode(mode)

        print(f"\n{'='*60}")
        print(f"Category: {cat_name}  |  mode: {mode}  |  k_shot={k_shot}")
        print(f"{'='*60}")

        # ---- 加载 support ----
        if support_json is not None and Path(support_json).exists():
            support_source = f"ranking: {support_json}"
            try:
                support_images, support_masks = load_best_supports(
                    ranking_json=support_json,
                    train_dataset=train_dataset,
                    category_id=category_id,
                    k_shot=k_shot,
                )
                print(f"  Loaded best support from {support_json}, count={len(support_images)}")
            except Exception as e:
                print(f"  [WARN] load support_json failed: {e}, fallback to random")
                support_source = f"random (fallback, seed={random_seed})"
                support_images, support_masks = load_random_supports(
                    train_dataset, category_id, k_shot, random_seed)
        else:
            if support_json is not None:
                print(f"  [WARN] {support_json} not found, using random support")
            support_source = f"random (seed={random_seed})"
            support_images, support_masks = load_random_supports(
                train_dataset, category_id, k_shot, random_seed)
            print(f"  Random support, count={len(support_images)}")

        if not support_images:
            print(f"[WARN] {cat_name}: no valid support, skip")
            return None

        self.print_support_info(support_images, support_masks,
                                 category_id, support_source)

        # ---- 缓存 support ----
        use_cached = False
        if predict_kwargs.get('use_memory_refinement'):
            try:
                self.sp_sam.set_support(support_images, support_masks)
                use_cached = True
                print("   Support memory cached")
            except Exception as e:
                print(f"   [WARN] set_support failed: {e}")

        # ---- 确定 query 图像 ----
        if query_ids is None:
            all_val_ids = list(val_dataset.id_to_filename.keys())
            if max_query is not None and max_query < len(all_val_ids):
                random.seed(random_seed)
                query_ids = random.sample(all_val_ids, max_query)
            else:
                query_ids = all_val_ids

        print(f"Query: {len(query_ids)} images")

        # ---- 累积该类别的混淆矩阵 ----
        seg_metrics = BinarySegMetrics()
        per_image   = []
        total_time  = 0.0
        n_processed = 0
        n_with_gt   = 0

        for img_id in tqdm(query_ids, desc=cat_name):
            try:
                query_img, gt_mask = val_dataset.get_sample(img_id, category_id)
            except Exception as e:
                print(f"  [WARN] load image {img_id} failed: {e}")
                continue

            has_gt = bool(gt_mask.sum() > 0)
            if has_gt:
                n_with_gt += 1

            t0 = time.time()
            try:
                if use_cached:
                    pred_results = self.sp_sam.predict_query(
                        query_img, **predict_kwargs)
                else:
                    pred_results = self.sp_sam.predict(
                        query_img, support_images, support_masks,
                        **predict_kwargs)

                pred_mask = pred_results.get('final_mask')
                if rough_only:
                    pred_mask = pred_results.get('rough_mask', pred_mask)
                if pred_mask is None:
                    pred_mask = np.zeros_like(gt_mask)

            except Exception as e:
                print(f"\n   [ERROR] predict failed (id={img_id}): {e}")
                pred_mask = np.zeros_like(gt_mask)

            elapsed = time.time() - t0
            total_time += elapsed
            n_processed += 1

            # 更新该类别的混淆矩阵
            seg_metrics.update(pred_mask, gt_mask)

            # 单张图 IoU（用于可视化，GT全0且pred全0时返回nan表示"不适用"）
            img_iou = compute_per_image_iou(pred_mask, gt_mask)

            per_image.append({
                'image_id':       img_id,
                'iou_fg':         img_iou if not np.isnan(img_iou) else 0.0,
                'iou_fg_raw':     img_iou,  # 保留nan，用于可视化
                'inference_time': elapsed,
                'category_id':    category_id,
                'category':       cat_name,
                'has_gt':         has_gt,
            })

            # 存储到 per_image_results，供统一背景计算和可视化使用
            self.per_image_results[img_id][category_id] = {
                'pred_mask':      pred_mask.copy(),
                'gt_mask':        gt_mask.copy(),
                'iou_fg':         img_iou if not np.isnan(img_iou) else 0.0,
                'iou_fg_raw':     img_iou,
                'has_gt':         has_gt,
                'query_img':      query_img,
                'support_images': support_images,
                'support_masks':  support_masks,
            }

        if n_processed == 0:
            return None

        m = seg_metrics.summary()
        mean_time = total_time / n_processed

        print(f"\n{'~'*55}")
        print(f"  {cat_name}: {n_processed} images ({n_with_gt} with GT)")
        print(f"  Confusion matrix:")
        print(f"             pred_bg        pred_fg")
        print(f"   gt_bg  | TN={m['TN']:>12,} | FP={m['FP']:>12,} |")
        print(f"   gt_fg  | FN={m['FN']:>12,} | TP={m['TP']:>12,} |")
        print(f"  IoU(fg) = {m['iou_fg']:.4f}  |  PixAcc = {m['pixel_acc']:.4f}")
        print(f"  Support: {support_source}")
        print(f"  Mean inference time: {mean_time:.3f}s")
        print(f"{'~'*55}")

        return {
            'category_id':    category_id,
            'category':       cat_name,
            'num_images':     n_processed,
            'num_with_gt':    n_with_gt,
            'support_source': support_source,
            'iou_fg':         m['iou_fg'],
            'pixel_acc':      m['pixel_acc'],
            'TP':             m['TP'],
            'FP':             m['FP'],
            'FN':             m['FN'],
            'TN':             m['TN'],
            'mean_time':      mean_time,
            'per_sample':     per_image,
        }

    # ----------------------------------------------------------
    def compute_unified_background_iou(self, query_ids, category_ids):
        """
        计算统一的背景 IoU。

        背景定义：
          bg_gt[i]   = 1  iff  对所有前景类 c, gt_mask_c[i] == 0
          bg_pred[i] = 1  iff  对所有前景类 c, pred_mask_c[i] == 0

        这样背景 IoU 的含义与 ablation_seg 中 4×4 矩阵的 class 0 一致：
        只有真正不属于任何前景类的像素才被视为背景。
        """
        bg_metrics = UnifiedBackgroundMetrics()
        n_computed = 0

        for img_id in query_ids:
            if img_id not in self.per_image_results:
                continue

            cat_data = self.per_image_results[img_id]

            # 检查是否所有类别都有结果
            if not all(cid in cat_data for cid in category_ids):
                continue

            # 取一个参考shape
            ref_data = cat_data[category_ids[0]]
            H, W = ref_data['gt_mask'].shape

            # bg_gt: 所有类别GT都为0的像素
            any_fg_gt = np.zeros((H, W), dtype=bool)
            for cid in category_ids:
                any_fg_gt |= (cat_data[cid]['gt_mask'] > 0)
            bg_gt = (~any_fg_gt).astype(np.uint8)

            # bg_pred: 所有类别预测都为0的像素
            any_fg_pred = np.zeros((H, W), dtype=bool)
            for cid in category_ids:
                any_fg_pred |= (cat_data[cid]['pred_mask'] > 0)
            bg_pred = (~any_fg_pred).astype(np.uint8)

            bg_metrics.update(bg_pred, bg_gt)
            n_computed += 1

        result = bg_metrics.summary()
        result['num_images'] = n_computed
        return result

    # ----------------------------------------------------------
    def save_visualizations(self, top_k: int = 30):
        if not self.per_image_results:
            print("[WARN] no visualization data")
            return

        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        filtered_images = []
        for img_id, cat_data in self.per_image_results.items():
            # 只用有GT的类别的IoU来排序
            valid_ious = []
            for cat_id, data in cat_data.items():
                raw_iou = data.get('iou_fg_raw', data.get('iou_fg', 0.0))
                if not np.isnan(raw_iou):
                    valid_ious.append(raw_iou)
            
            if valid_ious:
                mean_iou = float(np.mean(valid_ious))
            else:
                mean_iou = 0.0
            filtered_images.append((img_id, mean_iou))

        img_scores = sorted(filtered_images, key=lambda x: -x[1])

        print(f"\nSaving top {min(top_k, len(img_scores))} visualizations...")

        for rank, (img_id, mean_iou_fg) in enumerate(
                tqdm(img_scores[:top_k], desc="Saving vis"), 1):
            cat_data   = self.per_image_results[img_id]
            gt_masks   = {cid: d['gt_mask']        for cid, d in cat_data.items()}
            pred_masks = {cid: d['pred_mask']       for cid, d in cat_data.items()}
            # 用 raw IoU (含nan) 来显示
            ious       = {cid: d.get('iou_fg_raw', d['iou_fg']) for cid, d in cat_data.items()}
            query_img  = next(iter(cat_data.values()))['query_img']
            sup_imgs   = {cid: d['support_images']  for cid, d in cat_data.items()}
            sup_masks  = {cid: d['support_masks']   for cid, d in cat_data.items()}

            fig = make_per_image_figure(query_img, gt_masks, pred_masks,
                                        sup_imgs, sup_masks, ious, img_id)
            fig.savefig(vis_dir / f"rank{rank:03d}_id{img_id}_iou{mean_iou_fg:.3f}.png",
                        dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        print(f"Visualizations saved to: {vis_dir}")

    # ----------------------------------------------------------
    def save_per_image_results(self, all_category_results, mode, k_shot):
        """保存每张图像的所有类别结果"""
        per_image_data = []
        
        all_image_ids = set()
        for cat_result in all_category_results:
            for sample in cat_result.get('per_sample', []):
                all_image_ids.add(sample['image_id'])
        
        for img_id in sorted(all_image_ids):
            img_record = {
                'image_id': int(img_id),
                'categories': {}
            }
            
            for cat_result in all_category_results:
                cat_id = cat_result['category_id']
                cat_name = cat_result['category']
                
                cat_samples = [s for s in cat_result.get('per_sample', []) 
                              if s['image_id'] == img_id]
                if cat_samples:
                    sample = cat_samples[0]
                    img_record['categories'][cat_name] = {
                        'iou_fg': float(sample['iou_fg']),
                        'inference_time': float(sample['inference_time']),
                        'has_gt': bool(sample.get('has_gt', True))
                    }
                else:
                    img_record['categories'][cat_name] = {
                        'iou_fg': 0.0,
                        'inference_time': 0.0,
                        'has_gt': False
                    }
            
            # 只用有GT的类别算 mean_iou
            ious = [cat_info['iou_fg'] for cat_info in img_record['categories'].values() 
                    if cat_info.get('has_gt', True)]
            img_record['mean_iou'] = float(np.mean(ious)) if ious else 0.0
            
            per_image_data.append(img_record)
        
        # CSV
        df_data = []
        for record in per_image_data:
            row = {
                'image_id': record['image_id'],
                'mean_iou': record['mean_iou']
            }
            for cat_name, cat_info in record['categories'].items():
                row[f'{cat_name}_iou'] = cat_info['iou_fg']
                row[f'{cat_name}_time'] = cat_info['inference_time']
                row[f'{cat_name}_has_gt'] = cat_info.get('has_gt', True)
            df_data.append(row)
        
        if df_data:
            pd.DataFrame(df_data).to_csv(
                self.output_dir / f'per_image_results_{mode}_{k_shot}shot.csv',
                index=False, encoding='utf-8'
            )
            print(f"Per-image results saved to CSV")
            
            with open(self.output_dir / f'per_image_results_{mode}_{k_shot}shot.json', 
                     'w', encoding='utf-8') as f:
                json.dump(per_image_data, f, indent=2, ensure_ascii=False)
            print(f"Per-image results saved to JSON")

    # ----------------------------------------------------------
    def run(self, dataset_root, k_shot=5, mode='cmrs_memory',
            max_query=None, random_seed=42,
            category_ids=None, top_k_vis=30,
            support_json: str = None):

        print(f"\n{'='*70}")
        print(f"SP-SAM Satellite Dataset Evaluation")
        print(f"Mode: {mode}  |  K-shot: {k_shot}")
        print(f"Metrics: per-class binary IoU + unified background IoU")
        if support_json:
            print(f"Support: best selection ({support_json})")
        else:
            print(f"Support: random (seed={random_seed})")
        print(f"{'='*70}")

        train_dataset = SatelliteDataset(dataset_root, split='train')
        val_dataset   = SatelliteDataset(dataset_root, split='val')

        if category_ids is None:
            category_ids = list(train_dataset.categories.keys())

        # ---- 确定统一的 query 图像列表 ----
        # 所有类别使用相同的图像列表，确保背景IoU计算的一致性
        all_val_ids = list(val_dataset.id_to_filename.keys())
        if max_query is not None and max_query < len(all_val_ids):
            random.seed(random_seed)
            query_ids = random.sample(all_val_ids, max_query)
        else:
            query_ids = all_val_ids
        print(f"Total query images: {len(query_ids)}")

        # ---- 逐类别预测 ----
        all_cat_results = []
        for cat_id in category_ids:
            result = self.evaluate_category(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                category_id=cat_id,
                k_shot=k_shot,
                mode=mode,
                random_seed=random_seed,
                support_json=support_json,
                query_ids=query_ids,  # 传入统一的图像列表
            )
            if result is not None:
                all_cat_results.append(result)

        if not all_cat_results:
            print("\n[ERROR] no valid results")
            return {}

        # ---- 计算统一背景 IoU ----
        bg_result = self.compute_unified_background_iou(query_ids, category_ids)
        iou_bg = bg_result['iou_bg']

        # ---- 汇总指标 ----
        fg_ious = {r['category']: r['iou_fg'] for r in all_cat_results}
        miou_fg  = float(np.mean(list(fg_ious.values())))
        miou_all = float(np.mean([iou_bg] + list(fg_ious.values())))

        # 像素精度（合并所有类别的混淆矩阵）
        total_correct = sum(r['TP'] + r['TN'] for r in all_cat_results)
        total_pixels  = sum(r['TP'] + r['TN'] + r['FP'] + r['FN'] for r in all_cat_results)
        overall_pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

        n_images = len(query_ids)

        # ---- 打印汇总 ----
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS  |  mode={mode}  k_shot={k_shot}  images={n_images}")
        print(f"{'='*80}")
        print(f"  Per-class IoU (from independent binary confusion matrices):")
        print(f"    background  : {iou_bg:.4f}  (unified: pixel is bg iff no class predicts fg)")
        for r in all_cat_results:
            gt_pct = r['num_with_gt'] / r['num_images'] * 100 if r['num_images'] > 0 else 0
            print(f"    {r['category']:<14}: {r['iou_fg']:.4f}  "
                  f"({r['num_with_gt']}/{r['num_images']} images with GT, {gt_pct:.0f}%)")
        print(f"  --------------------------------")
        print(f"  mIoU(fg)  [3 fg classes]     : {miou_fg:.4f}  ({miou_fg*100:.2f}%)")
        print(f"  mIoU(all) [bg + 3 fg classes] : {miou_all:.4f}  ({miou_all*100:.2f}%)")
        print(f"  pixel_acc                     : {overall_pixel_acc:.4f}")
        print(f"")
        print(f"  Background confusion (unified):")
        print(f"    bg_TP={bg_result['bg_TP']:>12,}  bg_FP={bg_result['bg_FP']:>10,}")
        print(f"    bg_FN={bg_result['bg_FN']:>12,}  bg_TN={bg_result['bg_TN']:>10,}")
        print(f"")
        print(f"  Per-class confusion matrices:")
        for r in all_cat_results:
            print(f"    {r['category']:<14}  TP={r['TP']:>10,}  FP={r['FP']:>8,}  "
                  f"FN={r['FN']:>8,}  TN={r['TN']:>12,}")
        print(f"{'='*80}")

        # ---- 保存结果 ----
        self._save_results(all_cat_results, iou_bg, miou_fg, miou_all,
                          overall_pixel_acc, bg_result, mode, k_shot, support_json)
        self.save_per_image_results(all_cat_results, mode, k_shot)
        self.save_visualizations(top_k=top_k_vis)

        return {
            'iou_bg':           iou_bg,
            'miou_fg':          miou_fg,
            'miou_all':         miou_all,
            'pixel_acc':        overall_pixel_acc,
            'mode':             mode,
            'k_shot':           k_shot,
            'per_class_iou':    fg_ious,
            'category_results': all_cat_results,
            'bg_result':        bg_result,
        }

    # ----------------------------------------------------------
    def _save_results(self, all_cat_results, iou_bg, miou_fg, miou_all,
                     pixel_acc, bg_result, mode, k_shot, support_json):
        # CSV: per-category results
        rows = [{k: v for k, v in r.items() if k != 'per_sample'}
                for r in all_cat_results]
        pd.DataFrame(rows).to_csv(
            self.output_dir / f'results_{mode}_{k_shot}shot.csv', 
            index=False, encoding='utf-8')

        # JSON: full results
        with open(self.output_dir / f'results_{mode}_{k_shot}shot.json',
                  'w', encoding='utf-8') as f:
            json.dump({
                'mode':        mode,
                'k_shot':      k_shot,
                'support_json': support_json,
                'iou_bg':      iou_bg,
                'miou_fg':     miou_fg,
                'miou_all':    miou_all,
                'pixel_acc':   pixel_acc,
                'bg_confusion': bg_result,
                'metric_note': (
                    'miou_fg = mean(IoU_body, IoU_solar_panel, IoU_antenna). '
                    'Each fg class IoU from independent binary confusion matrix. '
                    'iou_bg = unified background IoU: pixel is bg iff no class predicts/is fg. '
                    'miou_all = mean(iou_bg, IoU_body, IoU_solar_panel, IoU_antenna). '
                    'SP-SAM predicts each class independently (binary); '
                    'ablation_seg uses 4-class softmax (mutually exclusive). '
                    'miou_fg is semantically aligned between the two.'
                ),
                'categories': [{k: v for k, v in r.items() if k != 'per_sample'}
                               for r in all_cat_results],
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {self.output_dir}")


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',  type=str, required=True)
    parser.add_argument('--mode',          type=str, default='cmrs_memory',
                        choices=['rough_only', 'cmrs_predictor',
                                 'cmrs_memory', 'memory_only'])
    parser.add_argument('--k_shot',        type=int, default=5)
    parser.add_argument('--max_query',     type=int, default=None,
                        help='max query images, None for all val images')
    parser.add_argument('--category_ids',  type=int, nargs='+', default=None)
    parser.add_argument('--top_k_vis',     type=int, default=30)
    parser.add_argument('--output_dir',    type=str, default='satellite_results_full')
    parser.add_argument('--support_json',  type=str, default=None,
                        help='support_selector ranking JSON; '
                             'random selection if not provided')
    parser.add_argument('--dino_model',    type=str, default='dinov3_vitb16')
    parser.add_argument('--sam2_model',    type=str, default='large')
    parser.add_argument('--device',        type=str, default='cuda')
    parser.add_argument('--random_seed',   type=int, default=42)
    args = parser.parse_args()

    import torch
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    from sp_sam_complete import SPSAMModel
    from src.model_manager import ModelManager

    print("\nStep 1: Loading models...")
    manager = ModelManager(device=args.device)
    if args.dino_model.startswith('dinov3'):
        dino_model, dino_transform = manager.load_dinov3_model(args.dino_model)
    else:
        dino_model, dino_transform = manager.load_dinov2_model(args.dino_model)
    sam2_model, sam2_predictor, _ = manager.load_sam2_model(args.sam2_model)

    print("\nStep 2: Initializing SP-SAM...")
    sp_sam = SPSAMModel(
        sam2_model=sam2_model,
        sam2_predictor=sam2_predictor,
        dino_model=dino_model,
        dino_transform=dino_transform,
        device=args.device,
        sam2_model_type=args.sam2_model,
    )

    print("\nStep 3: Running evaluation...")
    evaluator = SatelliteEvaluator(sp_sam, output_dir=args.output_dir)
    results = evaluator.run(
        dataset_root=args.dataset_root,
        k_shot=args.k_shot,
        mode=args.mode,
        max_query=args.max_query,
        random_seed=args.random_seed,
        category_ids=args.category_ids,
        top_k_vis=args.top_k_vis,
        support_json=args.support_json,
    )

    if results:
        print(f"\nDone!")
        print(f"   mIoU(fg)  = {results['miou_fg']:.4f}  ({results['miou_fg']*100:.2f}%)")
        print(f"   mIoU(all) = {results['miou_all']:.4f}  ({results['miou_all']*100:.2f}%)")
        print(f"   IoU(bg)   = {results['iou_bg']:.4f}")


if __name__ == '__main__':
    main()
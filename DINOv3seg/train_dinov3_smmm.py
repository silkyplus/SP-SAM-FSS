"""
train_dinov3_smmm.py — DINOv3-FPN-SMMM / DINOv3-FPN-Converse2D-SMMM 训练脚本

用于测试“方案 B：高低层分组 SMMM 融合”：

    low  = SMMM(Block5,  Block11)
    high = SMMM(Block17, Block23)
    x    = SMMM(low, high)

支持四种训练：
  1) DINOv3-FPN-SMMM + LoRA
     python train_dinov3_smmm.py --model fpn_smmm

  2) DINOv3-FPN-Converse2D-SMMM + LoRA
     python train_dinov3_smmm.py --model converse_smmm

  3) DINOv3-FPN-SMMM noLoRA
     python train_dinov3_smmm.py --model fpn_smmm --no_lora

  4) DINOv3-FPN-Converse2D-SMMM noLoRA
     python train_dinov3_smmm.py --model converse_smmm --no_lora

训练策略：
- LoRA 版：冻结 DINOv3 原始 backbone，训练 LoRA + decoder。
- noLoRA 版：冻结 DINOv3 backbone，只训练 decoder。
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, NUM_WORKERS, EPOCHS, PATIENCE,
    LR_LORA, LR_HEAD, WEIGHT_DECAY, WARMUP_FRAC, GRAD_CLIP,
    LOG_EVERY, VAL_EVERY,
    OUTPUT_DIR, SEED, USE_BF16,
    NUM_CLASSES,
)
from dataset import SatelliteSegTrain, make_train_sampler
from losses import CEDiceLoss
from eval import run_validation
from model_smmm import (
    DinoV3SegLoRASMMM,
    DinoV3SegNoLoRASMMM,
    DinoV3SegLoRAConverseSMMM,
    DinoV3SegNoLoRAConverseSMMM,
)


# ------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def build_model(args, device: str, dtype: torch.dtype):
    if args.model == "fpn_smmm" and not args.no_lora:
        return DinoV3SegLoRASMMM(
            num_classes=NUM_CLASSES,
            device=device,
            dtype=dtype,
            gn_groups=args.gn_groups,
            fusion_residual=not args.no_fusion_residual,
        )

    if args.model == "fpn_smmm" and args.no_lora:
        return DinoV3SegNoLoRASMMM(
            num_classes=NUM_CLASSES,
            device=device,
            dtype=dtype,
            gn_groups=args.gn_groups,
            fusion_residual=not args.no_fusion_residual,
        )

    if args.model == "converse_smmm" and not args.no_lora:
        return DinoV3SegLoRAConverseSMMM(
            num_classes=NUM_CLASSES,
            device=device,
            dtype=dtype,
            use_refine=not args.no_converse_refine,
            gn_groups=args.gn_groups,
            fusion_residual=not args.no_fusion_residual,
        )

    if args.model == "converse_smmm" and args.no_lora:
        return DinoV3SegNoLoRAConverseSMMM(
            num_classes=NUM_CLASSES,
            device=device,
            dtype=dtype,
            use_refine=not args.no_converse_refine,
            gn_groups=args.gn_groups,
            fusion_residual=not args.no_fusion_residual,
        )

    raise ValueError(f"未知模型: {args.model}")


def split_params(model: nn.Module, no_lora: bool):
    """
    LoRA 版：
      - decoder 参数
      - backbone 中 requires_grad=True 的 LoRA 参数

    noLoRA 版：
      - 只训练 decoder 参数
    """
    if no_lora:
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        return [], decoder_params

    lora_params, decoder_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "decoder" in name:
            decoder_params.append(p)
        else:
            lora_params.append(p)
    return lora_params, decoder_params


def count_params(params):
    return sum(p.numel() for p in params)


def keep_backbone_eval_if_nolora(model: nn.Module, no_lora: bool):
    """
    noLoRA 训练时 backbone 完全冻结，保持 eval 模式更稳。
    LoRA 训练时不要这样做，否则会影响 LoRA dropout 等训练行为。
    """
    if no_lora and hasattr(model, "backbone"):
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad = False


def save_ckpt(model: nn.Module, path: Path, epoch: int,
              miou_fg: float, miou_all: float,
              optimizer=None, global_step=None, history=None,
              extra_meta=None):
    """
    保存所有 requires_grad=True 的参数：
      - LoRA 版：LoRA + decoder
      - noLoRA 版：decoder
    """
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k in trainable_names
    }

    ckpt = {
        "state_dict": sd,
        "epoch": int(epoch),
        "miou_fg": float(miou_fg),
        "miou_all": float(miou_all),
        "model_type": "dinov3_smmm",
    }
    if extra_meta:
        ckpt.update(extra_meta)
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if global_step is not None:
        ckpt["global_step"] = int(global_step)
    if history is not None:
        ckpt["history"] = history

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"  [ckpt] saved -> {path}")


def load_resume(model: nn.Module, optimizer, resume_path: str, device: str):
    if not resume_path:
        return 1, -1.0, 0, []

    p = Path(resume_path)
    if not p.exists():
        raise FileNotFoundError(f"resume checkpoint 不存在: {p}")

    print(f"\n[Resume] loading: {p}")
    ckpt = torch.load(str(p), map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded state_dict, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing examples: {missing[:5]}")
    if unexpected:
        print(f"  unexpected examples: {unexpected[:5]}")

    if isinstance(ckpt, dict) and optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("  optimizer restored")

    start_epoch = int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else 1
    best_miou = float(ckpt.get("miou_fg", -1.0)) if isinstance(ckpt, dict) else -1.0
    global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    history = ckpt.get("history", []) if isinstance(ckpt, dict) else []
    return start_epoch, best_miou, global_step, history


def make_run_name(args):
    lora_tag = "nolora" if args.no_lora else "lora"
    if args.model == "fpn_smmm":
        return f"run_smmm_fpn_{lora_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    return f"run_smmm_converse_{lora_tag}_{time.strftime('%Y%m%d_%H%M%S')}"


# ------------------------------------------------------------
# 主训练
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fpn_smmm", "converse_smmm"],
        help="fpn_smmm 或 converse_smmm",
    )
    parser.add_argument("--no_lora", action="store_true",
                        help="不注入 LoRA，只训练 decoder")
    parser.add_argument("--run_name", type=str, default="",
                        help="输出 run 目录名；默认自动生成")
    parser.add_argument("--resume", type=str, default="",
                        help="续训 checkpoint，一般填 last.pth")
    ##python .\train_dinov3_smmm.py --model converse_smmm --run_name run_smmm_converse_lora --resume D:\pycharm_projects\NOTRAING\NOTRAING\DINOv3seg\runs\run_smmm_converse_lora\last.pth
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr_lora", type=float, default=LR_LORA)
    parser.add_argument("--lr_head", type=float, default=LR_HEAD)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max_val_images", type=int, default=None,
                        help="调试用：验证时最多跑多少张；正式实验不要设置")

    parser.add_argument("--gn_groups", type=int, default=8,
                        help="SMMM 中 GroupNorm 最大 groups")
    parser.add_argument("--no_fusion_residual", action="store_true",
                        help="关闭 SMMMGroupFusionB 输出残差，默认开启")
    parser.add_argument("--no_converse_refine", action="store_true",
                        help="只对 converse_smmm 生效：关闭 Converse2D(scale=1) refine")

    parser.add_argument("--fp32", action="store_true",
                        help="backbone 使用 fp32；默认 cuda 下使用 bf16")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.fp32 else (
        torch.bfloat16 if (USE_BF16 and torch.cuda.is_available()) else torch.float32
    )

    print(f"device = {device}")
    print(f"dtype  = {dtype}")
    print(f"model  = {args.model}")
    print(f"LoRA   = {not args.no_lora}")

    if args.run_name:
        run_dir = OUTPUT_DIR / args.run_name
    else:
        run_dir = OUTPUT_DIR / make_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir = {run_dir}")

    config_snapshot = vars(args).copy()
    config_snapshot.update({
        "device": device,
        "dtype": str(dtype),
        "output_dir": str(run_dir),
        "fusion": "SMMM scheme B: low=SMMM(block5,block11), high=SMMM(block17,block23), out=SMMM(low,high)",
    })
    with open(run_dir / "config_smmm_train.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # ---- 数据 ----
    train_ds = SatelliteSegTrain()
    sampler = make_train_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    print(f"train pairs   = {len(train_ds)}")
    print(f"steps/epoch   = {steps_per_epoch}")
    print(f"total steps   = {total_steps}")
    print(f"warmup steps  = {warmup_steps}")

    # ---- 模型 ----
    model = build_model(args, device=device, dtype=dtype)
    keep_backbone_eval_if_nolora(model, args.no_lora)

    lora_params, decoder_params = split_params(model, args.no_lora)
    print(f"LoRA params       : {count_params(lora_params):,}")
    print(f"Decoder/SMMM params: {count_params(decoder_params):,}")

    if args.no_lora:
        optimizer = AdamW(
            [{"params": decoder_params, "lr": args.lr_head}],
            weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = AdamW(
            [
                {"params": lora_params, "lr": args.lr_lora},
                {"params": decoder_params, "lr": args.lr_head},
            ],
            weight_decay=WEIGHT_DECAY,
        )

    loss_fn = CEDiceLoss().to(device)

    start_epoch, best_miou, global_step, history = load_resume(
        model=model,
        optimizer=optimizer,
        resume_path=args.resume,
        device=device,
    )
    keep_backbone_eval_if_nolora(model, args.no_lora)

    patience_cnt = 0

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        keep_backbone_eval_if_nolora(model, args.no_lora)

        t_epoch = time.time()
        run_loss, run_ce, run_dice = 0.0, 0.0, 0.0
        run_cnt = 0

        for it, (img, msk) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)

            lr_scale = cosine_warmup(global_step, total_steps, warmup_steps)
            if args.no_lora:
                optimizer.param_groups[0]["lr"] = args.lr_head * lr_scale
            else:
                optimizer.param_groups[0]["lr"] = args.lr_lora * lr_scale
                optimizer.param_groups[1]["lr"] = args.lr_head * lr_scale

            logits = model(img)
            loss, parts = loss_fn(logits, msk)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            trainable_params = lora_params + decoder_params
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
            optimizer.step()

            run_loss += float(loss.item())
            run_ce += float(parts["ce"])
            run_dice += float(parts["dice"])
            run_cnt += 1
            global_step += 1

            if (it + 1) % LOG_EVERY == 0:
                if args.no_lora:
                    lr_msg = f"lr_head={optimizer.param_groups[0]['lr']:.2e}"
                else:
                    lr_msg = (
                        f"lr_lora={optimizer.param_groups[0]['lr']:.2e}  "
                        f"lr_head={optimizer.param_groups[1]['lr']:.2e}"
                    )
                print(
                    f"  ep{epoch} it{it+1}/{steps_per_epoch}  "
                    f"loss={run_loss/run_cnt:.4f}  "
                    f"ce={run_ce/run_cnt:.4f}  "
                    f"dice={run_dice/run_cnt:.4f}  "
                    f"{lr_msg}"
                )

        ep_loss = run_loss / max(1, run_cnt)
        print(
            f"[epoch {epoch}] train loss = {ep_loss:.4f}  "
            f"time = {time.time() - t_epoch:.1f}s"
        )

        # ---- 验证 ----
        if epoch % VAL_EVERY == 0:
            print(f"[epoch {epoch}] running val ...")
            keep_backbone_eval_if_nolora(model, args.no_lora)

            val_res = run_validation(
                model,
                device=device,
                dtype=dtype,
                max_images=args.max_val_images,
                verbose=True,
            )

            miou_fg = float(val_res["miou_fg"])
            miou_all = float(val_res["miou_all"])
            per_class = {k: float(v) for k, v in val_res["per_class_iou"].items()}

            print(f"  val mIoU (fg3) = {miou_fg:.4f}")
            print(f"  val mIoU (all4)= {miou_all:.4f}")
            for k, v in per_class.items():
                print(f"    {k:15s}: {v:.4f}")

            history.append({
                "epoch": int(epoch),
                "train_loss": float(ep_loss),
                "miou_fg": miou_fg,
                "miou_all": miou_all,
                "per_class": per_class,
                "lr_lora": float(optimizer.param_groups[0]["lr"]) if not args.no_lora else 0.0,
                "lr_head": float(optimizer.param_groups[-1]["lr"]),
            })

            extra_meta = {
                "arch": args.model,
                "no_lora": bool(args.no_lora),
                "fusion": "SMMM_B",
            }

            save_ckpt(
                model,
                run_dir / "last.pth",
                epoch=epoch,
                miou_fg=miou_fg,
                miou_all=miou_all,
                optimizer=optimizer,
                global_step=global_step,
                history=history,
                extra_meta=extra_meta,
            )

            if miou_fg > best_miou:
                best_miou = miou_fg
                patience_cnt = 0
                save_ckpt(
                    model,
                    run_dir / "best.pth",
                    epoch=epoch,
                    miou_fg=miou_fg,
                    miou_all=miou_all,
                    optimizer=optimizer,
                    global_step=global_step,
                    history=history,
                    extra_meta=extra_meta,
                )
                print(f"  ★ new best miou_fg = {best_miou:.4f}")
            else:
                patience_cnt += 1
                print(f"  patience {patience_cnt}/{args.patience}")

            with open(run_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"\nDone. best miou_fg = {best_miou:.4f}")
    print(f"Artifacts in: {run_dir}")


if __name__ == "__main__":
    main()

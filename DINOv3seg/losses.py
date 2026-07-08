"""
losses.py — CE (class-weighted) + Dice

Dice 对小目标（antenna）友好，不受像素比例影响
CE 提供像素级监督，配合类别权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, CE_WEIGHT, DICE_WEIGHT, CLASS_WEIGHTS


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss，忽略 ignore_index"""
    def __init__(self, num_classes=NUM_CLASSES, smooth=1.0,
                 ignore_index=-100, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer('cw', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.cw = None

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) long
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        # one-hot（处理 ignore）
        valid = (target != self.ignore_index)
        tgt = target.clone()
        tgt[~valid] = 0
        one_hot = F.one_hot(tgt, num_classes=C).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1).float()
        probs = probs * valid.unsqueeze(1).float()

        # 按类计算 dice
        dims = (0, 2, 3)  # 在 batch/H/W 上聚合
        inter = (probs * one_hot).sum(dim=dims)                       # (C,)
        card  = probs.sum(dim=dims) + one_hot.sum(dim=dims)           # (C,)
        dice_per_class = (2 * inter + self.smooth) / (card + self.smooth)   # (C,)

        loss_per_class = 1.0 - dice_per_class
        if self.cw is not None:
            loss_per_class = loss_per_class * self.cw.to(loss_per_class.device)
            return loss_per_class.sum() / self.cw.sum()
        return loss_per_class.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES,
                 class_weights=CLASS_WEIGHTS,
                 ce_w=CE_WEIGHT, dice_w=DICE_WEIGHT):
        super().__init__()
        cw = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=-100)
        self.dice = DiceLoss(num_classes=num_classes,
                             class_weights=class_weights)
        self.ce_w = ce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        l_ce = self.ce(logits, target)
        l_dice = self.dice(logits, target)
        return self.ce_w * l_ce + self.dice_w * l_dice, {
            'ce': l_ce.detach().item(),
            'dice': l_dice.detach().item(),
        }


if __name__ == "__main__":
    loss_fn = CEDiceLoss()
    logits = torch.randn(2, 4, 64, 64)
    target = torch.randint(0, 4, (2, 64, 64))
    loss, parts = loss_fn(logits, target)
    print(f"loss={loss.item():.4f}  ce={parts['ce']:.4f}  dice={parts['dice']:.4f}")

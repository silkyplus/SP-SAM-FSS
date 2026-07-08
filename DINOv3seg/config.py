"""
config.py — 集中配置
改这里就够，不用去动其他文件
"""
from pathlib import Path

# ==============================================================
# 路径
# ==============================================================
DATA_ROOT = r"D:\data\final_dataset\final_dataset"
TRAIN_IMG_DIR  = Path(DATA_ROOT) / "images" / "train"
TRAIN_MASK_DIR = Path(DATA_ROOT) / "masks"  / "train"
VAL_IMG_DIR    = Path(DATA_ROOT) / "images" / "val"
VAL_MASK_DIR   = Path(DATA_ROOT) / "masks"  / "val"

# DINOv3 仓库路径（含 dinov3/models/vision_transformer.py）
DINOV3_REPO = r"D:\pycharm_projects\NOTRAING\NOTRAING\dinov3_main"

# DINOv3 权重（卫星预训练，ViT-L/16）
DINOV3_WEIGHTS = r"D:\pycharm_projects\NOTRAING\NOTRAING\dinov3_weights\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DINOV3_ARCH    = "dinov3_vitl16"    # 用于 torch.hub.load

# 输出
OUTPUT_DIR = Path(r"D:\pycharm_projects\NOTRAING\NOTRAING\DINOv3seg\runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 类别
# ==============================================================
# 类别 index → 名称（0=background 必须放第一个）
CLASS_NAMES = ['background', 'body', 'solar_panel', 'antenna']
NUM_CLASSES = len(CLASS_NAMES)

# GT mask 颜色（RGB 顺序，因为 dataset.py 用 PIL 读）
RGB_TO_CLASS = {
    (0, 0, 0):     0,   # background
    (0, 255, 0):   1,   # body
    (255, 0, 0):   2,   # solar_panel
    (0, 0, 255):   3,   # antenna
}

# 可视化用（保存预测 mask 时）
CLASS_TO_RGB = {v: k for k, v in RGB_TO_CLASS.items()}


# ==============================================================
# 模型
# ==============================================================
# ViT-L/16 的特征维度
EMBED_DIM  = 1024
PATCH_SIZE = 16

# 从哪些 transformer block 抽多尺度特征（ViT-L 共 24 层，index 0~23）
# 抽 4 层做 FPN 融合
FPN_LAYERS = [5, 11, 17, 23]

# LoRA 配置
LORA_R       = 16        # rank
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
# 对哪些 Linear 层插 LoRA（DINOv3 ViT 的 attention 里一般叫这些名字）
# 最常见的是 qkv 合并矩阵和输出投影
LORA_TARGET_MODULES = ["qkv", "proj"]


# ==============================================================
# 数据 / 训练
# ==============================================================
# 训练输入尺寸（512x512 随机裁剪，保持原图分辨率，对细天线友好）
TRAIN_CROP = 512

# 验证用滑窗（与训练尺寸一致）
VAL_WINDOW = 512
VAL_STRIDE = 384         # overlap = 512-384 = 128

# ImageNet 统计（DINOv3 预训练用的）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 超参
BATCH_SIZE        = 4        # RTX 5060 Ti 16G, bf16, ViT-L, 512x512
NUM_WORKERS       = 2        # Windows 下可能 > 0 有问题，若卡住改 0
EPOCHS            = 50
PATIENCE          = 12       # val mIoU 多少轮不涨就停
LR_LORA           = 1e-4     # LoRA 参数的学习率
LR_HEAD           = 1e-3     # FPN 头的学习率
WEIGHT_DECAY      = 1e-4
WARMUP_FRAC       = 0.05     # 前 5% 的 step warmup
GRAD_CLIP         = 1.0

# 损失权重
CE_WEIGHT   = 1.0
DICE_WEIGHT = 1.0
# 类别权重（见 losses.py 里的解释）
# 手动给 antenna 更高权重，其他保持低
CLASS_WEIGHTS = [0.3, 1.0, 1.0, 5.0]   # [bg, body, solar, antenna]

# 采样策略：含 antenna 的图在一个 epoch 里被抽到的概率倍数
ANTENNA_OVERSAMPLE = 3.0

# AMP
USE_BF16 = True        # 5060 Ti 支持 bf16


# ==============================================================
# 杂项
# ==============================================================
SEED = 42
LOG_EVERY = 20         # 多少 step 打一次 loss
VAL_EVERY = 1          # 多少 epoch eval 一次

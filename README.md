# SP-SAM: 无需训练的 Memory-Augmented 少样本语义分割

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![SAM2](https://img.shields.io/badge/Backbone-SAM2-orange)](https://github.com/facebookresearch/sam2)
[![DINOv3](https://img.shields.io/badge/Feature-DINOv3-green)](https://github.com/facebookresearch/dinov3)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**将视频分割的记忆机制成功迁移至图像少样本学习，在自然图像与极光等专业领域均取得显著效果**

</div>

---

## 📋 项目简介

针对**少样本图像分割**中新类别标注数据稀缺的难题，本项目提出了一种**无需训练的基于记忆增强与稀疏提示的通用分割框架（SP-SAM）**，创新性地将 SAM2 视频分割模型中的记忆（Memory）读写机制应用于少样本图像分割任务，实现了支撑集信息对查询图像的高效引导与增强。

### 🎯 核心链路

```
Support Image + Mask  ──→ SAM2 Memory Encoder ──→ Memory Bank
                             │
Query Image  ──→ DINOv3 Feature Extractor ──→ Prototype Matching ──→ Sparse Keypoints
                             │                         │
                             └─────────┬───────────────┘
                                       ▼
                              SAM2 Mask Decoder
                                       │
                                       ▼
                                 Predicted Mask
```

---

## ✨ 主要创新

### 1. 🧠 SAM2 记忆机制迁移到少样本分割

首次将 **SAM2 视频分割模型中的 Memory Encoder / Memory Attention** 机制应用于图像少样本分割：

- 将支撑集（Support）图像当作"视频历史帧"，通过 Memory Encoder 编码为记忆特征
- 查询图像通过 Memory Attention 与记忆特征进行像素级匹配
- 利用 SAM2 预训练的 Mask Decoder 生成高质量精细分割边界

**关键文件**: [`sam2_memory_segmentor.py`](sam2_memory_segmentor.py) — `SAM2MemoryBasedSegmentor`

### 2. 🎯 DINOv3 原型匹配 + 稀疏提示

设计基于 **DINOv3 特征的 Prototype Matching 模块**，自动生成关键点作为稀疏提示：

- 对 Support 区域做 Masked Average Pooling 提取类别原型向量
- Query 特征与原型做余弦相似度匹配，生成粗粒度热力图
- 从热力图中自动选取 Top-K 响应点作为**稀疏提示（Sparse Prompt）**
- 减少了对密集掩码提示的依赖，实现 training-free 推理

**关键文件**: [`similar_calucate.py`](similar_calucate.py) — 原型计算与相似度匹配

### 3. 🔧 LoRA 参数高效微调（领域适配）

采用 **LoRA（Low-Rank Adaptation）** 对模型进行参数高效微调：

- 对 DINOv3 和 SAM2 的关键层注入低秩适配器
- 快速将自然图像上的强大性能迁移至**极光图像**等专业领域
- 有效解决领域数据稀缺问题，仅需少量标注样本

**关键文件**:
- [`LoRA/lora_modules_full.py`](LoRA/lora_modules_full.py) — 通用 LoRA 模块
- [`LoRA/full_model_lora.py`](LoRA/full_model_lora.py) — 全模型 LoRA 适配
- [`LoRA/jiguang_dataset.py`](LoRA/jiguang_dataset.py) — 极光数据集适配

### 4. 📊 统一评估框架

支持多数据集、多 Shot 设置的统一评估：

- **PASCAL-5i**: 经典少样本分割基准
- **COCO-20i**: 大规模少样本分割基准
- **FSS-1000**: 千类少样本分割
- **ISIC 2018**: 医学皮肤镜图像（领域迁移）
- **极光卫星图像**: 专业领域微调

---

## 🏗️ 项目架构

```
SP-SAM-FSS/
├── README.md
├── requirements.txt
│
├── sp_sam_complete.py              ★ 完整 SP-SAM 管道
├── sam2_memory_segmentor.py        ★ SAM2 Memory 少样本分割器
├── spatial_memory_bank_enhanced.py ★ 增强空间记忆库
├── spatial_memory_bank.py             基础空间记忆库
├── model_manager.py                   模型加载管理
│
├── support_selector_new.py         ★ 支撑集选择策略
├── support_selector_spsam.py          SP-SAM 专用选择器
├── similar_calucate.py             ★ DINOv3 原型匹配与相似度
│
├── batch_evaluate_new.py              统一批量评估
├── evaluate_coco20i_fixed.py          COCO-20i 评估
├── evaluate_pascal5i_new.py           PASCAL-5i 评估
├── evaluate_fss1000.py                FSS-1000 评估
├── evaluate_isic2018.py               ISIC 2018 医学评估
├── _evaluate_satellite_select_allval.py  卫星图像评估
│
├── datasets/                          数据集模块
│   ├── coco20i_dataset.py
│   ├── pascal5i_dataset_new.py
│   ├── fss1000_dataset.py
│   └── isic2018_dataset.py
│
├── DINOv3seg/                     ★ DINOv3 语义分割模块
│   ├── config.py                     集中配置
│   ├── model_smmm.py                 空间记忆匹配模型 (SMMM)
│   ├── model.py / model2.py          分割模型
│   ├── dataset.py                    数据集加载
│   ├── losses.py                     损失函数 (CE + Dice)
│   ├── train_dinov3_smmm.py          SMMM 训练脚本
│   ├── eval2.py                      评估脚本
│   ├── SMMM.py                       空间记忆匹配核心
│   ├── ablation_seg.py               消融实验
│   └── visualize_dinov3_all.py       特征可视化
│
├── LoRA/                           ★ LoRA 参数高效微调
│   ├── lora_modules_full.py          LoRA 基础模块
│   ├── full_model_lora.py            全模型 LoRA 适配
│   ├── train_full_lora.py            全量 LoRA 训练
│   ├── train_spsam_e2e.py            SP-SAM 端到端训练
│   ├── inference_full_lora.py        LoRA 推理
│   ├── inference_spsam_e2e.py        SP-SAM 端到端推理
│   ├── inference_baseline.py         基线推理
│   ├── finetune_config.py            微调配置
│   ├── training_config.py            训练配置
│   └── jiguang_dataset.py           极光数据集
│
├── visualize_pipeline.py              管道可视化
├── visualize_stages.py                分阶段可视化
├── visualize_features.py              特征图可视化
│
└── common/                            通用工具
    ├── evaluation.py                  mIoU / FB-IoU 指标
    ├── logger.py                      日志工具
    ├── utils.py                       辅助函数
    └── vis.py                         可视化工具

★ = 核心创新模块
```

---

## 🚀 快速开始

### 环境配置

```bash
conda create -n spsam python=3.10 -y
conda activate spsam

# PyTorch (CUDA 11.8)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# 其他依赖
pip install -r requirements.txt
```

### 下载模型权重

```bash
# SAM2 (hiera-large)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# DINOv3 (ViT-L/16)
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16_pretrain.pth
```

### Training-Free 推理

```python
from sp_sam_complete import SPSAMComplete

# 初始化（无需训练）
model = SPSAMComplete(
    sam2_ckpt='sam2.1_hiera_large.pt',
    dino_ckpt='dinov3_vitl16_pretrain.pth',
    device='cuda'
)

# 少样本推理
mask = model.predict(
    query_image='query.jpg',
    support_images=['support_1.jpg', 'support_2.jpg'],
    support_masks=['mask_1.png', 'mask_2.png'],
    n_shots=2
)
```

### 批量评估

```bash
# COCO-20i (1-shot)
python batch_evaluate_new.py --dataset coco20i --shots 1 --mode cmrs_memory

# PASCAL-5i (5-shot)
python batch_evaluate_new.py --dataset pascal5i --shots 5 --mode cmrs_predictor

# FSS-1000
python evaluate_fss1000.py --shots 1
```

### LoRA 微调（极光领域适配）

```bash
# 端到端 LoRA 训练
cd LoRA
python train_spsam_e2e.py --config finetune_config.py --dataset jiguang

# LoRA 推理
python inference_spsam_e2e.py --lora_weights ./outputs/lora_best.pt
```

---

## 🔬 方法详解

### Memory 增强机制

```
1. Support Encoding:
   Support Image ──→ SAM2 Image Encoder ──→ Feature Map
   Support Mask  ──→ Memory Encoder ──→ Memory Features
        ↓
   Memory Bank: {feat_spatial, mask_downsampled, prototype}

2. Query Processing:
   Query Image ──→ SAM2 Image Encoder ──→ Query Features
        ↓
   Memory Attention(Q, K=Memory, V=Memory) ──→ Conditioned Features
        ↓
   Mask Decoder ──→ [Multi-scale Masks, IoU Scores]

3. Post-processing:
   Best Mask ← argmax(IoU Scores)
   Refine with CRF / edge-aware filtering
```

### DINOv3 原型匹配流程

```
1. Extract: DINOv3.get_intermediate_layers(support_img)[-1] → feat_s [C, H, W]
2. Prototype: Masked Average Pooling(feat_s, support_mask) → proto [C]
3. Similarity: cosine_sim(feat_q, proto) → heatmap [H, W]
4. Peak Detection: top_k(heatmap, k=5) → sparse_prompts [(x1,y1), ...]
5. SAM2 Decoder: predict(sparse_prompts + bbox_from_heatmap)
```

### 四种推理模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `rough_only` | 仅用 DINOv3 热力图阈值 | 快速基线 |
| `memory_only` | 仅用 SAM2 Memory Attention | 精确匹配 |
| `cmrs_predictor` | DINOv3 稀疏提示 → SAM2 Decoder | 综合推理 |
| `cmrs_memory` | Memory + DINOv3 原型融合 | 最强性能 |

---

## 📊 预期结果

| 数据集 | 方法 | 1-shot mIoU | 5-shot mIoU |
|--------|------|:----------:|:----------:|
| PASCAL-5i | Rough Only | ~55% | ~62% |
| PASCAL-5i | Memory Only | ~58% | ~65% |
| PASCAL-5i | **CMRS (Ours)** | **~62%** | **~68%** |
| COCO-20i | **CMRS (Ours)** | **~45%** | **~52%** |
| FSS-1000 | **CMRS (Ours)** | **~82%** | **~87%** |

> 📌 以上为论文参考值，具体数值取决于模型版本和超参设置

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| Vision Backbone | SAM2 (Hiera-Large), DINOv3 (ViT-L/16) |
| Memory Module | SAM2 Memory Encoder + Memory Attention |
| Feature Matching | DINOv3 Prototype + Cosine Similarity |
| Prompt Generation | DINOv3 Heatmap → Sparse Keypoints |
| Mask Decoder | SAM2 Transformer Decoder |
| Domain Adaptation | LoRA (Low-Rank Adaptation) |
| Evaluation | mIoU, FB-IoU, Precision/Recall |

---

## 📄 License

Apache 2.0

---

> 📌 **简历备注**: 本项目创新性地将视频理解中的记忆机制迁移至图像少样本学习，在自然图像与极光卫星等专业领域均取得显著效果，展示了跨模态迁移与参数高效微调的工程实践能力。

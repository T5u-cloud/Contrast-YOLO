# Contrast-YOLO
**Dual-Box Contrastive Learning for Fine-Grained Power Tower Detection in Remote Sensing Imagery**

**面向遥感图像输电塔细粒度检测的双框对比学习方法**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)
[![Built on](https://img.shields.io/badge/Built%20on-Ultralytics%20YOLO11-purple.svg)](https://github.com/ultralytics/ultralytics)

> A training-time auxiliary branch that teaches a YOLO detector to disentangle
> tower body from cast shadow, at **zero inference cost**.
>
> 一个训练时挂载的辅助分支，让 YOLO 检测器学会区分塔身与阴影特征——
> **推理阶段零额外开销**。

---

## 🎯 Overview / 项目概述

Fine-grained classification of power transmission towers in remote-sensing
imagery is challenging because the tower body alone is often visually
ambiguous, while the **cast shadow encodes critical shape cues**. Standard
single-box annotation entangles these two signals, making it hard for the
network to learn stable type-discriminative features.

在遥感图像中对输电塔进行细粒度分类（区分伞型、酒杯型、T 型、V 型、门型、其他）
存在挑战：塔身本身在俯视图下纹理相似，**塔身投下的阴影才编码了关键的形态线索**。
而常规的单框标注把"塔身"和"阴影"揉进同一个 bounding box，
让网络难以学到稳定的塔型判别特征。

This project proposes:
本项目提出：

1. **A dual-box annotation scheme** — every tower instance is annotated with
   two boxes: an *Overall* box (tower body + shadow) and a *Body* box
   (tower body only).

   **双框标注方案** —— 每个塔实例标注两个框：
   *Overall*（塔身 + 阴影的最小外接矩形）和 *Body*（仅塔身的最小外接矩形）。

2. **A training-only contrastive auxiliary branch** — built on top of YOLO11,
   it pairs Overall–Body boxes via an IoA-based matching algorithm, extracts
   RoI features from the P3 layer, and applies three supervised contrastive
   losses to shape the feature space along tower-type semantics.

   **训练阶段挂载的对比学习辅助分支** —— 在 YOLO11 之上，
   通过基于 IoA 的配对算法关联 Overall 与 Body 框，
   从 P3 层提取 RoI 特征，施加三路 SupCon 损失，
   将特征空间按塔型语义重新组织。

3. **Zero deployment overhead** — the auxiliary branch is bypassed at
   inference, so the deployed model is structurally identical to vanilla
   YOLO11n.

   **零部署开销** —— 推理时辅助分支被旁路，部署模型与原始 YOLO11n 完全等价。

---

## 🔑 Key Contributions / 核心贡献

- **Dual-box annotation** explicitly separates the body and shadow signals
  while preserving their physical relationship.
  **双框标注**显式分离塔身与阴影信号，同时保留两者的物理关联。

- **IoA-based greedy pairing** (Intersection-over-Body-Area) robustly matches
  Overall–Body pairs even under crowded scenes — more reliable than IoU for
  this asymmetric containment relationship.
  **基于 IoA 的贪心配对**：以 Body 面积为分母的交并比，
  在密集场景下稳定匹配 Overall–Body 对——
  比传统 IoU 更适合这种非对称包含关系。

- **Three-stream contrastive loss** (Body / Diff / Pair) jointly regularizes
  the body feature, the learned body–shadow differential feature, and their
  joint representation.
  **三路对比损失**（Body / Diff / Pair）联合约束：
  塔身特征、差异特征 F_diff、以及两者的联合表示。

- **Training-time only** — auxiliary branch adds no inference FLOPs and no
  deployment parameters.
  **仅训练阶段**——辅助分支不增加推理 FLOPs，不增加部署参数。

---

## 📊 Results / 实验结果

Evaluated on a self-collected dataset of **1,935 remote-sensing images**
(test split: 291 images / 1,136 instances).
在自建的 **1,935 张遥感图像**数据集上评估（test split：291 张 / 1,136 个实例）。

### Main Results / 主要结果

| Metric             | YOLO11n Baseline | Contrast-YOLO (Ours) | Δ          |
|:-------------------|-----------------:|---------------------:|:----------:|
| **mAP@0.5**        | 0.6174           | **0.6365**           | **+1.91%** |
| **mAP@0.5:0.95**   | 0.3199           | **0.3334**           | +1.36%     |
| **Precision**      | 0.6080           | **0.7376**           | **+12.96%**|
| Recall             | 0.6041           | 0.5908               | -1.33%     |

> The large Precision gain with marginal Recall loss indicates that the
> contrastive supervision sharpens decision boundaries between tower types,
> substantially reducing false positives.
>
> Precision 大幅提升而 Recall 仅轻微下降，说明对比监督
> 锐化了塔型之间的判别边界，**显著抑制了误检**。

### Overall vs Body Decomposition / 双框拆分

| Avg. mAP@0.5 | Baseline | Ours   | Δ      |
|:-------------|---------:|-------:|:------:|
| Overall box  | 0.6282   | 0.6566 | +2.84% |
| Body box     | 0.6066   | 0.6164 | +0.98% |

### Per-Class mAP@0.5 / 分类别结果

| Class             | Baseline | Ours   | Δ       |
|:------------------|---------:|-------:|:-------:|
| Umbrella_Overall  | 0.7631   | 0.7751 | +1.21%  |
| Umbrella_Body     | 0.7521   | 0.7859 | +3.38%  |
| Wineglass_Overall | 0.7096   | 0.7413 | +3.17%  |
| Wineglass_Body    | 0.6804   | 0.7084 | +2.79%  |
| T_Overall         | 0.7602   | 0.7919 | +3.18%  |
| T_Body            | 0.8704   | 0.8351 | -3.53%  |
| V_Overall         | 0.8667   | 0.8558 | -1.09%  |
| V_Body            | 0.7938   | 0.8440 | +5.02%  |
| Gate_Overall      | 0.2553   | 0.2601 | +0.48%  |
| Gate_Body         | 0.1866   | 0.1959 | +0.94%  |
| Other_Overall     | 0.4142   | 0.5155 | +10.13% |
| Other_Body        | 0.3564   | 0.3290 | -2.73%  |

> Numbers above are from a **single training seed**. Multi-seed validation
> and full ablation studies are in progress.
>
> 以上为**单次训练种子**的结果。多种子验证与完整消融实验正在进行中。

---

## 🧠 Method / 方法

```
                     ┌──────────────────────────────┐
                     │       YOLO11n Backbone       │
                     └──────────────┬───────────────┘
                                    │
                                    ▼  P3 feature map
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
   ┌────────────────┐                          ┌────────────────────────────┐
   │ Detection Head │                          │  Contrastive Aux Branch    │
   │  (box/cls/dfl) │                          │  (training only)           │
   └────────────────┘                          │                            │
                                               │  ┌──────────────────────┐  │
                                               │  │ IoA-based pairing    │  │
                                               │  │ Overall ↔ Body       │  │
                                               │  └──────────┬───────────┘  │
                                               │             ▼              │
                                               │  ┌──────────────────────┐  │
                                               │  │ RoIAlign on P3       │  │
                                               │  │ → F_body, F_overall  │  │
                                               │  └──────────┬───────────┘  │
                                               │             ▼              │
                                               │  ┌──────────────────────┐  │
                                               │  │ DiffEncoder (MLP)    │  │
                                               │  │ → F_diff             │  │
                                               │  └──────────┬───────────┘  │
                                               │             ▼              │
                                               │  ┌──────────────────────┐  │
                                               │  │ SupCon × 3:          │  │
                                               │  │   L_body, L_diff,    │  │
                                               │  │   L_pair             │  │
                                               │  └──────────────────────┘  │
                                               └────────────────────────────┘
```
<img width="4143" height="2270" alt="网络流程图" src="https://github.com/user-attachments/assets/79271bff-945b-4162-b684-4feb1bf14cc8" />

**Total loss / 总损失:**


```
L_total = L_yolo  +  λ_body · L_body  +  λ_diff · L_diff  +  λ_pair · L_pair
```

with `λ_body = 0.1`, `λ_diff = 0.1`, `λ_pair = 0.05`, temperature `τ = 0.07`.

Full implementation: [`src/contrastive.py`](src/contrastive.py)
Integration patch:   [`src/loss_patch.md`](src/loss_patch.md)

---

## 📁 Repository Structure / 仓库结构

```
Contrast-YOLO/
├── README.md                  # This file / 本文件
├── LICENSE                    # AGPL-3.0
├── .gitignore
├── requirements.txt           # Python dependencies / 依赖
│
├── src/                       # ★ Core contribution / 核心贡献
│   ├── contrastive.py         #   Contrastive aux branch implementation
│   │                          #   对比学习辅助分支实现
│   └── loss_patch.md          #   Patch notes for ultralytics/utils/loss.py
│                              #   对 loss.py 的改动说明
│
├── configs/
│   └── tower.yaml             # Dataset config / 数据集配置
│
├── scripts/
│   ├── train.py               # Training entry / 训练入口
│   └── eval.py                # Evaluation entry / 评估入口
│
├── samples/
│   ├── labels/                # 3 example label files / 3 个示例标注
│   └── README.md              # Annotation format / 标注格式说明
│
└── results/                   # Reported metrics / 结果记录
```

---

## 🛠️ Setup & Usage / 环境与使用

### 1. Install / 安装

```bash
# Clone this repo / 克隆仓库
git clone https://github.com/T5u-cloud/Contrast-YOLO.git
cd Contrast-YOLO

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

This project is built on top of **Ultralytics YOLO11**. The files in `src/`
must be placed under `ultralytics/utils/` of an Ultralytics installation,
and `loss.py` patched as described in [`src/loss_patch.md`](src/loss_patch.md).

本项目基于 **Ultralytics YOLO11** 构建。`src/` 中的文件需要放置到
Ultralytics 安装目录的 `ultralytics/utils/` 下，
并按 [`src/loss_patch.md`](src/loss_patch.md) 中说明对 `loss.py` 进行 patch。

### 2. Prepare data / 准备数据

The dataset is **not redistributed** (see Dataset section below). To
reproduce, you need your own tower dataset in YOLO format following the
annotation scheme described in [`samples/README.md`](samples/README.md).

数据集**不随仓库分发**（见下方 Dataset 章节）。如需复现，
请按 [`samples/README.md`](samples/README.md) 中的标注规范
准备自己的 YOLO 格式输电塔数据集。

### 3. Train / 训练

```bash
# Baseline (set USE_CONTRASTIVE = False in scripts/train.py)
# Baseline（在 scripts/train.py 中设 USE_CONTRASTIVE = False）
python scripts/train.py

# Our method (set USE_CONTRASTIVE = True)
# 本文方法（设 USE_CONTRASTIVE = True）
python scripts/train.py
```

### 4. Evaluate / 评估

```bash
python scripts/eval.py --weights runs/tower/contrastive_v1/weights/best.pt \
                       --data configs/tower.yaml \
                       --split test
```

---

## 📦 Dataset / 数据集

| Split | Images | Instances |
|:------|-------:|----------:|
| Train | 1,354  | 5,178     |
| Val   | 290    | 1,098     |
| Test  | 291    | 1,136     |
| **Total** | **1,935** | **7,412** |

**6 tower types × 2 box types = 12 classes**, with long-tailed distribution
(Gate and T classes have notably fewer instances).

**6 种塔型 × 2 种框 = 12 类**，长尾分布（Gate 和 T 类样本数显著较少）。

### Data source / 数据来源

Images were **self-collected from Google Earth imagery** for academic
research purposes only. Due to Google Earth's terms of service and image
copyright, **the dataset is not redistributed** in this public repository.

图像由作者**自行从 Google Earth 影像中采集**，仅用于学术研究。
受 Google Earth 服务条款与影像版权约束，
**数据集不随本公开仓库分发**。

See [`samples/`](samples/) for the annotation format.
标注格式见 [`samples/`](samples/)。

---

## 🙏 Acknowledgements / 致谢

This project is built on the excellent
[**Ultralytics YOLO11**](https://github.com/ultralytics/ultralytics)
framework, licensed under AGPL-3.0. The following files are modified from
or added to the upstream codebase:

本项目基于优秀的
[**Ultralytics YOLO11**](https://github.com/ultralytics/ultralytics)
框架构建（AGPL-3.0 许可）。相对于上游代码，本项目修改 / 新增了以下文件：

- `ultralytics/utils/contrastive.py` — **new** / **新增**
- `ultralytics/utils/loss.py`        — modified (see `src/loss_patch.md`)
                                       已修改（详见 `src/loss_patch.md`）

The supervised contrastive loss follows
**Khosla et al., "Supervised Contrastive Learning"**, NeurIPS 2020.
有监督对比损失参考
**Khosla et al., "Supervised Contrastive Learning"**, NeurIPS 2020。

---

## 📄 License / 许可

This project is licensed under **AGPL-3.0**, consistent with the upstream
Ultralytics YOLO11 license. See [LICENSE](LICENSE) for details.

本项目采用 **AGPL-3.0** 许可，与上游 Ultralytics YOLO11 一致。
详见 [LICENSE](LICENSE) 文件。

---

## 👤 Author / 作者

**Su Xuanhao / 苏煊皓**
College of Land Science and Technology, China Agricultural University
中国农业大学 土地科学与技术学院

---

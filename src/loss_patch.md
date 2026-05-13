# Patch to `ultralytics/utils/loss.py`
# 对 `ultralytics/utils/loss.py` 的改动说明

This file documents the changes made to `ultralytics/utils/loss.py` to integrate
the contrastive auxiliary branch. The full modified file is not included here
because it is mostly unchanged from upstream Ultralytics YOLO11 code (AGPL-3.0).

本文件记录了为接入对比学习辅助分支而对 `ultralytics/utils/loss.py` 所做的改动。
完整文件未上传，因为绝大部分代码源自 Ultralytics YOLO11（AGPL-3.0），与上游一致。

---

## Changes to `class v8DetectionLoss` / `v8DetectionLoss` 类的改动

### 1. In `__init__` / 在 `__init__` 中

Add the contrastive module after the existing initialization:
在原有初始化代码之后，添加对比学习模块：

```python
# ★ Contrastive learning module (training-only)
# ★ 对比学习模块（仅训练阶段使用）
self.contrastive = PairContrastiveLoss(
    in_channels=256,       # P3 channel count (256 for YOLO11n)
                           # P3 层通道数（YOLO11n 为 256）
    temperature=0.07,
    lambda_body=0.1,
    lambda_diff=0.1,
    lambda_pair=0.05,
).to(device)

# Cache for P3 feature, filled externally during forward
# P3 特征缓存，由 forward hook 外部填充
self.p3_cache = None
```

### 2. In `__call__` / 在 `__call__` 中

After the standard YOLO loss computation (box, cls, dfl), before returning,
add the contrastive loss:

在标准的 YOLO 三路 loss（box / cls / dfl）计算完成之后、return 之前，
追加对比学习 loss：

```python
# ★ Add contrastive loss (training-only, gated by global switch)
# ★ 追加对比学习 loss（仅训练阶段，受全局开关控制）
contrast_loss = torch.tensor(0.0, device=self.device)
if USE_CONTRASTIVE_LOSS and self.p3_cache is not None:
    try:
        contrast_loss, _ = self.contrastive(self.p3_cache, batch)
    except Exception as e:
        print(f"[WARN] Contrastive loss failed: {e}")
        contrast_loss = torch.tensor(0.0, device=self.device)
    self.p3_cache = None  # 用完即清，防止跨 iteration 串味

total_loss = loss.sum() * batch_size + contrast_loss
return total_loss, loss.detach()
```

### 3. P3 cache hook / P3 特征捕获 hook

The P3 feature is captured during the forward pass via a hook registered on
the detection head's P3 input branch. The hook writes to `loss_fn.p3_cache`
before `__call__` is invoked.

在 forward 过程中，通过注册在检测头 P3 输入分支上的 hook 来捕获 P3 特征。
hook 会在 `__call__` 被调用之前将特征写入 `loss_fn.p3_cache`。

(The hook registration logic lives in the training entry script.)
（hook 的注册逻辑位于训练入口脚本中。）

---

## Design Notes / 设计说明

- The contrastive branch is gated by `USE_CONTRASTIVE_LOSS`, a global flag set
  in the training entry script. This allows the same codebase to run both
  baseline and contrastive experiments by toggling a single switch.

  对比学习分支由 `USE_CONTRASTIVE_LOSS` 全局开关控制，开关在训练入口脚本中设置。
  通过切换一个开关即可在同一份代码上跑 baseline 和 contrastive 两组实验，
  保证实验对比的严格一致性。

- The `p3_cache` is set to `None` immediately after consumption to prevent
  stale features from leaking across iterations.

  `p3_cache` 在使用后立即置 `None`，避免跨 iteration 残留导致特征污染。

- At inference time, `loss_fn` is not invoked, so the contrastive branch is
  completely bypassed — no additional FLOPs or parameters at deployment.

  推理阶段不会调用 `loss_fn`，对比学习分支被完全旁路，
  部署时不增加任何 FLOPs 和参数。
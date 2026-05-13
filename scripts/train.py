# ===========================================================================
# Training Entry Script for Tower Detection
# 输电塔检测训练入口脚本
#
# This script trains the YOLO11 detector with an optional contrastive
# auxiliary branch. The contrastive branch is enabled or disabled via a
# single global switch (USE_CONTRASTIVE), which makes baseline / ablation
# comparisons strictly reproducible from the same codebase.
#
# 本脚本基于 YOLO11 训练输电塔检测器，可选挂载对比学习辅助分支。
# 对比学习分支通过一个全局开关 USE_CONTRASTIVE 控制启停，
# 这样 baseline 与对比学习实验可以从同一份代码直接复现，保证严格可比。
#
# Usage / 使用方法:
#   - Set USE_CONTRASTIVE = False to train baseline
#     设 USE_CONTRASTIVE = False 训练 baseline
#   - Set USE_CONTRASTIVE = True  to train contrastive variant
#     设 USE_CONTRASTIVE = True  训练对比学习版本
# ===========================================================================

from ultralytics import YOLO
from ultralytics.utils import loss as loss_module


# ---------------------------------------------------------------------------
# Experiment switch / 实验开关
# ---------------------------------------------------------------------------
# True  = with contrastive auxiliary branch (our method)
#         启用对比学习辅助分支（本文方法）
# False = vanilla YOLO11 baseline
#         不启用，纯 YOLO11 baseline
USE_CONTRASTIVE = False


# ---------------------------------------------------------------------------
# P3 feature hook / P3 特征捕获 hook
# ---------------------------------------------------------------------------
# Registers a forward hook on the backbone's P3 layer to capture its output
# and cache it into the loss function. The contrastive loss reads from this
# cache to extract RoI features during loss computation.
#
# 在 backbone 的 P3 层上注册一个 forward hook，捕获其输出特征图，
# 并缓存到 loss 函数的 p3_cache 字段。对比学习分支在计算 loss 时
# 从这个缓存里读取 P3 特征，用 RoIAlign 提取每个 bbox 的区域特征。
# ---------------------------------------------------------------------------
def setup_p3_hook(model_wrapper):
    """Register a forward hook to capture P3 features into loss_fn.p3_cache.

    在 P3 层注册 forward hook，把特征写入 loss_fn.p3_cache 中。
    """
    actual_model = model_wrapper.model

    # YOLO11 backbone: P3 output sits at model[4] (after the second C3k2 block).
    # YOLO11 主干网络中，P3 的输出位于 model[4]（第二个 C3k2 之后）。
    P3_LAYER_INDEX = 4
    target_layer = actual_model.model[P3_LAYER_INDEX]

    def hook(module, input, output):
        # criterion is created lazily by ultralytics on first forward.
        # criterion 由 ultralytics 在首次 forward 时延迟创建，故需判空。
        if hasattr(actual_model, 'criterion') and actual_model.criterion is not None:
            actual_model.criterion.p3_cache = output

    handle = target_layer.register_forward_hook(hook)
    return handle


# ---------------------------------------------------------------------------
# Main entry / 主入口
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # Propagate the switch into the loss module so v8DetectionLoss can read it
    # at runtime. Must be done BEFORE model construction.
    # 把开关传给 loss 模块，让 v8DetectionLoss 在运行时能读到。
    # 必须在构建模型之前完成。
    loss_module.USE_CONTRASTIVE_LOSS = USE_CONTRASTIVE
    print(f">>> USE_CONTRASTIVE = {USE_CONTRASTIVE}")

    # Build YOLO11n model from scratch config (weights loaded via `pretrained`).
    # 从配置文件构建 YOLO11n（预训练权重通过 pretrained 参数加载）。
    model = YOLO("yolo11n.yaml")

    # Register P3 hook only when contrastive branch is enabled.
    # 仅在启用对比学习时才注册 P3 hook，避免 baseline 跑额外逻辑。
    if USE_CONTRASTIVE:
        print(">>> Contrastive branch ENABLED / 对比学习分支已启用")
        hook_handle = setup_p3_hook(model)
    else:
        print(">>> Baseline mode / Baseline 模式")
        hook_handle = None

    # Experiment naming follows the switch so runs/ folders don't collide.
    # 实验名跟开关绑定，避免 baseline 和 contrastive 的输出目录互相覆盖。
    exp_name = "contrastive_v1" if USE_CONTRASTIVE else "baseline1"

    # ---------------------------------------------------------------------
    # Training / 训练
    # ---------------------------------------------------------------------
    # Note on hyperparameters / 超参说明:
    #   - epochs=200, patience=50: long schedule with early stopping
    #     长训练 + early stopping
    #   - imgsz=608: input resolution suited to remote-sensing imagery
    #     针对遥感图像设定的输入分辨率
    #   - batch=32: tuned for RTX 4060 8GB
    #     按 RTX 4060 8GB 显存调过
    #   - cache=True: cache decoded images in RAM to speed up data loading
    #     缓存解码后的图像加快数据加载
    #   - deterministic=False: allows cudnn benchmark for speed
    #     允许 cudnn benchmark，训练更快（牺牲严格可复现性）
    # ---------------------------------------------------------------------
    results = model.train(
        data="ultralytics/cfg/datasets/tower.yaml",
        epochs=200,
        batch=32,
        imgsz=608,
        workers=0,
        device="0",
        pretrained="weights/yolo11n.pt",
        cache=True,
        deterministic=False,
        patience=50,
        name=exp_name,
        project="runs/tower",
    )

    # Clean up the hook to avoid dangling references / 清理 hook 引用
    if hook_handle is not None:
        hook_handle.remove()
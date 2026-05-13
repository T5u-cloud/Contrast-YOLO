# ===========================================================================
# Evaluation Entry Script
# 评估入口脚本
#
# Minimal evaluation script: loads a trained checkpoint and reports
# COCO-style detection metrics (mAP@0.5, mAP@0.5:0.95, P, R) on the
# specified split.
#
# 最小评估脚本：加载训练好的权重，在指定 split 上输出
# COCO 风格的检测指标（mAP@0.5、mAP@0.5:0.95、Precision、Recall）。
#
# Usage / 使用方法:
#   python scripts/eval.py --weights runs/tower/baseline1/weights/best.pt
#   python scripts/eval.py --weights runs/tower/contrastive_v1/weights/best.pt \
#                          --split test
# ===========================================================================

import argparse
from ultralytics import YOLO


def evaluate(weights: str, data: str, split: str = "test"):
    """Run validation on the given checkpoint and split.

    Args:
        weights: path to a .pt checkpoint
                 模型权重路径（.pt 文件）
        data:    path to dataset YAML
                 数据集配置 YAML 路径
        split:   'val' or 'test'
                 评估子集
    """
    model = YOLO(weights)
    metrics = model.val(
        data=data,
        split=split,
        save_json=True,
        plots=True,
        verbose=True,
    )

    # Summary table / 汇总指标
    print("\n" + "=" * 50)
    print(f"  Eval Summary  |  split = {split}")
    print("=" * 50)
    print(f"  mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to .pt checkpoint / 模型权重路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/tower.yaml",
        help="Path to dataset YAML / 数据集配置路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Evaluation split / 评估子集",
    )
    args = parser.parse_args()

    evaluate(args.weights, args.data, args.split)
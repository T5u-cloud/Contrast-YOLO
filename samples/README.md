# Dataset Samples / 数据集格式示例

This folder shows the **label format** of the tower detection dataset.
Image files are **not included** because the dataset was sourced from
Google Earth imagery and is bound by its terms of use.

本目录展示输电塔检测数据集的**标注格式**。
图像文件**未上传**——数据采自 Google Earth 影像，受其使用条款约束。

---

## Label Format / 标注格式

Each label file follows the **YOLO format**, one bounding box per line:

每个 label 文件遵循 **YOLO 格式**，每行一个 bounding box：

```
<class_id> <cx> <cy> <w> <h>
```

- `class_id` — integer in [0, 11]
                整数，范围 [0, 11]
- `cx, cy`   — box center, normalized to [0, 1]
                框中心点，归一化到 [0, 1]
- `w, h`     — box width / height, normalized to [0, 1]
                框宽高，归一化到 [0, 1]

---

## Class ID Encoding / 类别编码

```
class_id = physical_type * 2 + box_type

physical_type ∈ {0..5}:
    0 = Umbrella    伞型
    1 = Wineglass   酒杯型
    2 = T           T 型
    3 = V           V 型
    4 = Gate        门型
    5 = Other       其他

box_type:
    0 = Overall  (tower body + shadow)  整体框（塔身 + 阴影）
    1 = Body     (tower body only)      塔身框（仅塔身）
```

Even ids are **Overall** boxes, odd ids are **Body** boxes,
paired by consecutive indices: (0,1), (2,3), (4,5) ...

偶数 id 为 **Overall** 框，奇数 id 为 **Body** 框，
按相邻索引成对：(0,1), (2,3), (4,5) ...

| ID | Class | Meaning / 含义 |
|---:|:------|:---------------|
|  0 | Umbrella_Overall  | 伞型塔 — 整体框（含阴影） |
|  1 | Umbrella_Body     | 伞型塔 — 仅塔身 |
|  2 | Wineglass_Overall | 酒杯型塔 — 整体框 |
|  3 | Wineglass_Body    | 酒杯型塔 — 仅塔身 |
|  4 | T_Overall         | T 型塔 — 整体框 |
|  5 | T_Body            | T 型塔 — 仅塔身 |
|  6 | V_Overall         | V 型塔 — 整体框 |
|  7 | V_Body            | V 型塔 — 仅塔身 |
|  8 | Gate_Overall      | 门型塔 — 整体框 |
|  9 | Gate_Body         | 门型塔 — 仅塔身 |
| 10 | Other_Overall     | 其他塔型 — 整体框 |
| 11 | Other_Body        | 其他塔型 — 仅塔身 |

---

## Geometric Constraint / 几何约束

For every tower instance:
- `Body` box is geometrically **contained within** the `Overall` box.
- The pair shares the same physical tower type.
- This containment is the basis of the **IoA-based pairing algorithm**
  used by our contrastive auxiliary branch.

对每个塔实例：
- `Body` 框在几何上**完全包含于** `Overall` 框内。
- 两者属于同一物理塔型。
- 这一包含关系是对比学习辅助分支中
  **基于 IoA 的配对算法**的设计依据。

---

## Example Files / 示例文件

The three sample `.txt` files in `labels/` illustrate:

- `example_001_umbrella.txt`        — 1 Umbrella tower (one Overall + Body pair)
                                       1 个伞型塔（一对 Overall + Body）
- `example_002_wineglass_with_T.txt`— 1 Wineglass + 1 T tower
                                       1 个酒杯型 + 1 个 T 型，共 2 对
- `example_003_gate.txt`            — 1 Gate tower (rare class)
                                       1 个门型塔（小样本类）

These coordinate values are **synthetic** (not from real images).
They demonstrate the file structure only.

这些坐标数值是**示意性的**（不来自真实图像），仅用于说明文件结构。
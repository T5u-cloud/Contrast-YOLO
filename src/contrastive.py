# ===========================================================================
# Decoupled Contrastive Learning for Paired BBox Detection
# 双框对比学习模块（用于成对边界框检测）
#
# This module is the core contribution of the tower-detection project.
# It implements a training-only auxiliary branch that learns body-shadow
# differential features via contrastive learning, without adding any
# inference-time overhead.
#
# 本模块是 tower-detection 项目的核心贡献。
# 它实现了一个仅在训练阶段使用的辅助分支，通过对比学习引导骨干网络
# 学习"塔身-阴影"的差异特征，推理阶段完全不参与计算，零额外开销。
#
# Original location in the working tree: ultralytics/utils/contrastive.py
# This file is a verbatim copy from the trained model's source.
# 原始路径：ultralytics/utils/contrastive.py
# 本文件为训练时实际使用的源码原样拷贝，未做任何修改。
#
# Key components / 主要组件:
#   - DiffEncoder           : MLP that learns F_diff from (F_body, F_overall)
#                             用 MLP 从 (F_body, F_overall) 学习差异特征 F_diff
#   - extract_pair_features : Greedy Overall-Body pairing via IoA + class mask
#                             基于 IoA + 物理类别约束的贪心配对算法
#   - SupConLoss            : Supervised contrastive loss (Khosla et al., 2020)
#                             有监督对比损失（Khosla et al., NeurIPS 2020）
#   - PairContrastiveLoss   : Wraps the above with three loss terms
#                             整合上述组件，组合 Body / Diff / Pair 三路 loss
# ===========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class DiffEncoder(nn.Module):
    """
    Difference Encoder：从 (F_body, F_overall) 学习差异特征 F_diff

    用 MLP 而非简单减法，因为 CNN 特征是非线性的，
    F_overall - F_body ≠ "纯阴影特征"
    """

    def __init__(self, in_channels=256, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_channels)
        )

    def forward(self, F_body, F_overall):
        """
        Args:
            F_body: [N, C] body features (已全局池化)
            F_overall: [N, C] overall features (已全局池化)
        Returns:
            F_diff: [N, C] difference features
        """
        x = torch.cat([F_body, F_overall], dim=1)  # [N, 2C]
        F_diff = self.mlp(x)
        return F_diff


def extract_pair_features(p3_feature, batch, roi_size=7, stride=8):
    """
    从 P3 特征图提取每对 (Overall, Body) 的 region features

    ★ 修复：用 cls_id 配对，而不是按面积排序 ★
    """
    bboxes = batch["bboxes"]  # [N, 4] cxcywh 归一化
    batch_idx = batch["batch_idx"]
    cls_labels = batch["cls"].view(-1).long()

    device = p3_feature.device
    bboxes = bboxes.to(device)
    batch_idx = batch_idx.to(device)
    cls_labels = cls_labels.to(device)

    B, C, H, W = p3_feature.shape
    img_size = H * stride

    F_body_list = []
    F_overall_list = []
    physical_cls_list = []

    for b in range(B):
        img_mask = (batch_idx == b)
        boxes = bboxes[img_mask]
        labels = cls_labels[img_mask]

        if len(boxes) < 2:
            continue

        is_overall = (labels % 2 == 0)
        is_body = (labels % 2 == 1)

        overall_boxes = boxes[is_overall]
        overall_labels = labels[is_overall]
        body_boxes = boxes[is_body]
        body_labels = labels[is_body]

        if len(overall_boxes) == 0 or len(body_boxes) == 0:
            continue

        # 1. 将 cxcywh 转换为 xyxy 格式 (ops.box_iou 需要 xyxy)
        def cxcywh_to_xyxy(box_tensor):
            x_c, y_c, w, h = box_tensor.unbind(-1)
            b_x1 = x_c - 0.5 * w
            b_y1 = y_c - 0.5 * h
            b_x2 = x_c + 0.5 * w
            b_y2 = y_c + 0.5 * h
            return torch.stack([b_x1, b_y1, b_x2, b_y2], dim=-1)

        ov_xyxy_norm = cxcywh_to_xyxy(overall_boxes)
        bd_xyxy_norm = cxcywh_to_xyxy(body_boxes)

        # 2. 批量计算交集面积 (Intersection)
        # ov_xyxy_norm: [N, 4], bd_xyxy_norm: [M, 4]
        # bbox_overlaps 会计算 N x M 的 IoU，但我们需要自定义的 IoA
        inter_x1 = torch.max(ov_xyxy_norm[:, None, 0], bd_xyxy_norm[None, :, 0])
        inter_y1 = torch.max(ov_xyxy_norm[:, None, 1], bd_xyxy_norm[None, :, 1])
        inter_x2 = torch.min(ov_xyxy_norm[:, None, 2], bd_xyxy_norm[None, :, 2])
        inter_y2 = torch.min(ov_xyxy_norm[:, None, 3], bd_xyxy_norm[None, :, 3])

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h  # [N, M] 的交集矩阵

        # 3. 计算 Body 的面积
        body_area = body_boxes[:, 2] * body_boxes[:, 3]  # [M]

        # 4. 计算 IoA 矩阵 (Intersection over Area of Body)
        ioa_matrix = inter_area / (body_area[None, :] + 1e-8)  # 形状: [N, M]

        # 5. 生成物理类别匹配掩码 (同物理类别才允许配对)
        ov_phys_cls = overall_labels // 2  # [N]
        bd_phys_cls = body_labels // 2  # [M]
        cls_match_mask = (ov_phys_cls[:, None] == bd_phys_cls[None, :])  # [N, M] 的布尔矩阵

        # 屏蔽不同物理类别的 IoA (将其设为 -1)
        ioa_matrix = torch.where(cls_match_mask, ioa_matrix, torch.tensor(-1.0, device=device))

        # 6. 为每个 Overall 寻找最佳的 Body
        paired_overall = []
        paired_body = []
        paired_physical_cls = []
        used_body_idx = set()

        # 尽管这里还有个小循环，但 N 和 M 一般不大，且最耗时的几何计算已经通过张量矩阵一次性算完了
        # 按行(每个Overall)寻找最大 IoA
        max_ioa_vals, max_ioa_indices = torch.max(ioa_matrix, dim=1)

        for ov_idx in range(len(overall_boxes)):
            best_body_idx = max_ioa_indices[ov_idx].item()
            best_ioa = max_ioa_vals[ov_idx].item()

            # 必须满足有交集 (IoA > 0) 且该 Body 未被占用
            if best_ioa > 0.1 and best_body_idx not in used_body_idx:
                paired_overall.append(ov_idx)
                paired_body.append(best_body_idx)
                paired_physical_cls.append(ov_phys_cls[ov_idx].item())
                used_body_idx.add(best_body_idx)

        if len(paired_overall) == 0:
            continue

        # ============================================
        # 转换坐标 + RoIAlign
        # ============================================
        def to_xyxy_pixel(boxes):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = (cx - w / 2) * img_size
            y1 = (cy - h / 2) * img_size
            x2 = (cx + w / 2) * img_size
            y2 = (cy + h / 2) * img_size
            return torch.stack([x1, y1, x2, y2], dim=1)

        # 取出配对后的 boxes
        ov_paired = overall_boxes[torch.tensor(paired_overall, device=device)]
        bd_paired = body_boxes[torch.tensor(paired_body, device=device)]

        ov_xyxy = to_xyxy_pixel(ov_paired)
        bd_xyxy = to_xyxy_pixel(bd_paired)

        K = len(paired_overall)
        batch_indices = torch.full((K, 1), b, dtype=torch.float32, device=device)

        rois_overall = torch.cat([batch_indices, ov_xyxy], dim=1)
        rois_body = torch.cat([batch_indices, bd_xyxy], dim=1)

        F_overall = roi_align(
            p3_feature, rois_overall,
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / stride,
            aligned=True
        )
        F_body = roi_align(
            p3_feature, rois_body,
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / stride,
            aligned=True
        )

        F_overall_pool = F_overall.mean(dim=[2, 3])
        F_body_pool = F_body.mean(dim=[2, 3])

        physical_cls = torch.tensor(paired_physical_cls, device=device, dtype=torch.long)

        F_body_list.append(F_body_pool)
        F_overall_list.append(F_overall_pool)
        physical_cls_list.append(physical_cls)

    if len(F_body_list) == 0:
        return None, None, None

    F_body_all = torch.cat(F_body_list, dim=0)
    F_overall_all = torch.cat(F_overall_list, dim=0)
    physical_cls_all = torch.cat(physical_cls_list, dim=0)

    return F_body_all, F_overall_all, physical_cls_all


def compute_iou_cxcywh(box1, box2):
    """
    计算两个 cxcywh 格式 bbox 的 IoU
    box1, box2: [4] tensor (cx, cy, w, h)
    """
    # 转 xyxy
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    # 交集
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 并集
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    union_area = b1_area + b2_area - inter_area

    iou = inter_area / (union_area + 1e-8)
    return iou.item()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Reference: Khosla et al., NeurIPS 2020
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [N, C] L2-normalized features
            labels: [N] class labels
        Returns:
            loss: scalar
        """
        device = features.device
        N = features.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=device)

        # 构造正样本 mask
        labels = labels.view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)  # [N, N]
        # 排除自己
        mask_self = torch.eye(N, device=device)
        mask_positive = mask_positive - mask_self

        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [N, N]

        # 数值稳定：减去最大值
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # 排除自己（对角线）
        exp_sim = torch.exp(sim_matrix) * (1 - mask_self)

        # SupCon loss
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 对每个 anchor，取所有正样本的平均
        positive_count = mask_positive.sum(dim=1)
        # 避免除零
        valid = positive_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss_per_anchor = -(mask_positive * log_prob).sum(dim=1) / (positive_count + 1e-8)
        loss = loss_per_anchor[valid].mean()

        return loss


class PairContrastiveLoss(nn.Module):
    """
    完整的对比学习模块
    包含 DiffEncoder + 三个 contrastive loss
    """

    def __init__(self, in_channels=256, temperature=0.07,
                 lambda_body=0.1, lambda_diff=0.1, lambda_pair=0.05):
        super().__init__()
        self.diff_encoder = DiffEncoder(in_channels)
        self.supcon = SupConLoss(temperature)

        self.lambda_body = lambda_body
        self.lambda_diff = lambda_diff
        self.lambda_pair = lambda_pair

    def forward(self, p3_feature, batch):
        """
        Args:
            p3_feature: [B, C, H, W] P3 特征图
            batch: dict
        Returns:
            loss: scalar
            log_dict: dict, 用于 logging
        """
        # 提取 pair features
        F_body, F_overall, cls = extract_pair_features(p3_feature, batch)

        if F_body is None or F_body.shape[0] < 2:
            zero_loss = torch.tensor(0.0, device=p3_feature.device)
            return zero_loss, {"L_body": 0.0, "L_diff": 0.0, "L_pair": 0.0}

        # 计算 F_diff
        F_diff = self.diff_encoder(F_body, F_overall)

        # L2 归一化
        F_body = F.normalize(F_body, dim=1)
        F_diff = F.normalize(F_diff, dim=1)

        # 计算三个 loss
        L_body = self.supcon(F_body, cls)
        L_diff = self.supcon(F_diff, cls)

        # L_pair: 用 (F_body, F_diff) 的拼接
        F_combined = torch.cat([F_body, F_diff], dim=1)  # [K, 2C]
        F_combined = F.normalize(F_combined, dim=1)
        L_pair = self.supcon(F_combined, cls)

        total_loss = (self.lambda_body * L_body +
                      self.lambda_diff * L_diff +
                      self.lambda_pair * L_pair)

        log_dict = {
            "L_body": L_body.item(),
            "L_diff": L_diff.item(),
            "L_pair": L_pair.item()
        }

        return total_loss, log_dict
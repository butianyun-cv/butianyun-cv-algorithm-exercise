# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_loss.py
# 实现功能：目标检测训练损失函数


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

import torch
import torch.nn as nn
import math
from butianyun_utils import P
from butianyun_cv_options import BUTIANYUN_INPUT_IMAGE_SIZE


def butianyun_ciou_loss(pred, target, eps=1e-7):
    """计算CIoU损失
    pred: (N, 4) - [cx, cy, w, h] 归一化坐标
    target: (N, 4) - [cx, cy, w, h] 归一化坐标
    返回: (N,) 每个样本的CIoU损失
    """
    # 转换为左上角和右下角坐标
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    # 交集
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 并集
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + eps

    # IoU
    iou = inter_area / union_area

    # 中心点距离
    pred_cx = pred[:, 0]
    pred_cy = pred[:, 1]
    target_cx = target[:, 0]
    target_cy = target[:, 1]
    center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # 最小包围框对角线
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # 宽高比惩罚 (v)
    v = (4 / math.pi ** 2) * (torch.atan(target[:, 2] / (target[:, 3] + eps))
                                - torch.atan(pred[:, 2] / (pred[:, 3] + eps))) ** 2
    alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - center_dist / enc_diag - alpha * v
    return 1.0 - ciou


class butianyun_object_detection_loss(nn.Module):
    """目标检测损失函数

    检测头输出: (B, 4+class_count, H, W)
    通道组织: [cx, cy, w, h, class_0, class_1, ...]
    """

    def __init__(self, class_count, input_size=None):
        super(butianyun_object_detection_loss, self).__init__()
        self.class_count = class_count
        self.input_size = input_size or BUTIANYUN_INPUT_IMAGE_SIZE
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # 损失权重
        self.box_weight = 7.5
        self.cls_weight = 0.5

    def forward(self, predictions, targets):
        """
        predictions: (B, 4+class_count, H, W) 网络原始输出
        targets: list of list, 每个元素为该图像的标注 [[class_id, cx, cy, w, h], ...]
                 cx, cy, w, h 为归一化坐标 [0, 1]

        返回: (total_loss, box_loss, cls_loss)
        """
        B, C, H, W = predictions.shape
        device = predictions.device
        stride = self.input_size / H

        # 分离预测: box(4) + class(class_count)
        pred_box = predictions[:, :4]        # (B, 4, H, W)
        pred_cls = predictions[:, 4:]         # (B, class_count, H, W)

        # 构建网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # 解码预测框 ()
        # cx = (sigmoid(pred_cx) * 2 - 0.5 + grid_x) * stride / input_size
        # cy = (sigmoid(pred_cy) * 2 - 0.5 + grid_y) * stride / input_size
        # w  = (sigmoid(pred_w) * 2) ^ 2
        # h  = (sigmoid(pred_h) * 2) ^ 2
        pred_cx_decoded = (pred_box[:, 0].sigmoid() * 2 - 0.5 + grid_x) * stride / self.input_size
        pred_cy_decoded = (pred_box[:, 1].sigmoid() * 2 - 0.5 + grid_y) * stride / self.input_size
        pred_w_decoded = (pred_box[:, 2].sigmoid() * 2) ** 2
        pred_h_decoded = (pred_box[:, 3].sigmoid() * 2) ** 2

        # 构建目标张量
        target_box = torch.zeros(B, 4, H, W, device=device)
        target_cls = torch.zeros(B, self.class_count, H, W, device=device)
        target_mask = torch.zeros(B, H, W, device=device)  # 正样本掩码

        # 将GT分配到对应的网格单元
        for b in range(B):
            if targets[b] is None or len(targets[b]) == 0:
                continue
            for box in targets[b]:
                if len(box) < 5:
                    continue
                class_id, cx, cy, w, h = int(box[0]), box[1], box[2], box[3], box[4]
                # 计算GT中心点所在的网格坐标
                gx = int(cx * W)
                gy = int(cy * H)
                gx = max(0, min(gx, W - 1))
                gy = max(0, min(gy, H - 1))

                target_mask[b, gy, gx] = 1.0
                target_box[b, 0, gy, gx] = cx
                target_box[b, 1, gy, gx] = cy
                target_box[b, 2, gy, gx] = w
                target_box[b, 3, gy, gx] = h
                if class_id < self.class_count:
                    target_cls[b, class_id, gy, gx] = 1.0

        num_pos = target_mask.sum().clamp(min=1)

        # === Box损失 (CIoU) ===
        pos_mask_bool = target_mask > 0
        if pos_mask_bool.any():
            pbox = torch.stack([
                pred_cx_decoded[pos_mask_bool],
                pred_cy_decoded[pos_mask_bool],
                pred_w_decoded[pos_mask_bool],
                pred_h_decoded[pos_mask_bool],
            ], dim=-1)
            tbox = target_box.permute(0, 2, 3, 1)[pos_mask_bool]
            box_loss = butianyun_ciou_loss(pbox, tbox).mean()
        else:
            box_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # === 分类损失 (BCE) ===
        # 正样本计算BCE, 负样本的class target全为0
        cls_pred = pred_cls.permute(0, 2, 3, 1).reshape(-1, self.class_count)
        cls_target = target_cls.permute(0, 2, 3, 1).reshape(-1, self.class_count)
        cls_loss_all = self.bce(cls_pred, cls_target)

        # 正样本的分类损失权重更高
        mask_flat = target_mask.reshape(-1, 1).expand(-1, self.class_count)
        cls_loss_pos = (cls_loss_all * mask_flat).sum() / num_pos
        cls_loss_neg = (cls_loss_all * (1 - mask_flat)).sum() / max((B * H * W - num_pos.item()) * self.class_count, 1)
        cls_loss = cls_loss_pos + 0.5 * cls_loss_neg

        # === 总损失 ===
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        return total_loss, box_loss.detach(), cls_loss.detach()

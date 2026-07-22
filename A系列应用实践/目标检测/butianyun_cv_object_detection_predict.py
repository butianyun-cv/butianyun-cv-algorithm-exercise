# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_predict.py
# 实现功能：目标检测预测


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

import os
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image, ImageDraw
from butianyun_utils import P

from butianyun_cv_options import BUTIANYUN_TRAIN_MODEL_PATH, BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_TRAIN_DATASET_PATH
from butianyun_cv_object_detection_net import butianyun_object_detection_net
from butianyun_cv_object_detection_dataset_yolo import butianyun_object_detection_dataset_yolo


BUTIANYUN_DETECTION_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]


def butianyun_decode_predictions(predictions, input_size, conf_threshold=0.25):
    """解码网络输出为目标检测结果
    predictions: (B, 4+class_count, H, W) 网络原始输出
    返回: list of list, 每张图像的检测结果 [[class_id, cx, cy, w, h, confidence], ...]
    """
    B, C, H, W = predictions.shape
    class_count = C - 4
    stride = input_size / H

    pred_box = predictions[:, :4]
    pred_cls = predictions[:, 4:]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )

    # YOLOv8解码
    pred_cx = (pred_box[:, 0].sigmoid() * 2 - 0.5 + grid_x) * stride / input_size
    pred_cy = (pred_box[:, 1].sigmoid() * 2 - 0.5 + grid_y) * stride / input_size
    pred_w = (pred_box[:, 2].sigmoid() * 2) ** 2
    pred_h = (pred_box[:, 3].sigmoid() * 2) ** 2

    cls_scores = pred_cls.sigmoid()

    results = []
    for b in range(B):
        detections = []
        for i in range(H):
            for j in range(W):
                cls_score = cls_scores[b, :, i, j]
                max_score, max_cls = cls_score.max(dim=0)
                if max_score.item() >= conf_threshold:
                    cx = pred_cx[b, i, j].item()
                    cy = pred_cy[b, i, j].item()
                    w = pred_w[b, i, j].item()
                    h = pred_h[b, i, j].item()
                    detections.append([max_cls.item(), cx, cy, w, h, max_score.item()])
        results.append(detections)

    return results


def butianyun_nms(detections, iou_threshold=0.45):
    """非极大值抑制
    detections: [[class_id, cx, cy, w, h, confidence], ...]
    返回: 过滤后的检测结果
    """
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    keep = []

    while len(detections) > 0:
        best = detections[0]
        keep.append(best)
        remaining = []

        for det in detections[1:]:
            iou = butianyun_compute_iou(best[1:5], det[1:5])
            if iou < iou_threshold:
                remaining.append(det)

        detections = remaining

    return keep


def butianyun_compute_iou(box1, box2, eps=1e-7):
    """计算两个归一化框的IoU
    box: [cx, cy, w, h]
    """
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area + eps

    return inter_area / union_area


def butianyun_cv_object_detection_predict(img_path_list, model_path, dataset_path=None):
    P("预测开始")
    from butianyun_cv_options import butianyun_set_random_seed
    butianyun_set_random_seed()

    if dataset_path is None:
        dataset_path = BUTIANYUN_TRAIN_DATASET_PATH

    ds = butianyun_object_detection_dataset_yolo(dataset_path)
    class_count = ds.butianyun_get_class_count()

    # 从checkpoint推断class_count，避免因数据集路径问题导致形状不匹配
    if class_count == 0:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
        head_bias_key = 'detection_head.3.bias'
        if head_bias_key in ckpt:
            class_count = ckpt[head_bias_key].shape[0] - 4
            P(f"从checkpoint推断class_count={class_count}")

    model = butianyun_object_detection_net(class_count)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    P(f"模型已加载: {model_path}")

    result_dir = os.path.join('data', 'result')
    os.makedirs(result_dir, exist_ok=True)

    result = []

    with torch.inference_mode():
        for img_path in img_path_list:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                P(f"图像路径: {img_path} 打开失败")
                continue
            img_h, img_w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_INPUT_IMAGE_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
            img_tensor = np.ascontiguousarray(img_resized, dtype=np.float32)
            img_tensor /= 255.0
            img_tensor = torch.from_numpy(np.transpose(img_tensor, (2, 0, 1))).unsqueeze(0)

            outputs = model(img_tensor)
            detections = butianyun_decode_predictions(outputs, BUTIANYUN_INPUT_IMAGE_SIZE)[0]
            detections = butianyun_nms(detections)

            # 在原图上绘制检测框
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            img_result = []
            P(f"图像路径: {img_path}")
            for det in detections:
                class_id, cx, cy, w, h, conf = det
                class_name = ds.butianyun_get_class_name(int(class_id) + 1)
                # 转换为像素坐标
                px_cx = int(cx * img_w)
                px_cy = int(cy * img_h)
                px_w = int(w * img_w)
                px_h = int(h * img_h)
                px_x1 = max(px_cx - px_w // 2, 0)
                px_y1 = max(px_cy - px_h // 2, 0)
                px_x2 = min(px_cx + px_w // 2, img_w)
                px_y2 = min(px_cy + px_h // 2, img_h)
                # 确保坐标合法，避免中心超出图像边界时x2<x1或y2<y1
                px_x2 = max(px_x2, px_x1)
                px_y2 = max(px_y2, px_y1)
                P(f"  目标: {class_name} ({int(class_id)+1}) 置信度: {conf:.4f} "
                      f"边界框: ({px_x1}, {px_y1}, {px_x2}, {px_y2})")

                color = BUTIANYUN_DETECTION_COLORS[int(class_id) % len(BUTIANYUN_DETECTION_COLORS)]
                if px_x1 < px_x2 and px_y1 < px_y2:
                    draw.rectangle([px_x1, px_y1, px_x2, px_y2], outline=color, width=2)
                    draw.text((px_x1, max(px_y1 - 15, 0)), f"{class_name} {conf:.2f}", fill=color)

                img_result.append({
                    'img_path': img_path,
                    'class_id': int(class_id) + 1,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': [px_x1, px_y1, px_x2, px_y2],
                })

            # 保存画框后的图片到data/result目录
            img_name = os.path.basename(img_path)
            save_path = os.path.join(result_dir, img_name)
            imageio.imwrite(save_path, np.array(pil_img))
            P(f"  结果已保存: {save_path}")

            result.extend(img_result)

    return result

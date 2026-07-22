# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_test.py
# 实现功能：目标检测测试


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################


import os
import numpy as np
import torch
from butianyun_utils import P

from butianyun_cv_options import (
    BUTIANYUN_TRAIN_DATASET_PATH, BUTIANYUN_TRAIN_MODEL_PATH,
    BUTIANYUN_TRAIN_BATCH_SIZE, BUTIANYUN_INPUT_IMAGE_SIZE
)
from butianyun_cv_object_detection_dataset_yolo import butianyun_object_detection_dataset_yolo
from butianyun_cv_object_detection_dataloader import butianyun_create_dataloader
from butianyun_cv_object_detection_net import butianyun_object_detection_net
from butianyun_cv_object_detection_loss import butianyun_object_detection_loss
from butianyun_cv_object_detection_predict import butianyun_decode_predictions, butianyun_nms, butianyun_compute_iou


def butianyun_compute_ap(recalls, precisions):
    """计算Average Precision (VOC)"""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def butianyun_compute_map(all_predictions, all_ground_truths, class_count, iou_threshold=0.5):
    """计算mAP@iou_threshold

    all_predictions: list, 每个元素为一张图像的预测 [[class_id, cx, cy, w, h, confidence], ...]
    all_ground_truths: list, 每个元素为一张图像的标注 [[class_id, cx, cy, w, h], ...]
    class_count: 类别数量
    iou_threshold: IoU阈值

    返回: (mAP, per_class_AP_dict)
    """
    ap_dict = {}

    for cls_id in range(class_count):
        # 收集该类别所有预测，按置信度降序
        all_det = []
        for img_idx, preds in enumerate(all_predictions):
            for pred in preds:
                if int(pred[0]) == cls_id:
                    all_det.append((img_idx, pred[1:5], pred[5]))
        all_det.sort(key=lambda x: x[2], reverse=True)

        # 收集该类别所有GT，标记是否已匹配
        gt_matched = {}
        n_gt = 0
        for img_idx, gts in enumerate(all_ground_truths):
            matched_list = []
            for gt in gts:
                if int(gt[0]) == cls_id:
                    matched_list.append(False)
                    n_gt += 1
                else:
                    matched_list.append(None)  # 非该类别
            gt_matched[img_idx] = matched_list

        if n_gt == 0:
            if len(all_det) == 0:
                ap_dict[cls_id] = 1.0
            else:
                ap_dict[cls_id] = 0.0
            continue

        tp = np.zeros(len(all_det))
        fp = np.zeros(len(all_det))

        for det_idx, (img_idx, pred_box, conf) in enumerate(all_det):
            gts = all_ground_truths[img_idx]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if int(gt[0]) != cls_id:
                    continue
                iou = butianyun_compute_iou(pred_box, gt[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if not gt_matched[img_idx][best_gt_idx]:
                    tp[det_idx] = 1
                    gt_matched[img_idx][best_gt_idx] = True
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap_dict[cls_id] = butianyun_compute_ap(recalls, precisions)

    mAP = np.mean(list(ap_dict.values())) if ap_dict else 0.0
    return mAP, ap_dict


def butianyun_cv_object_detection_test(dataset_path, model_path):
    P("测试开始")
    from butianyun_cv_options import butianyun_set_random_seed
    butianyun_set_random_seed()

    test_dir = os.path.join(dataset_path, 'test')
    test_dataset = butianyun_object_detection_dataset_yolo(test_dir)
    test_loader = butianyun_create_dataloader(test_dataset, batch_size=BUTIANYUN_TRAIN_BATCH_SIZE, shuffle=False)
    test_size = len(test_dataset)
    P(f"测试集: {test_size}")

    ds = test_dataset
    class_count = ds.butianyun_get_class_count()
    model = butianyun_object_detection_net(class_count)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    P(f"模型已加载: {model_path}")

    criterion = butianyun_object_detection_loss(class_count)
    total_losses = []
    box_losses = []
    cls_losses = []
    total_batches = len(test_loader)

    all_predictions = []
    all_ground_truths = []

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            total_loss, box_loss, cls_loss = criterion(outputs, labels)
            total_losses.append(total_loss)
            box_losses.append(box_loss)
            cls_losses.append(cls_loss)

            # 解码预测结果用于计算mAP
            batch_dets = butianyun_decode_predictions(outputs, BUTIANYUN_INPUT_IMAGE_SIZE)
            for i, dets in enumerate(batch_dets):
                nms_dets = butianyun_nms(dets)
                all_predictions.append(nms_dets)
                all_ground_truths.append(labels[i])

            print(f"\r  Batch {batch_idx+1}/{total_batches} "
                  f"loss: {total_loss.item():.6f} box: {box_loss.item():.6f} cls: {cls_loss.item():.6f}", end="", flush=True)

        print()

    avg_box_loss = torch.stack(box_losses).mean().item()
    avg_cls_loss = torch.stack(cls_losses).mean().item()
    total_loss = avg_box_loss * criterion.box_weight + avg_cls_loss * criterion.cls_weight

    P(f"测试结果 - loss: {total_loss:.6f} "
          f"box: {avg_box_loss:.6f} cls: {avg_cls_loss:.6f}")

    # 计算mAP@0.5
    mAP, ap_dict = butianyun_compute_map(all_predictions, all_ground_truths, class_count, iou_threshold=0.5)
    P(f"mAP@0.5: {mAP:.4f}")
    for cls_id, ap in ap_dict.items():
        cls_name = ds.butianyun_get_class_name(cls_id + 1)
        P(f"  AP@0.5 - {cls_name}: {ap:.4f}")

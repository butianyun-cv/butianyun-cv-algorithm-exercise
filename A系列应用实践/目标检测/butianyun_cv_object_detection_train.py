# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_train.py
# 实现功能：目标检测训练


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################


import os
import numpy as np
import torch
import torch.optim as optim
from butianyun_utils import P

from butianyun_cv_options import (
    BUTIANYUN_TRAIN_DATASET_PATH, BUTIANYUN_TRAIN_MODEL_PATH,
    BUTIANYUN_TRAIN_EPOCHS, BUTIANYUN_TRAIN_LR, BUTIANYUN_TRAIN_BATCH_SIZE, BUTIANYUN_INPUT_IMAGE_SIZE
)
from butianyun_cv_object_detection_dataset_yolo import butianyun_object_detection_dataset_yolo
from butianyun_cv_object_detection_dataloader import butianyun_create_dataloader
from butianyun_cv_object_detection_net import butianyun_object_detection_net
from butianyun_cv_object_detection_loss import butianyun_object_detection_loss
from butianyun_cv_object_detection_predict import butianyun_decode_predictions, butianyun_nms, butianyun_compute_iou
from butianyun_cv_object_detection_test import butianyun_compute_map


def butianyun_compute_batch_map(outputs, labels, class_count, input_size, iou_threshold=0.5):
    """计算一个batch的mAP"""
    batch_dets = butianyun_decode_predictions(outputs, input_size)
    all_predictions = []
    all_ground_truths = []
    for i, dets in enumerate(batch_dets):
        nms_dets = butianyun_nms(dets)
        all_predictions.append(nms_dets)
        all_ground_truths.append(labels[i])
    mAP, _ = butianyun_compute_map(all_predictions, all_ground_truths, class_count, iou_threshold)
    return mAP




def butianyun_cv_object_detection_train(dataset_path, model_path):
    from butianyun_cv_options import butianyun_set_random_seed
    butianyun_set_random_seed()

    train_dir = os.path.join(dataset_path, 'train')
    train_dataset = butianyun_object_detection_dataset_yolo(train_dir)
    train_loader = butianyun_create_dataloader(train_dataset, batch_size=BUTIANYUN_TRAIN_BATCH_SIZE, shuffle=True)
    train_size = len(train_dataset)
    
    val_dir = os.path.join(dataset_path, 'valid')
    val_dataset = butianyun_object_detection_dataset_yolo(val_dir)
    val_loader = butianyun_create_dataloader(val_dataset, batch_size=BUTIANYUN_TRAIN_BATCH_SIZE, shuffle=False)
    val_size = len(val_dataset)
    
    P(f"训练集: {train_size}, 验证集: {val_size}")


    ds = train_dataset
    class_count = ds.butianyun_get_class_count()
    model = butianyun_object_detection_net(class_count)
    criterion = butianyun_object_detection_loss(class_count)
    optimizer = optim.Adam(model.parameters(), lr=BUTIANYUN_TRAIN_LR)

    for epoch in range(BUTIANYUN_TRAIN_EPOCHS):
        P(f"Epoch {epoch+1}/{BUTIANYUN_TRAIN_EPOCHS} START ...")

        model.train()
        batch_losses = []
        box_losses = []
        cls_losses = []
        batch_maps = []
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            total_loss, box_loss, cls_loss = criterion(outputs, labels)
            total_loss.backward()
            optimizer.step()

            batch_losses.append(total_loss.detach())
            box_losses.append(box_loss)
            cls_losses.append(cls_loss)

            cur_loss = total_loss.item()
            cur_box = box_loss.item()
            cur_cls = cls_loss.item()

            with torch.inference_mode():
                outputs_eval = model(images)
                batch_mAP = butianyun_compute_batch_map(outputs_eval, labels, class_count, BUTIANYUN_INPUT_IMAGE_SIZE)
            batch_maps.append(batch_mAP)

            print(f"\r  Batch {batch_idx+1}/{total_batches} "
                  f"loss: {cur_loss:.6f} "
                  f"box: {cur_box:.6f} cls: {cur_cls:.6f} "
                  f"mAP@0.5: {batch_mAP:.4f}", end="", flush=True)

        train_loss = torch.stack(batch_losses).mean().item()
        avg_box_loss = torch.stack(box_losses).mean().item()
        avg_cls_loss = torch.stack(cls_losses).mean().item()
        train_mAP = np.mean(batch_maps)
        print()

        model.eval()
        with torch.inference_mode():
            val_box_losses = []
            val_cls_losses = []
            val_all_predictions = []
            val_all_ground_truths = []
            val_total_batches = len(val_loader)
            for val_batch_idx, (images, labels) in enumerate(val_loader):
                outputs = model(images)
                _, val_box_loss, val_cls_loss = criterion(outputs, labels)
                val_box_losses.append(val_box_loss)
                val_cls_losses.append(val_cls_loss)

                # 解码预测用于计算mAP
                batch_dets = butianyun_decode_predictions(outputs, BUTIANYUN_INPUT_IMAGE_SIZE)
                for i, dets in enumerate(batch_dets):
                    nms_dets = butianyun_nms(dets)
                    val_all_predictions.append(nms_dets)
                    val_all_ground_truths.append(labels[i])

                val_batch_mAP = butianyun_compute_map(
                    val_all_predictions[-len(labels):],
                    val_all_ground_truths[-len(labels):],
                    class_count, iou_threshold=0.5
                )[0]
                print(f"\r  Val Batch {val_batch_idx+1}/{val_total_batches} "
                      f"val mAP@0.5: {val_batch_mAP:.4f}", end="", flush=True)

            val_box_loss = torch.stack(val_box_losses).mean().item()
            val_cls_loss = torch.stack(val_cls_losses).mean().item()
            val_total_loss = val_box_loss * criterion.box_weight + val_cls_loss * criterion.cls_weight
            val_mAP, _ = butianyun_compute_map(val_all_predictions, val_all_ground_truths, class_count, iou_threshold=0.5)
            print()

        P(f"Epoch {epoch+1}/{BUTIANYUN_TRAIN_EPOCHS} FINISHED "
              f"train loss: {train_loss:.6f} "
              f"train box: {avg_box_loss:.6f} train cls: {avg_cls_loss:.6f} train mAP@0.5: {train_mAP:.4f} | "
              f"val loss: {val_total_loss:.6f} "
              f"val box: {val_box_loss:.6f} val cls: {val_cls_loss:.6f} val mAP@0.5: {val_mAP:.4f}")

    torch.save(model.state_dict(), model_path)
    P(f"模型已保存到: {model_path}")

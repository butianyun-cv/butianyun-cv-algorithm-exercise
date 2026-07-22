# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_dataset.py
# 实现功能：目标检测数据集


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from butianyun_utils import *
from butianyun_cv_options import BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_DATA_LOADER_WORKER_COUNT
from butianyun_cv_object_detection_dataset_utils import *


def butianyun_detection_collate_fn(batch):
    """自定义collate函数，支持目标检测中每张图像标注数量不同的情况"""
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    images = torch.stack(images, 0)
    return images, labels


def butianyun_create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=None, pin_memory=True):
    if num_workers is None:
        num_workers = min(butianyun_get_cpu_count(), BUTIANYUN_DATA_LOADER_WORKER_COUNT)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=butianyun_detection_collate_fn,
    )
    return dataloader

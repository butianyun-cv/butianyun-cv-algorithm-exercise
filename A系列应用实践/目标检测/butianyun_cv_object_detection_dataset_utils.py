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
from butianyun_utils import P
from butianyun_cv_options import BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_DATA_LOADER_WORKER_COUNT



BUTIANYUN_INPUT_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg'}


def butianyun_batch_load_images(img_path_list):
    batch_imgs = []
    valid_paths = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            P(f"图像路径: {img_path} 打开失败")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_INPUT_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))
        batch_imgs.append(img)
        valid_paths.append(img_path)
    if len(batch_imgs) == 0:
        return None, []
    batch_tensor = torch.from_numpy(np.array(batch_imgs))
    return batch_tensor, valid_paths

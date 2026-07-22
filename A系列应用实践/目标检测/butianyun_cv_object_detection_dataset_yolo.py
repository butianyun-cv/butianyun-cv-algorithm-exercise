# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_dataset_yolo.py
# 实现功能：目标检测数据集YOLO格式



############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

from ntpath import isdir
import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from butianyun_utils import P
from butianyun_cv_options import BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_DATA_LOADER_WORKER_COUNT
from butianyun_cv_object_detection_dataset_utils import *
from butianyun_utils import *



class butianyun_object_detection_dataset_yolo(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []
        self.butianyun_load_classes()
        self.butianyun_load_data()


    def butianyun_load_data(self):    
        if len(self.classes) == 0:
            return
        if not os.path.isdir(os.path.join(self.root_dir, 'images')) or not os.path.isdir(os.path.join(self.root_dir, 'labels')):
            return

        images_dir = os.path.join(self.root_dir, 'images')
        labels_dir = os.path.join(self.root_dir, 'labels')
        for img_name in os.listdir(images_dir):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in BUTIANYUN_INPUT_IMAGE_FORMATS:
                img_path = os.path.join(images_dir, img_name)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    boxes = []
                    with open(label_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                boxes.append([class_id, cx, cy, w, h])
                    self.labels.append(boxes)

        P(f"数据集路径: {self.root_dir}, 图片数量: {len(self.images)} 类型数量：{len(self.classes)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_INPUT_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


    def butianyun_load_classes(self):
        data_yaml_path = os.path.join(self.root_dir, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            if os.path.isdir(os.path.join(self.root_dir, 'images')) and os.path.isdir(os.path.join(self.root_dir, 'labels')):
                parent_yaml_path = os.path.join(os.path.dirname(self.root_dir), 'data.yaml')
                if os.path.exists(parent_yaml_path):
                    data_yaml_path = parent_yaml_path
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            if data_yaml and 'names' in data_yaml:
                names = data_yaml['names']
                if isinstance(names, dict):
                    self.classes = [names[k] for k in sorted(names.keys())]
                elif isinstance(names, list):
                    self.classes = list(names)

    def butianyun_get_classes(self):
        return self.classes

    def butianyun_get_class_id(self, class_name):
        return self.butianyun_get_classes().index(class_name) + 1

    def butianyun_get_class_name(self, class_id):
        return self.butianyun_get_classes()[class_id - 1]

    def butianyun_get_class_count(self):
        return len(self.butianyun_get_classes())

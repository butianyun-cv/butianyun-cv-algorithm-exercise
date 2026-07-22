# -*- coding: utf-8 -*-
# butianyun_cv_options.py
# 实现功能：各种选项


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################



import os
import random
import numpy as np
import torch
from butianyun_utils import P


butianyun_run_mode_str = os.environ.get('BUTIANYUN_RUN_MODE', 'TRAIN+TEST+PREDICT')
BUTIANYUN_RUN_MODE = butianyun_run_mode_str.upper().split('+', 3)
BUTIANYUN_TRAIN_DATASET_PATH = os.environ.get('BUTIANYUN_TRAIN_DATASET_PATH', 'butianyun_object_detection_fruits_dataset')
BUTIANYUN_TRAIN_MODEL_PATH = os.environ.get('BUTIANYUN_TRAIN_MODEL_PATH', 'butianyun_object_detection_fruits.pth')
BUTIANYUN_TRAIN_EPOCHS = int(os.environ.get('BUTIANYUN_TRAIN_EPOCHS', '100'))
BUTIANYUN_TRAIN_LR = float(os.environ.get('BUTIANYUN_TRAIN_LR', '0.01'))
BUTIANYUN_TRAIN_BATCH_SIZE = int(os.environ.get('BUTIANYUN_TRAIN_BATCH_SIZE', '100'))
BUTIANYUN_RANDOM_SEED = int(os.environ.get('BUTIANYUN_RANDOM_SEED', '42'))
butianyun_img_path_str=os.environ.get('BUTIANYUN_PREDICT_IMAGE_PATH', 'predict.png')
BUTIANYUN_PREDICT_IMAGE_PATH= butianyun_img_path_str.split(',', 10)
BUTIANYUN_INPUT_IMAGE_SIZE = int(os.environ.get('BUTIANYUN_INPUT_IMAGE_SIZE', 640))
BUTIANYUN_DATA_LOADER_WORKER_COUNT = int(os.environ.get('BUTIANYUN_DATA_LOADER_WORKER_COUNT', 2))
BUTIANYUN_BACKBONE_NETWORK_TYPE_LIST=['MobileNetV3Small', 'MobileNetV3Large', 'VGG16', 'VGG19_BN', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'EfficientNet_B0', 'DenseNet121']
BUTIANYUN_BACKBONE_NETWORK_TYPE = os.environ.get('BUTIANYUN_BACKBONE_NETWORK_TYPE', 'MobileNetV3Small')
BUTIANYUN_BACKBONE_NETWORK_USE_PRETRAIN = os.environ.get('BUTIANYUN_BACKBONE_NETWORK_USE_PRETRAIN', 'true').lower() == 'true'


def butianyun_set_random_seed(seed=None):
    if seed is None:
        seed = BUTIANYUN_RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def butianyun_show_options():
    P(f"BUTIANYUN_RUN_MODE={BUTIANYUN_RUN_MODE}")
    P(f"BUTIANYUN_TRAIN_DATASET_PATH={BUTIANYUN_TRAIN_DATASET_PATH}")
    P(f"BUTIANYUN_TRAIN_MODEL_PATH={BUTIANYUN_TRAIN_MODEL_PATH}")
    P(f"BUTIANYUN_TRAIN_EPOCHS={BUTIANYUN_TRAIN_EPOCHS}")
    P(f"BUTIANYUN_TRAIN_LR={BUTIANYUN_TRAIN_LR}")
    P(f"BUTIANYUN_TRAIN_BATCH_SIZE={BUTIANYUN_TRAIN_BATCH_SIZE}")
    P(f"BUTIANYUN_RANDOM_SEED={BUTIANYUN_RANDOM_SEED}")
    P(f"BUTIANYUN_PREDICT_IMAGE_PATH={BUTIANYUN_PREDICT_IMAGE_PATH}")
    P(f"BUTIANYUN_INPUT_IMAGE_SIZE={BUTIANYUN_INPUT_IMAGE_SIZE}")
    P(f"BUTIANYUN_DATA_LOADER_WORKER_COUNT={BUTIANYUN_DATA_LOADER_WORKER_COUNT}")
    P(f"BUTIANYUN_BACKBONE_NETWORK_TYPE_LIST={BUTIANYUN_BACKBONE_NETWORK_TYPE_LIST}")
    P(f"BUTIANYUN_BACKBONE_NETWORK_TYPE={BUTIANYUN_BACKBONE_NETWORK_TYPE}")
    P(f"BUTIANYUN_BACKBONE_NETWORK_USE_PRETRAIN={BUTIANYUN_BACKBONE_NETWORK_USE_PRETRAIN}")

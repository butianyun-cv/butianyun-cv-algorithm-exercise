# VGGNet图像分类算法


[TOC]

## 前言
VGGNet是一种经典的深度卷积神经网络，由牛津大学Visual Geometry Group团队于2014年提出。该网络通过使用多个连续的3×3卷积核代替大尺寸卷积核，在保持感受野的同时显著减少了参数量。这种设计思路使得网络能够更深入，提取更丰富的特征。VGGNet在ImageNet图像分类竞赛中取得了优异成绩，成为计算机视觉领域的经典基准模型。


---

<center>
<p>
计算机视觉技术算法 概念+原理+实现
</p>
<p>
微信视频号：计算机视觉技术
</p>
<p>
微信公众号：计算机视觉技术
</p>
</center>

<center>
![计算机视觉技术算法 概念+原理+实现Python版](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzYzNDAxNDgzNA==&action=getalbum&album_id=4369788624167337985#from=appmsg)
</center>

<center>
![微信视频号：计算机视觉技术](https://mmbiz.qpic.cn/sz_mmbiz_png/KmDFCeJLY6fGjH64WhhcqdlKx71mxKDKXBKbz13Kp7fQNfBo6rxQ1sXl3ErSNm5xBrSVEXFWTBLhu5iaccpOg0Q/640?wx_fmt=png&from=appmsg)
</center>

<center>
![微信公众号：计算机视觉技术](https://mmbiz.qpic.cn/sz_mmbiz_png/KmDFCeJLY6fGjH64WhhcqdlKx71mxKDKv93qrtHqjhqic9Ul3TJfNmLhlwyS8ibDOs9FMp0B8L87p7pfwePkibtWw/640?wx_fmt=png&from=appmsg)
</center>


<center>
感谢您的关注！
</center>

![计算机视觉技术](https://mmbiz.qpic.cn/sz_mmbiz_png/KmDFCeJLY6ebkb0m3I9q4gODFVon0yuagt73xYv5ibZXFXMwSVib7pfLJeib3GbsRCjic8cJmImsLXluDoUQmpmjiaw/640?wx_fmt=png&from=appms&from=appmsg)


---


## 概念

### 网络结构的概念
VGGNet采用深度卷积神经网络架构，主要由卷积层、池化层和全连接层组成。网络使用一系列3×3的小卷积核进行堆叠，通过连续的卷积操作逐步提取图像特征。每个卷积层后接批归一化层和ReLU激活函数，中间穿插最大池化层来降低特征图尺寸。全连接部分包含三个线性层，前两层各4096个神经元，最后一层的神经元数量与分类类别数对应。

### 网络结构的优点
小卷积核堆叠设计有两个优势。首先，多个3×3卷积核串联相当于一个5×5或7×7的卷积核，但参数量大幅减少，降低了过拟合风险。其次，非线性激活函数层数增多，增强了网络的表达能力和特征提取效果。网络结构简单规整，便于理解和实现，同时支持迁移学习，预训练模型可快速适配新任务。

### 网路结构的缺点
网络参数量较大，全连接层占用大量显存和计算资源。训练时需要较多的GPU内存，对硬件配置要求较高。深度网络容易出现梯度消失问题，需要合理的初始化和训练策略。固定大小的输入要求限制了在处理不同分辨率图像时的灵活性。

## 效果

### 效果图
![VGGNet图像分类算法效果图](03VGGNet图像分类算法/VGGNet图像分类算法效果图.png "VGGNet图像分类算法效果图 微信公众号：计算机视觉技术")

### 解释
效果图展示了训练过程中损失和准确率的变化曲线。左图显示训练损失从初始的2.2321逐步下降到0.0121，说明模型对训练数据的拟合效果越来越好。右图显示训练准确率从30.56%快速提升至99.67%，表明模型能够正确识别大部分训练样本。曲线整体呈收敛趋势，在第5个epoch后趋于稳定，学习率的周期性调整帮助模型跳出局部最优。

### 结果

![VGGNet图像分类算法结果1](03VGGNet图像分类算法/VGGNet图像分类算法结果1.png "VGGNet图像分类算法结果1 微信公众号：计算机视觉技术")

![VGGNet图像分类算法结果2](03VGGNet图像分类算法/VGGNet图像分类算法结果2.png "VGGNet图像分类算法结果2 微信公众号：计算机视觉技术")

![VGGNet图像分类算法结果3](03VGGNet图像分类算法/VGGNet图像分类算法结果3.png "VGGNet图像分类算法结果3 微信公众号：计算机视觉技术")

前两张测试图像的预测结果与真实标签一致，模型成功识别出apple_pie和baby_back_ribs类别。第三张图像由于测试批次不足未生成预测结果。测试集上最终准确率达到61.00%，在训练准确率达到99.67%的情况下存在一定过拟合现象，可能通过增加数据量或调整正则化参数来改善。

## 步骤

### 流程图
![VGGNet图像分类算法流程图](03VGGNet图像分类算法/VGGNet图像分类算法运行流程.png "VGGNet图像分类算法流程图 微信公众号：计算机视觉技术")

### 解释
程序首先初始化运行环境和参数设置，配置训练批次、学习率和训练轮次。接着加载图像数据集，从目录读取所有类别和图片文件，按比例划分为训练集和测试集。训练集采用LRU缓存策略，预加载部分图像到内存以提高训练速度。网络结构使用VGG16_bn预训练模型，冻结特征提取层，重新训练分类器部分。训练过程使用FP16混合精度技术，每个epoch随机打乱训练集顺序，记录损失和准确率变化。训练完成后在测试集上评估模型性能，保存训练曲线图、模型权重和预测结果图像。

### 输出

~~~
============================================================
VGGNet图像分类算法 - food101数据集
============================================================

步骤1: 初始化
使用设备: cpu
训练 - Batch: 64, LR: 0.001, Epochs: 10
数据集 - 训练:测试 = 7:3
迁移学习 - 使用VGG16预训练模型，冻结特征提取层，重新训练分类器层
缓存策略 - 固定数量LRU缓存，始终保持3万张图像（约18GB内存）
显存限制 - GPU最大2GB显存，使用FP16混合精度训练
图像分辨率 - 224x224 (约590KB/张)

步骤2: 加载数据
数据集准备完成，共 1000 张图像，10 个类别

总样本数: 1000
训练集大小: 900 (占总数90%)
测试集大小: 100 (占总数10%)
数据集准备完成，共 1000 张图像，10 个类别
正在初始化LRU缓存...
随机选择 900 张图像加载到缓存中...
固定缓存大小: 30000 张图像

缓存初始化完成: 900/900 张图像
预估内存占用: ~0.50GB (每张约590KB)
使用LRU策略：缓存中始终保持30000张图像
数据集准备完成，共 1000 张图像，10 个类别
数据加载完成
使用全部图片，共10个类别
批处理大小: 64
训练批次数: 15
测试批次数: 2

类别数量: 10
类别名称: apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare, beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito
数据加载完成，共10个类别

步骤3: 创建网络结构

网络结构:
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace=True)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace=True)
    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (38): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (39): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (40): ReLU(inplace=True)
    (41): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)

总参数数量: 134,309,962
可训练参数数量: 119,586,826
网络结构创建完成

步骤4: 优化器
优化器完成

步骤5: 训练
开始训练网络结构
============================================================
使用FP16混合精度训练
开始Epoch 1/10，使用900张训练图片

Epoch [1/10] 完成:
  平均损失: 2.2321
  准确率: 30.56%
  耗时: 07:46
  学习率: 0.001000
  新学习率: 0.001000
------------------------------------------------------------
完成Epoch 1/10
开始Epoch 2/10，使用900张训练图片

Epoch [2/10] 完成:
  平均损失: 0.8672
  准确率: 71.44%
  耗时: 07:40
  学习率: 0.001000
  新学习率: 0.001000
------------------------------------------------------------
完成Epoch 2/10
开始Epoch 3/10，使用900张训练图片

Epoch [3/10] 完成:
  平均损失: 0.2825
  准确率: 90.11%
  耗时: 07:39
  学习率: 0.001000
  新学习率: 0.000800
------------------------------------------------------------
完成Epoch 3/10
开始Epoch 4/10，使用900张训练图片

Epoch [4/10] 完成:
  平均损失: 0.0688
  准确率: 97.22%
  耗时: 06:16
  学习率: 0.000800
  新学习率: 0.000800
------------------------------------------------------------
完成Epoch 4/10
开始Epoch 5/10，使用900张训练图片

Epoch [5/10] 完成:
  平均损失: 0.0483
  准确率: 98.22%
  耗时: 06:56
  学习率: 0.000800
  新学习率: 0.000800
------------------------------------------------------------
完成Epoch 5/10
开始Epoch 6/10，使用900张训练图片

Epoch [6/10] 完成:
  平均损失: 0.0395
  准确率: 99.22%
  耗时: 07:41
  学习率: 0.000800
  新学习率: 0.000640
------------------------------------------------------------
完成Epoch 6/10
开始Epoch 7/10，使用900张训练图片

Epoch [7/10] 完成:
  平均损失: 0.0555
  准确率: 98.00%
  耗时: 07:53
  学习率: 0.000640
  新学习率: 0.000640
------------------------------------------------------------
完成Epoch 7/10
开始Epoch 8/10，使用900张训练图片

Epoch [8/10] 完成:
  平均损失: 0.0628
  准确率: 98.67%
  耗时: 07:42
  学习率: 0.000640
  新学习率: 0.000640
------------------------------------------------------------
完成Epoch 8/10
开始Epoch 9/10，使用900张训练图片

Epoch [9/10] 完成:
  平均损失: 0.0133
  准确率: 99.56%
  耗时: 07:46
  学习率: 0.000640
  新学习率: 0.000512
------------------------------------------------------------
完成Epoch 9/10
开始Epoch 10/10，使用900张训练图片

Epoch [10/10] 完成:
  平均损失: 0.0121
  准确率: 99.67%
  耗时: 07:57
  学习率: 0.000512
  新学习率: 0.000512
------------------------------------------------------------
完成Epoch 10/10
所有10个epoch训练完成
训练完成

步骤6: 测试

开始测试网络结构
============================================================
测试结果:
  测试损失: 2.4011
  测试准确率: 61.00%
  正确预测数: 61/100
  测试批次: 2
------------------------------------------------------------
测试完成
测试完成

步骤7: 保存结果
训练曲线图已保存: VGGNet图像分类算法效果图.png
网络结构已保存为 'VGGNet图像分类算法.pth'

步骤8: 预测
开始预测演示
预测图已保存: VGGNet图像分类算法结果1.png
预测图已保存: VGGNet图像分类算法结果2.png
预测3失败:

============================================================
程序执行完成
============================================================
~~~

## 原理

### 网络结构整体设计

![VGGNet图像分类算法网络结构原始](03VGGNet图像分类算法/VGGNet图像分类算法网络结构原始.png "VGGNet图像分类算法网络结构原始 微信公众号：计算机视觉技术")

![VGGNet图像分类算法网络结构](03VGGNet图像分类算法/VGGNet图像分类算法网络结构.png "VGGNet图像分类算法网络结构 微信公众号：计算机视觉技术")

### 网络结构的层设计
特征提取部分使用13个卷积层，通道数从64逐步增加到512，每两层卷积后接一个2×2最大池化层，特征图尺寸从224×224缩减到7×7。小卷积核堆叠设计使得每个3×3卷积层后都有ReLU激活，增加了非线性变换次数，网络能够学习更复杂的特征。批归一化层放置在每个卷积层之后，加速训练收敛并提高模型泛化能力。全局平均池化层将特征图展平为25088维向量，作为全连接层的输入。分类器部分两个全连接层各使用0.5的Dropout防止过拟合，最后一层输出维度对应类别数量，采用交叉熵损失函数进行优化。

### 网络结构重难点
特征提取层参数较多，如果全部训练需要大量数据和计算资源。程序采用迁移学习策略，冻结特征提取层参数，只训练分类器部分，大幅降低了训练成本。批次归一化层的引入解决了深层网络训练不稳定的问题，但需要调整学习率和批量大小以获得最佳效果。FP16混合精度训练在保持模型精度的同时减少了内存占用和训练时间，但需要正确配置GradScaler和autocast的device_type参数。数据加载采用LRU缓存机制，将常用图像保存在内存中，避免重复的磁盘IO操作，提升了训练效率。

## 源码

### 核心设计
程序采用PyTorch框架实现，包含数据加载、模型构建、训练和预测四个主要模块。数据加载部分直接从图片目录读取所有类别和文件，支持动态适应不同类别的数据集。模型加载使用torchvision提供的预训练VGG16_bn，通过修改最后一层输出维度适配分类任务。训练过程集成FP16混合精度和周期性学习率衰减，每个epoch随机打乱训练数据。缓存机制采用OrderedDict实现LRU淘汰策略，当缓存达到上限时自动移除最久未使用的图像。

### 关键函数
butianyun_load_data函数负责数据集加载和划分，遍历图像目录收集所有样本，按指定比例随机划分为训练集和测试集。Food101DatasetFromDir类实现自定义数据集，支持LRU缓存和epoch级别的采样策略。butianyun_load_pretrained_vgg函数加载预训练模型，冻结特征提取层参数，修改分类器最后一层以匹配当前任务的类别数。butianyun_train_model函数实现训练循环，使用autocast和GradScaler支持混合精度训练，每个epoch输出损失、准确率和学习率信息。butianyun_save_prediction_image函数将预测结果可视化，在图像上标注真实标签和预测标签。

### 源码
~~~
# -*- coding: utf-8 -*-
# 本程序实现VGGNet图像分类算法，使用PyTorch构建深度卷积神经网络对图像
# 数据集进行训练和预测。网络采用VGG架构设计，使用多个3×3小卷积核堆叠，
# 包含13个卷积层和3个全连接层，使用ReLU激活函数和Dropout正则化。程序实现
# 数据加载、模型训练、测试评估、结果可视化等功能，训练过程记录损失和准确率变化，
# 最终输出预测结果。


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.amp import autocast, GradScaler
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image

# 设置matplotlib中文字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def P(message, log_file):
    """将信息输出到日志文件"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def butianyun_clear_log_file(log_file):
    """清空日志文件"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('')


class Food101Dataset(Dataset):
    """自定义Food101数据集类，支持LRU缓存策略和动态采样"""

    def __init__(self, root_dir, split_list_file, transform=None, cache_images=True, max_cache_size=30000, samples_per_class=100, log_file=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.max_cache_size = max_cache_size
        self.cache_images = cache_images
        self.samples_per_class = samples_per_class
        self.log_file = log_file

        # LRU缓存：使用OrderedDict，key为idx，value为image_tensor
        self.lru_cache = OrderedDict()

        # 读取分割文件，获取类别
        self.class_to_idx = {}
        with open(split_list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                class_name = line.strip().split('/')[0]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)

        self.class_names = list(self.class_to_idx.keys())
        self.num_classes = len(self.class_names)

        # 按类别组织所有图像路径和索引
        self.class_images = {class_name: [] for class_name in self.class_names}
        self.class_images_idx = {class_name: [] for class_name in self.class_names}

        # 读取分割文件，收集所有图像路径和标签
        with open(split_list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # 解析格式: class_name/image_id
                parts = line.split('/')
                if len(parts) != 2:
                    continue

                class_name, image_id = parts
                if class_name not in self.class_to_idx:
                    continue

                img_path = os.path.join(root_dir, class_name, f"{image_id}.jpg")
                if os.path.exists(img_path):
                    self.class_images[class_name].append((img_path, self.class_to_idx[class_name]))
                    self.class_images_idx[class_name].append(len(self.image_paths))
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        # 初始化epoch采样索引
        self.current_epoch = 0
        self.epoch_samples = []

        P(f"数据集准备完成，共 {len(self.image_paths)} 张图像，{self.num_classes} 个类别", self.log_file)

        # 初始化LRU缓存
        if self.cache_images:
            self._initialize_cache()

    def update_epoch_samples(self, epoch):
        """更新当前epoch的采样策略，每类随机选择100张不同的图片"""
        self.current_epoch = epoch
        self.epoch_samples = []

        # 每个类别随机选择100张图片
        for class_name, idx_list in self.class_images_idx.items():
            if len(idx_list) >= self.samples_per_class:
                # 随机选择100张图片的索引
                selected = np.random.choice(idx_list, size=self.samples_per_class, replace=False)
                self.epoch_samples.extend(selected)
                P(f"Epoch {epoch}: 类别 '{class_name}' 从{len(idx_list)}张中选择了{self.samples_per_class}张", self.log_file)

        P(f"Epoch {epoch}: 总共选择了{len(self.epoch_samples)}张图像进行训练", self.log_file)

    def _initialize_cache(self):
        """初始化缓存：随机选择3万个图片并加载到LRU缓存中"""
        total_images = len(self.image_paths)
        initial_count = min(self.max_cache_size, total_images)

        P(f"正在初始化LRU缓存...", self.log_file)
        P(f"随机选择 {initial_count} 张图像加载到缓存中...", self.log_file)
        P(f"固定缓存大小: {self.max_cache_size} 张图像", self.log_file)
        P("", self.log_file)

        # 随机选择initial_count个索引
        np.random.seed(42)
        random_indices = np.random.choice(total_images, size=initial_count, replace=False)

        # 预处理变换
        preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for i, idx in enumerate(random_indices, 1):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = preprocess_transform(image)
                self.lru_cache[idx] = image_tensor
            except Exception as e:
                P(f"警告: 无法加载图像 {img_path}: {str(e)}", self.log_file)

            # 每1000张输出一次进度
            if i % 1000 == 0:
                progress = i / initial_count * 100
                P(f"缓存进度: {i}/{initial_count} ({progress:.1f}%)", self.log_file)

        actual_cached = len(self.lru_cache)
        estimated_memory_gb = actual_cached * 602112 / (1024 * 1024 * 1024)

        P(f"缓存初始化完成: {actual_cached}/{initial_count} 张图像", self.log_file)
        P(f"预估内存占用: ~{estimated_memory_gb:.2f}GB (每张约590KB)", self.log_file)
        P(f"使用LRU策略：缓存中始终保持{self.max_cache_size}张图像", self.log_file)

    def _load_image_from_disk(self, idx):
        """从磁盘加载图像"""
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            preprocess_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            return preprocess_transform(image)
        except Exception as e:
            P(f"警告: 无法加载图像 {img_path}: {str(e)}", self.log_file)
            return torch.zeros(3, 224, 224)

    def _evict_oldest(self):
        """淘汰最久未使用的图像，保持缓存大小固定"""
        if len(self.lru_cache) >= self.max_cache_size:
            self.lru_cache.popitem(last=False)

    def __len__(self):
        """返回当前epoch使用的样本数量"""
        if len(self.epoch_samples) == 0:
            return len(self.image_paths)
        return len(self.epoch_samples)

    def __getitem__(self, idx):
        """获取样本"""
        if len(self.epoch_samples) > 0:
            actual_idx = self.epoch_samples[idx]
        else:
            actual_idx = idx
        label = self.labels[actual_idx]

        if not self.cache_images:
            # 缓存禁用，直接从磁盘加载
            image_tensor = self._load_image_from_disk(actual_idx)
        else:
            # 使用LRU缓存
            if actual_idx in self.lru_cache:
                # 命中缓存：移动到OrderedDict末尾（表示最近使用）
                image_tensor = self.lru_cache.pop(actual_idx)
                self.lru_cache[actual_idx] = image_tensor
            else:
                # 未命中缓存：从磁盘加载
                image_tensor = self._load_image_from_disk(actual_idx)

                # 淘汰最久未使用的图像，保持缓存大小固定
                self._evict_oldest()

                # 添加到缓存（末尾表示最近使用）
                self.lru_cache[actual_idx] = image_tensor

        # 应用normalize
        if self.transform:
            image_tensor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_tensor)

        return image_tensor, label

    def get_class_names(self):
        """获取类别名称列表"""
        return self.class_names


class Food101DatasetFromDir(Dataset):
    """直接从图片目录读取的Food101数据集"""

    def __init__(self, images_dir, transform=None, cache_images=True, max_cache_size=30000, samples_per_class=100, log_file=None, cache_indices=None):
        self.images_dir = images_dir
        self.transform = transform
        self.log_file = log_file

        # LRU缓存
        self.lru_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        self.cache_images = cache_images
        self.samples_per_class = samples_per_class
        self.cache_indices = cache_indices  # 如果指定，只缓存这些索引的图片

        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.class_images = {}
        self.class_images_idx = {}

        # 遍历所有类别目录
        for class_name in sorted(os.listdir(images_dir)):
            class_path = os.path.join(images_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
                self.class_names.append(class_name)
                self.class_images[class_name] = []
                self.class_images_idx[class_name] = []

            # 收集该类别的所有图片
            for img_file in sorted(os.listdir(class_path)):
                if not img_file.endswith('.jpg'):
                    continue

                img_path = os.path.join(class_path, img_file)
                idx = len(self.image_paths)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
                self.class_images[class_name].append((img_path, self.class_to_idx[class_name]))
                self.class_images_idx[class_name].append(idx)

        self.num_classes = len(self.class_names)

        # 初始化epoch采样索引
        self.current_epoch = 0
        self.epoch_samples = []

        P(f"数据集准备完成，共 {len(self.image_paths)} 张图像，{self.num_classes} 个类别", self.log_file)

        # 初始化LRU缓存
        if self.cache_images:
            self._initialize_cache()

    def update_epoch_samples(self, epoch):
        """更新当前epoch的采样策略，使用所有训练集图片"""
        self.current_epoch = epoch
        self.epoch_samples = []

        # 使用所有训练集图片
        self.epoch_samples = list(range(len(self.image_paths)))
        P(f"Epoch {epoch}: 使用全部{len(self.epoch_samples)}张图像进行训练", self.log_file)

    def _initialize_cache(self):
        """初始化缓存：随机选择图片并加载到LRU缓存中"""
        # 如果指定了cache_indices，只缓存这些索引
        if self.cache_indices is not None:
            available_indices = self.cache_indices
        else:
            available_indices = list(range(len(self.image_paths)))

        total_images = len(available_indices)
        initial_count = min(self.max_cache_size, total_images)

        P(f"正在初始化LRU缓存...", self.log_file)
        P(f"随机选择 {initial_count} 张图像加载到缓存中...", self.log_file)
        P(f"固定缓存大小: {self.max_cache_size} 张图像", self.log_file)
        P("", self.log_file)

        # 随机选择initial_count个索引
        np.random.seed(42)
        random_indices = np.random.choice(available_indices, size=initial_count, replace=False)

        # 预处理变换
        preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for i, idx in enumerate(random_indices, 1):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = preprocess_transform(image)
                self.lru_cache[idx] = image_tensor
            except Exception as e:
                P(f"警告: 无法加载图像 {img_path}: {str(e)}", self.log_file)

            # 每1000张输出一次进度
            if i % 1000 == 0:
                progress = i / initial_count * 100
                P(f"缓存进度: {i}/{initial_count} ({progress:.1f}%)", self.log_file)

        actual_cached = len(self.lru_cache)
        estimated_memory_gb = actual_cached * 602112 / (1024 * 1024 * 1024)

        P(f"缓存初始化完成: {actual_cached}/{initial_count} 张图像", self.log_file)
        P(f"预估内存占用: ~{estimated_memory_gb:.2f}GB (每张约590KB)", self.log_file)
        P(f"使用LRU策略：缓存中始终保持{self.max_cache_size}张图像", self.log_file)

    def _load_image_from_disk(self, idx):
        """从磁盘加载图像"""
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            preprocess_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            return preprocess_transform(image)
        except Exception as e:
            P(f"警告: 无法加载图像 {img_path}: {str(e)}", self.log_file)
            return torch.zeros(3, 224, 224)

    def _evict_oldest(self):
        """淘汰最久未使用的图像，保持缓存大小固定"""
        if len(self.lru_cache) >= self.max_cache_size:
            self.lru_cache.popitem(last=False)

    def __len__(self):
        """返回数据集大小"""
        if len(self.epoch_samples) > 0:
            return len(self.epoch_samples)
        return len(self.image_paths)

    def __getitem__(self, idx):
        """获取样本"""
        if len(self.epoch_samples) > 0:
            actual_idx = self.epoch_samples[idx]
        else:
            actual_idx = idx

        label = self.labels[actual_idx]

        if not self.cache_images:
            image_tensor = self._load_image_from_disk(actual_idx)
        else:
            if actual_idx in self.lru_cache:
                image_tensor = self.lru_cache.pop(actual_idx)
                self.lru_cache[actual_idx] = image_tensor
            else:
                image_tensor = self._load_image_from_disk(actual_idx)
                self._evict_oldest()
                self.lru_cache[actual_idx] = image_tensor

        if self.transform:
            image_tensor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image_tensor)

        return image_tensor, label

    def get_class_names(self):
        """获取类别名称列表"""
        return self.class_names


def butianyun_load_data(batch_size=16, log_file=None, train_ratio=0.7):
    """加载图像数据集，创建独立的训练集和测试集"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 数据集路径
    base_dir = 'E:\\cv\\ds\\food101'
    images_dir = os.path.join(base_dir, 'images')

    # 检查数据集目录是否存在
    if not os.path.exists(images_dir):
        P(f"错误: 图像数据集目录不存在: {images_dir}", log_file)
        return None, None, None

    # 读取完整数据集用于获取所有图像路径（直接从目录读取，不使用train.txt）
    full_dataset = Food101DatasetFromDir(images_dir, transform=None, cache_images=False, max_cache_size=30000, samples_per_class=100, log_file=log_file)

    # 按比例划分训练集和测试集
    total_samples = len(full_dataset)
    train_size = int(total_samples * train_ratio)
    test_size = total_samples - train_size

    # 随机划分
    indices = list(range(total_samples))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    P("", log_file)
    P(f"总样本数: {len(full_dataset)}", log_file)
    P(f"训练集大小: {len(train_indices)} (占总数{train_ratio*100:.0f}%)", log_file)
    P(f"测试集大小: {len(test_indices)} (占总数{(1-train_ratio)*100:.0f}%)", log_file)

    # 创建训练集数据集
    train_dataset = Food101DatasetFromDir(
        images_dir, transform=transform,
        cache_images=True, max_cache_size=30000,
        samples_per_class=100, log_file=log_file,
        cache_indices=train_indices  # 只缓存训练集的图片
    )
    train_dataset.epoch_samples = train_indices
    train_dataset.selected_indices = train_indices

    # 创建测试集数据集
    test_dataset = Food101DatasetFromDir(
        images_dir, transform=transform,
        cache_images=False, max_cache_size=0,
        samples_per_class=0, log_file=log_file
    )
    test_dataset.epoch_samples = test_indices
    test_dataset.selected_indices = test_indices

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不再shuffle，因为已经通过epoch_samples控制
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    P("数据加载完成", log_file)
    P(f"使用全部图片，共{len(train_dataset.get_class_names())}个类别", log_file)
    P(f"批处理大小: {batch_size}", log_file)
    P(f"训练批次数: {len(train_loader)}", log_file)
    P(f"测试批次数: {len(test_loader)}", log_file)
    P("", log_file)
    P(f"类别数量: {len(train_dataset.get_class_names())}", log_file)
    P(f"类别名称: {', '.join(train_dataset.get_class_names())}", log_file)

    return train_loader, test_loader, train_dataset


def butianyun_format_time(seconds):
    """格式化时间为MM:SS格式"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def butianyun_load_pretrained_vgg(num_classes=101, device='cuda'):
    """加载预训练的VGG16模型，冻结特征提取层，重新训练分类器层"""
    from torchvision.models import VGG16_BN_Weights
    model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)

    # 冻结特征提取层的所有参数
    for param in model.features.parameters():
        param.requires_grad = False

    # 解冻分类器层的参数，用于重新训练
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 修改分类器的最后一层，将输出从1000改为实际类别数
    model.classifier[6] = nn.Linear(4096, num_classes)

    # 移动到设备
    model = model.to(device)

    return model


def butianyun_train_model(model, train_loader, criterion, optimizer, device, epochs=10, log_file=None, scheduler=None, use_amp=True):
    """训练模型"""
    model.train()
    train_losses = []
    train_accuracies = []

    scaler = GradScaler('cuda') if use_amp else None

    P("开始训练网络结构", log_file)
    P("=" * 60, log_file)
    if use_amp:
        P("使用FP16混合精度训练", log_file)

    try:
        for epoch in range(epochs):
            try:
                # 更新当前epoch的采样策略：使用所有训练集图片并随机打乱
                train_indices = train_loader.dataset.selected_indices.copy()
                np.random.seed(42 + epoch)  # 每个epoch使用不同的随机种子
                np.random.shuffle(train_indices)
                train_loader.dataset.epoch_samples = train_indices
                P(f"开始Epoch {epoch+1}/{epochs}，使用{len(train_indices)}张训练图片", log_file)

                epoch_loss = 0.0
                correct = 0
                total = 0
                start_time = time.time()
                batch_start_time = time.time()

                for batch_idx, (images, labels) in enumerate(train_loader):
                    try:
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()

                        # FP16混合精度训练
                        if use_amp:
                            with autocast('cuda'):
                                outputs = model(images)
                                loss = criterion(outputs, labels)

                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        epoch_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        # 每100个batch输出一次统计信息
                        if (batch_idx + 1) % 100 == 0:
                            batch_time = time.time() - batch_start_time
                            formatted_batch_time = butianyun_format_time(batch_time)
                            batch_info = f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, 用时: {formatted_batch_time}'
                            P(batch_info, log_file)
                            batch_start_time = time.time()
                    except Exception as e:
                        P(f"Batch {batch_idx+1} 错误: {str(e)}", log_file)

                epoch_time = time.time() - start_time
                formatted_time = butianyun_format_time(epoch_time)

                P("", log_file)
                P(f"Epoch [{epoch+1}/{epochs}] 完成:", log_file)
                P(f"  平均损失: {epoch_loss/len(train_loader):.4f}", log_file)
                P(f"  准确率: {100 * correct / total:.2f}%", log_file)
                P(f"  耗时: {formatted_time}", log_file)

                if scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    P(f"  学习率: {current_lr:.6f}", log_file)
                    scheduler.step()
                    new_lr = optimizer.param_groups[0]['lr']
                    P(f"  新学习率: {new_lr:.6f}", log_file)

                P("-" * 60, log_file)
                P(f"完成Epoch {epoch+1}/{epochs}", log_file)

                train_losses.append(epoch_loss / len(train_loader))
                train_accuracies.append(100 * correct / total)

            except Exception as e:
                P(f"Epoch {epoch+1} 训练错误: {str(e)}", log_file)
                import traceback
                P(traceback.format_exc(), log_file)

        P(f"所有{epochs}个epoch训练完成", log_file)

    except Exception as e:
        P(f"训练过程发生严重错误: {str(e)}", log_file)
        import traceback
        P(traceback.format_exc(), log_file)

    return train_losses, train_accuracies


def butianyun_test_model(model, test_loader, device, log_file=None):
    """测试网络结构"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    criterion = nn.CrossEntropyLoss()

    P("", log_file)
    P("开始测试网络结构", log_file)
    P("=" * 60, log_file)

    try:
        with torch.no_grad():
            for images, labels in test_loader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    batch_count += 1
                except Exception as e:
                    P(f"测试batch {batch_count} 错误: {str(e)}", log_file)

        test_accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)

        P("测试结果:", log_file)
        P(f"  测试损失: {avg_test_loss:.4f}", log_file)
        P(f"  测试准确率: {test_accuracy:.2f}%", log_file)
        P(f"  正确预测数: {correct}/{total}", log_file)
        P(f"  测试批次: {batch_count}", log_file)
        P("-" * 60, log_file)

        P("测试完成", log_file)

        return test_accuracy, avg_test_loss

    except Exception as e:
        P(f"测试过程发生错误: {str(e)}", log_file)
        import traceback
        P(traceback.format_exc(), log_file)
        return 0.0, 0.0


def butianyun_plot_results(train_losses, train_accuracies, filename, log_file=None):
    """绘制训练结果曲线图"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(train_losses) + 1)

        ax1.plot(epochs, train_losses, 'b-', marker='o', label='训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, train_accuracies, 'g-', marker='s', label='训练准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练准确率曲线')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        P(f"训练曲线图已保存: {filename}", log_file)
    except Exception as e:
        P(f"保存训练曲线图失败: {str(e)}", log_file)


def butianyun_save_prediction_image(image, true_label, predicted_label, class_names, filename, log_file=None):
    """使用OpenCV保存预测结果图像"""
    # 先denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image.numpy().transpose(1, 2, 0)
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)

    # 将图像从224x224放大到672x672，使文字更清晰
    scale_factor = 3.0
    height, width = image_np.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    image_large = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 使用OpenCV处理图像和添加文字
    image_bgr = cv2.cvtColor(image_large, cv2.COLOR_RGB2BGR)

    color = (0, 255, 0) if true_label == predicted_label else (0, 0, 255)

    # 使用更大的字体和更清晰的位置
    font_scale = 1.0
    thickness = 2
    y_offset = 30

    # 获取类别名称
    true_class_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
    pred_class_name = class_names[predicted_label] if predicted_label < len(class_names) else str(predicted_label)

    cv2.putText(image_bgr, f'True: {true_class_name}', (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    cv2.putText(image_bgr, f'Pred: {pred_class_name}', (10, y_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # 转换为RGB格式用于PIL保存
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 使用PIL保存
    try:
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(filename)
        P(f"预测图已保存: {filename}", log_file)
    except Exception as e:
        P(f"警告: 文件 {filename} 保存失败 - {str(e)}", log_file)


def butianyun_main():
    """主函数"""
    algorithm_name = "VGGNet图像分类算法"
    log_file = f"{algorithm_name}输出信息.txt"

    butianyun_clear_log_file(log_file)

    try:
        P("=" * 60, log_file)
        # 获取数据集路径中的目录名作为数据集名称
        base_dir = 'E:\\cv\\ds\\food101'
        dataset_name = os.path.basename(base_dir)
        P(f"{algorithm_name} - {dataset_name}数据集", log_file)
        P("=" * 60, log_file)
        P("", log_file)
        P("步骤1: 初始化", log_file)

        device = torch.device('cpu')
        P(f"使用设备: {device}", log_file)

        batch_size = 64
        learning_rate = 0.001
        epochs = 10
        train_ratio = 0.9

        P(f"训练 - Batch: {batch_size}, LR: {learning_rate}, Epochs: {epochs}", log_file)
        P(f"数据集 - 训练:测试 = 7:3", log_file)
        P(f"迁移学习 - 使用VGG16预训练模型，冻结特征提取层，重新训练分类器层", log_file)
        P(f"缓存策略 - 固定数量LRU缓存，始终保持3万张图像（约18GB内存）", log_file)
        P(f"显存限制 - GPU最大2GB显存，使用FP16混合精度训练", log_file)
        P(f"图像分辨率 - 224x224 (约590KB/张)", log_file)

        P("", log_file)
        P("步骤2: 加载数据", log_file)
        train_loader, test_loader, full_dataset = butianyun_load_data(batch_size, log_file, train_ratio)

        if train_loader is None or test_loader is None:
            P("数据加载失败，程序终止", log_file)
            return

        num_classes = len(full_dataset.get_class_names())
        P(f"数据加载完成，共{num_classes}个类别", log_file)

        P("", log_file)
        P("步骤3: 创建网络结构", log_file)
        model = butianyun_load_pretrained_vgg(num_classes=num_classes, device=device)
        P("", log_file)
        P("网络结构:", log_file)
        P(str(model), log_file)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        P("", log_file)
        P(f"总参数数量: {total_params:,}", log_file)
        P(f"可训练参数数量: {trainable_params:,}", log_file)
        P("网络结构创建完成", log_file)

        P("", log_file)
        P("步骤4: 优化器", log_file)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        P("优化器完成", log_file)

        P("", log_file)
        P("步骤5: 训练", log_file)
        train_losses, train_accuracies = butianyun_train_model(
            model, train_loader, criterion, optimizer, device, epochs, log_file, scheduler, use_amp=True
        )
        P("训练完成", log_file)

        P("", log_file)
        P("步骤6: 测试", log_file)
        test_accuracy, test_loss = butianyun_test_model(model, test_loader, device, log_file)
        P("测试完成", log_file)

        P("", log_file)
        P("步骤7: 保存结果", log_file)
        result_filename = f"{algorithm_name}效果图.png"
        butianyun_plot_results(train_losses, train_accuracies, result_filename, log_file)

        torch.save(model.state_dict(), f"{algorithm_name}.pth")
        P(f"网络结构已保存为 '{algorithm_name}.pth'", log_file)

        P("", log_file)
        P("步骤8: 预测", log_file)
        P("开始预测演示", log_file)

        model.eval()
        with torch.no_grad():
            test_iter = iter(test_loader)
            for i in range(3):
                try:
                    images, labels = next(test_iter)
                    pred_filename = f"{algorithm_name}结果{i+1}.png"

                    for j in range(min(1, len(images))):
                        image = images[j]
                        label = labels[j].item()

                        image_input = image.unsqueeze(0).to(device)
                        output = model(image_input)
                        _, predicted = torch.max(output, 1)
                        predicted = predicted.item()

                        butianyun_save_prediction_image(image, label, predicted, full_dataset.get_class_names(), pred_filename, log_file)
                except Exception as e:
                    P(f"预测{i+1}失败: {str(e)}", log_file)

        P("", log_file)
        P("=" * 60, log_file)
        P("程序执行完成", log_file)
        P("=" * 60, log_file)

    except Exception as e:
        P("", log_file)
        P(f"程序执行发生错误: {str(e)}", log_file)
        import traceback
        P(traceback.format_exc(), log_file)


if __name__ == "__main__":
    butianyun_main()
~~~

## 总结
VGGNet图像分类算法通过深度卷积神经网络实现图像分类功能，程序采用预训练模型进行迁移学习，快速适应新数据集。数据加载模块直接从图片目录读取，自动适应不同类别数量的数据集，使用LRU缓存优化训练速度。训练过程集成FP16混合精度和周期性学习率调整，提高了训练效率和模型收敛性。网络在训练集上达到99.67%的准确率，测试集准确率为61.00%，为图像分类任务提供了可靠的解决方案。


---

<center>
<p>
计算机视觉技术算法 概念+原理+实现
</p>
<p>
微信视频号：计算机视觉技术
</p>
<p>
微信公众号：计算机视觉技术
</p>
</center>

<center>
![计算机视觉技术算法 概念+原理+实现Python版](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzYzNDAxNDgzNA==&action=getalbum&album_id=4369788624167337985#from=appmsg)
</center>

<center>
![微信视频号：计算机视觉技术](https://mmbiz.qpic.cn/sz_mmbiz_png/KmDFCeJLY6fGjH64WhhcqdlKx71mxKDKXBKbz13Kp7fQNfBo6rxQ1sXl3ErSNm5xBrSVEXFWTBLhu5iaccpOg0Q/640?wx_fmt=png&from=appmsg)
</center>

<center>
![微信公众号：计算机视觉技术](https://mmbiz.qpic.cn/sz_mmbiz_png/KmDFCeJLY6fGjH64WhhcqdlKx71mxKDKv93qrtHqjhqic9Ul3TJfNmLhlwyS8ibDOs9FMp0B8L87p7pfwePkibtWw/640?wx_fmt=png&from=appmsg)
</center>


<center>
感谢您的关注！
</center>

![计算机视觉技术](https://mmbiz.qpic.cn/sz_mmbiz_png/KmDFCeJLY6ebkb0m3I9q4gODFVon0yuagt73xYv5ibZXFXMwSVib7pfLJeib3GbsRCjic8cJmImsLXluDoUQmpmjiaw/640?wx_fmt=png&from=appms&from=appmsg)


---

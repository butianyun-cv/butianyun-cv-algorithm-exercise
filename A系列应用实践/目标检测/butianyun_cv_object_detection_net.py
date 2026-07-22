# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_net.py
# 实现功能：目标检测网络


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    vgg16, VGG16_Weights,
    vgg19_bn, VGG19_BN_Weights,
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights,
)
from butianyun_utils import P
from butianyun_cv_options import BUTIANYUN_BACKBONE_NETWORK_TYPE, BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_BACKBONE_NETWORK_USE_PRETRAIN

BUTIANYUN_SUPPORTED_BACKBONES = [
    'MobileNetV3Small', 'MobileNetV3Large',
    'VGG16', 'VGG19_BN',
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
    'EfficientNet_B0', 'DenseNet121',
]


class butianyun_resnet_backbone(nn.Module):
    """ResNet特征提取器，用于目标检测（去掉avgpool和fc）"""
    def __init__(self, resnet_model, out_channels):
        super(butianyun_resnet_backbone, self).__init__()
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def butianyun_create_object_detection_backbone(network_type, use_pretrain=None):
    if use_pretrain is None:
        use_pretrain = BUTIANYUN_BACKBONE_NETWORK_USE_PRETRAIN

    if network_type == 'MobileNetV3Small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrain else None
        model = mobilenet_v3_small(weights=weights)
        backbone = model.features
        out_channels = 576
    elif network_type == 'MobileNetV3Large':
        weights = MobileNet_V3_Large_Weights.DEFAULT if use_pretrain else None
        model = mobilenet_v3_large(weights=weights)
        backbone = model.features
        out_channels = 960
    elif network_type == 'VGG16':
        weights = VGG16_Weights.DEFAULT if use_pretrain else None
        model = vgg16(weights=weights)
        backbone = model.features
        out_channels = 512
    elif network_type == 'VGG19_BN':
        weights = VGG19_BN_Weights.DEFAULT if use_pretrain else None
        model = vgg19_bn(weights=weights)
        backbone = model.features
        out_channels = 512
    elif network_type == 'ResNet18':
        weights = ResNet18_Weights.DEFAULT if use_pretrain else None
        model = resnet18(weights=weights)
        backbone = butianyun_resnet_backbone(model, 512)
        out_channels = 512
    elif network_type == 'ResNet34':
        weights = ResNet34_Weights.DEFAULT if use_pretrain else None
        model = resnet34(weights=weights)
        backbone = butianyun_resnet_backbone(model, 512)
        out_channels = 512
    elif network_type == 'ResNet50':
        weights = ResNet50_Weights.DEFAULT if use_pretrain else None
        model = resnet50(weights=weights)
        backbone = butianyun_resnet_backbone(model, 2048)
        out_channels = 2048
    elif network_type == 'ResNet101':
        weights = ResNet101_Weights.DEFAULT if use_pretrain else None
        model = resnet101(weights=weights)
        backbone = butianyun_resnet_backbone(model, 2048)
        out_channels = 2048
    elif network_type == 'EfficientNet_B0':
        weights = EfficientNet_B0_Weights.DEFAULT if use_pretrain else None
        model = efficientnet_b0(weights=weights)
        backbone = model.features
        out_channels = 1280
    elif network_type == 'DenseNet121':
        weights = DenseNet121_Weights.DEFAULT if use_pretrain else None
        model = densenet121(weights=weights)
        backbone = model.features
        out_channels = 1024
    else:
        raise ValueError(f"不支持的骨干网络类型: {network_type}")

    return backbone, out_channels


class butianyun_object_detection_net(nn.Module):
    def __init__(self, class_count, backbone_type=None):
        super(butianyun_object_detection_net, self).__init__()
        if backbone_type is None:
            backbone_type = BUTIANYUN_BACKBONE_NETWORK_TYPE
        self.backbone, out_channels = butianyun_create_object_detection_backbone(backbone_type)
        self.detection_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 4 + class_count, 1),
        )
        P(f"骨干网络: {backbone_type}, 输出通道: {out_channels}")

    def forward(self, x):
        features = self.backbone(x)
        output = self.detection_head(features)
        return output


def butianyun_count_parameters(net):
    return sum(p.numel() for p in net.parameters())


def butianyun_count_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def butianyun_count_flops(net, input_size=(1, 3, BUTIANYUN_INPUT_IMAGE_SIZE, BUTIANYUN_INPUT_IMAGE_SIZE)):
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        batch_size, in_channels, h, w = input[0].size()
        out_channels = output.size(1)
        kernel_h, kernel_w = module.kernel_size
        groups = module.groups
        flops_per_instance = in_channels // groups * kernel_h * kernel_w * out_channels
        num_instances = h * w
        flops += flops_per_instance * num_instances
        if module.bias is not None:
            flops += out_channels * num_instances

    def linear_hook(module, input, output):
        nonlocal flops
        in_f = input[0].size(-1)
        out_f = output.size(-1)
        flops += in_f * out_f
        if module.bias is not None:
            flops += out_f

    def bn_hook(module, input, output):
        nonlocal flops
        num_elements = input[0].numel()
        flops += num_elements * 2

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            hooks.append(module.register_forward_hook(bn_hook))

    device = next(net.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        net(dummy_input)

    for hook in hooks:
        hook.remove()

    return flops


def butianyun_network_structure_dump(class_count):
    net = butianyun_object_detection_net(class_count)
    P(net)
    P(f"网络参数总数量: {butianyun_count_parameters(net)}")
    P(f"网络可训练参数数量: {butianyun_count_trainable_parameters(net)}")
    P(f"网络FLOPs数量(MFLOPs): {butianyun_count_flops(net) / 1024 / 1024:.2f}")

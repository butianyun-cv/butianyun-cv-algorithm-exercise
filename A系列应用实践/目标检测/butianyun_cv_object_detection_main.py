# -*- coding: utf-8 -*-
# butianyun_cv_object_detection_main.py
# 主程序：猫狗识别物体分类


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

from operator import indexOf
import sys
import argparse
import trace
import multiprocessing
from butianyun_cv_options import *
from butianyun_cv_object_detection_net import butianyun_network_structure_dump
from butianyun_cv_object_detection_train import butianyun_cv_object_detection_train
from butianyun_cv_object_detection_test import butianyun_cv_object_detection_test
from butianyun_cv_object_detection_predict import butianyun_cv_object_detection_predict
from butianyun_cv_object_detection_dataset_yolo import butianyun_object_detection_dataset_yolo
from butianyun_utils import P





def butianyun_usage():
    P("Options:")
    P("  --train                训练模型")
    P("  --test                 测试模型")
    P("  --predict              预测图像")
    P("  --model_path           模型路径")
    P("  --dataset_path         数据集路径")
    P("  --predict_img_path     预测图像路径")


def butianyun_main():
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    P(f"############################################################")
    P(f"#   微信公众号：计算机视觉技术")
    P(f"#   微信视频号：计算机视觉技术")
    P(f"#   网站         ：BUTIANYUN.COM")
    P(f"#   计算机视觉技术课程实践：目标检测")
    P(f"############################################################")

    butianyun_show_options()
    butianyun_usage()

    parser = argparse.ArgumentParser(description='目标检测')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--predict', action='store_true', help='预测图像')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--dataset_path', type=str, default=None, help='数据集路径')
    parser.add_argument('--predict_img_path', type=str, default=None, help='预测图像路径')
    args = parser.parse_args()

    dataset_path = BUTIANYUN_TRAIN_DATASET_PATH
    model_path = BUTIANYUN_TRAIN_MODEL_PATH

    run_mode_list= []
    run_mode = ""
    if args.train:
        run_mode_list.append('TRAIN')
    if args.test:
        run_mode_list.append('TEST')
    if args.predict:
        run_mode_list.append('PREDICT')
    
    if len(run_mode_list) == 0:
        run_mode_list = BUTIANYUN_RUN_MODE

    if args.dataset_path:
        dataset_path = args.dataset_path

    if args.model_path:
        model_path = args.model_path

    predict_img_path_list = BUTIANYUN_PREDICT_IMAGE_PATH
    if args.predict_img_path:
        predict_img_path_list = args.predict_img_path.split(',', 10)

    P(f"run_mode={run_mode_list}")
    ds = butianyun_object_detection_dataset_yolo(root_dir=dataset_path)
    butianyun_network_structure_dump(class_count=ds.butianyun_get_class_count())

    if run_mode_list.__contains__('TRAIN'):
        butianyun_cv_object_detection_train(dataset_path=dataset_path,model_path=model_path)
    
    if run_mode_list.__contains__('TEST'):
        butianyun_cv_object_detection_test(dataset_path=dataset_path, model_path=model_path)

    if run_mode_list.__contains__('PREDICT'):
        butianyun_cv_object_detection_predict(predict_img_path_list, model_path=model_path, dataset_path=dataset_path)



if __name__ == "__main__":
    butianyun_main()

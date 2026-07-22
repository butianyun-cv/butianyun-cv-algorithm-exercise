# -*- coding: utf-8 -*-
# butianyun_utils.py
# 实现功能：通用实用函数。


############################################################
#   微信公众号：计算机视觉技术
#   微信视频号：计算机视觉技术
#   网站         ：BUTIANYUN.COM
############################################################

import sys
import os
import cv2
from datetime import datetime


def butianyun_get_mei_path():
    try:
        path = sys._MEIPASS
    except Exception:
        path = os.path.abspath(".")
    return path


def butianyun_get_cascade_xml_path(xml_path):
    cascade_path = cv2.data.haarcascades + xml_path
    if os.path.exists(cascade_path):
        return cascade_path
    else:
        return os.path.join(butianyun_get_mei_path(), 'data', xml_path)

butianyun_start_time_print = datetime.now()

def P(msg):
    delta = datetime.now() - butianyun_start_time_print
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = delta.microseconds // 1000
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    print(f"{timestamp} {msg}")


def butianyun_get_cpu_count():
    return os.cpu_count() or 1

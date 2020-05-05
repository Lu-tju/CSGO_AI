from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from PIL import ImageGrab
import pyautogui


def arg_parse():
    """
    视频检测模块的参数转换

    """
    # 创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型
    parser = argparse.ArgumentParser(description='YOLO v3 检测模型')
    parser.add_argument("--bs", dest="bs", help="Batch size，默认为 1", default=1)
    parser.add_argument("--confidence", dest="confidence", help="目标检测结果置信度阈值", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS非极大值抑制阈值", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="配置文件", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="模型权重", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="网络输入分辨率. 分辨率越高,则准确率越高; 反之亦然", default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="待检测视频目录", default="video.avi", type=str)

    return parser.parse_args()


'''读取这些预设参数进来'''

args = arg_parse()  # args是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
MouseX, MouseY = pyautogui.position()   # 当前鼠标位置，防止CSGO窗口位置不准造成偏差
CUDA = torch.cuda.is_available()  # GPU环境是否可用

num_classes = 80  # coco 数据集有80类

# 初始化网络并载入权重
print("载入神经网络....")
model = Darknet(args.cfgfile)  # Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
model.load_weights(args.weightsfile)  # 将权重文件载入，并复制给对应的网络结构model中
print("模型加载成功.")

# 网络输入数据大小
model.net_info["height"] = args.reso  # 416  model类中net_info是一个字典。’’height’’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
inp_dim = int(model.net_info["height"])  # 416  inp_dim是网络输入图片尺寸（如416*416）
assert inp_dim % 32 == 0  # 如果设定的输入图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim > 32

# 如果GPU可用, 模型切换到cuda中运行
if CUDA:
    model.cuda()

# 变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
model.eval()

while 1:
    frame = ImageGrab.grab((0, 32, 1280, 500))  # 读入成功，ret=1，frame为图片
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # Img——>cv2

    img = prep_image(frame, inp_dim)  # 1*3*416*416

    im_dim = frame.shape[1], frame.shape[0]  # 保存原始大小：(640, 480)
    im_dim = torch.FloatTensor(im_dim).repeat(1, 2)  # 重复一次 [640,480,640,480]对应[x1,y1,x2,y2]

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    # 只进行前向计算，不计算梯度
    with torch.no_grad():
        # 得到每个预测方框在输入网络图片(416x416)坐标系中的坐标和宽高以及目标得分以及各个类别得分(x,y,w,h,s,s_cls1,s_cls2...)
        # 并且将tensor的维度转换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
        output = model(Variable(img), CUDA)  # 1*10647*85
    # 将方框属性转换成(ind,x1,y1,x2,y2,s,s_cls,index_cls)，去掉低分，NMS等操作，得到在输入网络坐标系中的最终预测结果
    # ind 是这个方框所属图片在这个batch中的序号，
    #  (第几张，左上x，左上y，右下x，右下y，bbox置信度，目标种类得分，什么物体）
    output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)  # 目标数n*8


    # 如果没有对象
    if type(output) == int:
        continue  # 不再执行以下，重新读图

    ''' 坐标转换为实际原图坐标'''
    # 将图片的尺寸的行数 重复 对象的数量 次
    im_dim = im_dim.repeat(output.size(0), 1)
    # 得到每个方框所在图片缩放系数
    # scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)#这是源代码，下面是我修改的代码
    scaling_factor = torch.min(int(args.reso) / im_dim, 1)[0].view(-1, 1)
    # 将方框的坐标(x1,y1,x2,y2)转换为相对于填充后的图片中包含原始图片区域（如416*312区域）的计算方式。
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
    # 将坐标映射回原始图片
    output[:, 1:5] /= scaling_factor
    # 将超过了原始图片范围的方框坐标限定在图片范围之内
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    # (第几张，左上x，左上y，右下x，右下y，bbox置信度，目标种类得分，什么物体）
    output = output[output[:, 7] == 0, :]  # 只保存是 人 的

    if output.shape[0] == 0:    # 如果没有人（有物体）
        continue  # 不再执行以下，重新读图

    # 对最近的目标操作：
    x = output[:, 1] + output[:, 3]
    x = x / 2 - MouseX
    null, index_x = torch.min(abs(x), 0)
    x = int(x[index_x])

    # y = output[0][2] + output[0][4]
    y = 0.6*output[index_x][2] + 0.4*output[index_x][4]     # 具体0.4、0.6的关系看情况，打头还是哪儿
    # y = int(y) / 2 - 392
    y = int(y) - MouseY

    currentMouseX = x / 1.56
    currentMouseY = y / 1.56

    A = time.time()
    pyautogui.moveRel(currentMouseX, currentMouseY)     # 这两步占用一半时间（各0.1s)
    B = time.time()
    pyautogui.click()
    C = time.time()
    # time.sleep(0.02)  # 开几枪也是自己改
    # pyautogui.click()
    # time.sleep(0.02)
    # pyautogui.click()
    # time.sleep(0.02)
    # pyautogui.click()
    time.sleep(0.2)
    print(B-A, C-B)

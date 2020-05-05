from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *  # 把一个模块中所有的函数都导入进来


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # unit8转换浮点型: ndarray to tensor
    img_ = Variable(img_)  # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    """
    读取cfg文件,即神经网络各个参数
    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型 对应于一个要建立的神经网络模块（层）
    """
    # 加载文件并过滤掉文本中多余内容
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list等价于readlines
    lines = [x for x in lines if len(x) > 0]  # 去掉空行
    lines = [x for x in lines if x[0] != '#']  # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    # cfg文件中的每个模块(层)用[]括起来最后组成一个列表，即每个层用一个字典block存储。
    block = {}  # 存放每个模块
    blocks = [] # 存放各个模块列表

    for line in lines:  # 每一行步进
        if line[0] == "[":  # 这是cfg文件中一个层(块)的开始,遇到[ ,开始一个新的块
            if len(block) != 0:  # 如果块不空, 说明是上一个块刚结束还没有保存
                blocks.append(block)  # 把这个块加入到blocks列表中去
                block = {}  # 清空block
            block["type"] = line[1:-1].rstrip()  # 把cfg的[]中的名作为block的一个属性(键 key):type
        else:   # 如果不是层的开始
            key, value = line.split("=")  # 按等号分割,将属性读入并存储:batch=1 --- batch和1
            block[key.rstrip()] = value.lstrip()  # 左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对
    blocks.append(block)  # 退出循环，将最后一个未加入的block加进去
    return blocks


# 配置文件定义了6种不同type
# 'net': 相当于超参数,网络全局配置的相关参数
# {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}

# cfg = parse_cfg("cfg/yolov3.cfg")     # 如此输入
# print(cfg)


class EmptyLayer(nn.Module):
    """
    为shortcut layer / route layer 准备, 具体功能不在此实现，在Darknet类的forward函数中有体现
    """

    def __init__(self):  # 继承nn.Module里的__init__属性
        # 首先找到EmptyLayer的父类nn.Module，然后把类EmptyLayer的对象self转换为父类nn.Module的对象，nn.Module(转换后)调用自己的__init__函数
        super(EmptyLayer, self).__init__()  # 继承主模块进来，固定格式


class DetectionLayer(nn.Module):
    '''yolo 检测层的具体实现, 在特征图上使用锚点预测目标区域和类别, 功能函数在predict_transform中'''

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors      #  作为DetectionLayer的参数anchors

''' 创建网络结构 '''
def create_modules(blocks):
    net_info = blocks[0]  # blocks[0]存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息
    module_list = nn.ModuleList()  # module_list用于存储每个block,每个block对应cfg文件中一个块，类似[convolutional]里面就对应一个卷积块
    # 卷积核的深度是由上一层的卷积核数量（或特征图深度）决定的。这意味着我们需要持续追踪被应用卷积层的卷积核数量。我们使用变量 prev_filter 来做这件事。我们将其初始化为 3，因为图像有对应 RGB 通道的 3 个通道。
    prev_filters = 3  # 初始值对应于输入数据3通道，用来存储我们需要持续追踪被应用卷积层的卷积核数量  （上一层的卷积核数量（或特征图深度））
    # 路由层（route layer）从前面层得到特征图（可能是拼接的）。如果在路由层之后有一个卷积层，那么卷积核将被应用到前面层的特征图上，精确来说是路由层得到的特征图。
    output_filters = []  # 我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。

    # 现在，我们的思路是迭代模块的列表，并为每个模块创建一个 PyTorch 模块。
    for index, x in enumerate(blocks[1:]):  # 这里，我们迭代block[1:] 而不是blocks，因为blocks的第一个元素是一个net块，它不属于前向传播。index是不断循环，记录第几层
        module = nn.Sequential()  # 这里每个块用nn.sequential()创建为了一个module,一个module有多个层。sequential：连续的，如，一个卷积模块包含有一个批量归一化层、一个 leaky ReLU 激活层以及一个卷积层。

        # check the type of block
        # create a new module for the block
        # append to module_list

        if (x["type"] == "convolutional"):  # x块的type属性如果是conv
            ''' 1. 卷积层 包含卷积、BN、leaky '''
            # 获取激活函数/批归一化/卷积层参数（通过字典的键获取值）
            activation = x["activation"]    # 激活函数=x的activation属性（0/1）
            try:
                batch_normalize = int(x["batch_normalize"])     # BN=x的batch_normalize属性（0/1）
                bias = False  # 卷积层后接BN就不需要bias
            except:
                batch_normalize = 0
                bias = True  # 卷积层后无BN层就需要bias

            # 各项参数
            filters = int(x["filters"])     # 卷积核的个数
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2        # padding保证卷积后大小不变
            else:
                pad = 0

            # 开始创建并添加相应层
            # Add the convolutional layer
            # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)    # 创建Conv层
            module.add_module("conv_{0}".format(index), conv)   # 将刚刚的conv层添加到module块，括号是命名下索引的意思，如conv_0,conv_1...

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)    # 输入参数为通道数
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            # 给定参数负轴系数0.1
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
# 结果：
#         Sequential(
#             (conv_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#             (batch_norm_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky_1): LeakyReLU(negative_slope=0.1, inplace)
#         )

        elif (x["type"] == "upsample"):
            '''
            2. upsampling layer
            没有使用 Bilinear2dUpsampling
            实际使用的为最近邻插值
            '''
            stride = int(x["stride"])  # 这个stride在cfg中就是2，所以下面的scale_factor写2或者stride是等价的
            upsample = nn.Upsample(scale_factor=2, mode="nearest")      # 最邻近插值，大小扩x2
            module.add_module("upsample_{}".format(index), upsample)    # 添加进module块

        # route layer -> Empty layer
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')    # 以,为分隔，进行切片
            # Start  of a route
            start = int(x["layers"][0])     # 第一个存start
            # end, if there exists one.

            # [route]可能有1个layers输入或2个
            # layers = -4
            # # 它输出这一层之前的第四个层的特征图。
            # layers = -1, 61
            # # 输出其前一层级与第 61 层的特征图，并将它们按深度拼接起来

            try:
                end = int(x["layers"][1])   # 如果有两个，第二个存end里
            except:
                end = 0
            # Positive anotation: 正值
            if start > 0:
                start = start - index
            if end > 0:  # 若end>0，由于end= end - index，再执行index + end输出的还是第end层的特征
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:  # 若end<0，前边end-index，现在又加index，所以还是end
                filters = output_filters[index + start] + output_filters[index + end]   # 如上例，第86层是route，输入为-1，61。这里即将85层卷积核个数和61层个数相加
            else:  # 如果没有第二个参数，end=0，则对应下面的公式，此时若start>0，由于start = start - index，再执行index + start输出的还是第start层的特征;若start<0，则start还是start，输出index+start(而start<0)故index向后退start层的特征。
                filters = output_filters[index + start]     # output_filters存储各层卷积核个数。例如，第36层是route，输入-1，即这里返回35层卷积核个数
        # shortcut corresponds to skip connection
        elif x["type"] == "shortcut":   # 好像空层就是用来占位的，表示这里有个层，使层的顺序别乱了
            shortcut = EmptyLayer()  # 使用空的层，因为它还要执行一个非常简单的操作（加）。没必要更新 filters 变量,因为它只是将前一层的特征图添加到后面的层上而已。
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer，就做了一件事：预处理anchor，分成对
        elif x["type"] == "yolo":
            # mask:使用哪一组预选框，例如：678组
            mask = x["mask"].split(",")     # ['6' '7' '8']
            mask = [int(x) for x in mask]   # [6,7,8]

            anchors = x["anchors"].split(",")   # 最终处理成坐标对的形式，并保留678组
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)  # 锚点,检测,位置回归,分类，这个类见predict_transform中
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)      # 添加到module_list层表
        prev_filters = filters          # 滤波器个数存储
        output_filters.append(filters)  # 添加到滤波器个数表

    return (net_info, module_list)

'''
利用creat_modules创建网络层结构，
他包含了预设的卷积层（Conv、BN、leaky）和上采样层，前向传播时，直接把数据x不断往里扔并更新即可
对于route、shortcut、yolo层，没有预设，需要在前向传播中自己定义计算过程
'''

''' 实现网络的前向传播 '''
class Darknet(nn.Module):       # 利用nn.Module 类别创建子类，并将我们的创建的类命名为 Darknet
    def __init__(self, cfgfile):    # 它需要有一个参数输入："cfg/yolov3.cfg"
        super(Darknet, self).__init__()     # 继承Darknet父类nn.Module中的主模块进来
        self.blocks = parse_cfg(cfgfile)  # 调用parse_cfg函数，self.blocks的意思是：为函数__init__上级类Darknet创建参数blocks，作为darknet的参数
        self.net_info, self.module_list = create_modules(self.blocks)  # 调用create_modules函数

    def forward(self, x, CUDA):     # 这个x是带着batch的，可以多张图一次性通过网络
        # 网络结构：从cfg中度入的modules，然后根据是那一层，执行具体的前向传播计算
        modules = self.blocks[1:]  # 除了net块之外的所有，forward这里用的是blocks列表中的各个block块字典
        outputs = {}  # 由于路由层和捷径层需要之前层的输出特征图，我们在字典 outputs 中缓存每个层的输出特征图

        write = 0  # write表示我们是否遇到第一个检测。write=0，则收集器尚未初始化，write=1，则收集器已经初始化，我们只需要将检测图与收集器级联起来即可。
        for i, module in enumerate(modules):    # i表示第几层，module表示是什么层。依次迭代cfg文件度入的blocks
            module_type = (module["type"])      # 什么层

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)      # 如果该模块是一个卷积层或上采样层，那么前向传播就直接扔进固定网络结构中传播就行了

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                # 如果只有一层时。从前面的if (layers[0]) > 0:语句中可知，如果layer[0]>0，则输出的就是当前layer[0]这一层的特征,如果layer[0]<0，输出就是从route层(第i层)向后退layer[0]层那一层得到的特征
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                # 第二个元素同理
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    ''' 仍以86层：-1，61为例 '''
                    map1 = outputs[i + layers[0]]   # map1 = 85层特征图
                    map2 = outputs[i + layers[1]]   # map2 = 61层特征图
                    x = torch.cat((map1, map2), 1)  # 第二个参数设为 1,这是因为我们希望将特征图沿深度级联起来。


            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]  # 求和运算，它将前一层的特征图和指定层相加（如-4，就是前面第4层）

            elif module_type == 'yolo':
                # yolo层：主要是将得到的tx,ty,th,tw转换成实际坐标框
                # 得到anchors框锚点（3对/次）
                anchors = self.module_list[i][0].anchors    # module_list[i]是第82层Detection_82，module_list[i][0]是他的层DetectionLayer()
                # 从net_info(实际就是blocks[0]，即[net])中get the input dimensions
                inp_dim = int(self.net_info["height"])  # inp_dim=416

                # Get the number of classes
                num_classes = int(module["classes"])    # 种类：80种

                # Transform
                x = x.data  # 这里得到的是预测的yolo层feature map，如最大的图x是：1*13*13*255
                # x即论文所要的输出（每个grid所对应的anchor坐标与宽高，以及出现目标的分数值与每种类别的值）
                # 经过predict_transform变换后的x：
                # 每一行为一个anchor，它是将x（例如13*13*255）展开，即共3*13*13=507个anchor
                # 每一行各个元素为每一个bbox的坐标、得分、各类可能（4+1+80个）
                # 同时计算每个方框在网络输入图片(416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)   # 1*507*85

                # 第一次得到各个bbox，先赋值，以后再拼接
                if not write:  # if no collector has been intialised. 因为一个空的tensor无法与一个有数据的tensor进行concatenate操作，
                    detections = x  # 所以detections的初始化在有预测值出来时才进行，
                    write = 1  # 用write = 1标记，当后面的分数出来后，直接concatenate操作即可。

                else:
                    '''
                    变换后x的维度是(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)，这里是在维度1上进行concatenate，即按照
                    anchor数量的维度进行连接，对应教程part3中的Bounding Box attributes图的行进行连接。yolov3中有3个yolo层，所以
                    对于每个yolo层的输出先用predict_transform()变成每行为一个anchor对应的预测值的形式(不看batch_size这个维度，x剩下的
                    维度可以看成一个二维tensor)，这样3个yolo层的预测值按照每个方框对应的行的维度进行连接。得到了这张图处所有anchor的预测值，后面的NMS等操作可以一次完成
                    '''
                    # 将在3个不同level的feature map上检测结果存储在 detections 里，按行连接下去
                    detections = torch.cat((detections, x), 1)  # 最终=1*10647*85 （10647=(52*52)+(26*26)+(13*13)]*3）

            outputs[i] = x

        return detections

    # blocks = parse_cfg('cfg/yolov3.cfg')
    # x,y = create_modules(blocks)
    # print(y)

    ''' 加载权重 '''
    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)  # 这里读取first 5 values权重
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)  # 加载 np.ndarray 中的剩余权重，权重是以float32类型存储的

        ptr = 0     # 追踪我们在权重数组中的位置
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]  # blocks中的第一个元素是网络参数和图像的描述，所以从blocks[1]开始读入

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])  # 当有bn层时，"batch_normalize"对应值为1
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model 将从weights文件中得到的权重bn_biases复制到model中(bn.bias.data)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:  # 如果 batch_normalize 的检查结果不是 True，只需要加载卷积层的偏置项
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


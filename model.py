import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *


class Net(nn.Module):

    # 可改-定义网络的普通操作
    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1, groups=20)
        self.fc = nn.Linear(5*5*40, 10)

    # 可改-网络的普通前向传播
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        return x

    # 可改-量化：设置量化函数的功能
    # 只有第一层的 conv 需要 qi，后面的模块基本是复用前面层的 qo 作为当前层的 qi
    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU()
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU()
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    # 量化网络前向传播；统计 min、max，同时模拟量化误差
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5*5*40)
        x = self.qfc(x)
        return x

    # 在统计完 min、max 后对一些变量进行固化
    # 在量化网络的时候，有些模块是没有定义 qi 的，因此这里需要传入前面层的 qo 作为当前层的 qi
    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    # 实际 inference 的时候用到的函数
    # 将输入 x 先量化到 int8，然后就是全量化的定点运算，得到最后一层的输出后，再反量化回 float 即可
    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5*5*40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class NetBN(nn.Module):

    # 可改-定义网络的普通操作
    def __init__(self, num_channels=1):
        super(NetBN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.bn1 = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.bn2 = nn.BatchNorm2d(40)
        self.fc = nn.Linear(5 * 5 * 40, 10)

    # 可改-网络的普通前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 40)
        x = self.fc(x)
        return x

    # 量化：设置量化函数的功能
    # 只有第一层的 conv 需要 qi，后面的模块基本是复用前面层的 qo 作为当前层的 qi
    # 卷积、BN、relu融合为一个
    def quantize(self, num_bits=8):
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    # 量化网络前向传播
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5*5*40)
        x = self.qfc(x)
        return x

    # 固化一些变量
    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    # 量化推理
    # 将输入 x 先量化到 int8，然后就是全量化的定点运算，得到最后一层的输出后，再反量化回 float 即可
    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5*5*40)

        qx = self.qfc.quantize_inference(qx)
        
        out = self.qfc.qo.dequantize_tensor(qx)
        return out

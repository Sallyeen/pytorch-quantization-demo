import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from function import FakeQuantize

# 1/11，按照uint，计算scale和zeropoint，都是浮点
# min_val, max_val需要根据具体情况专门统计，但是符号非符号的距离都一样
def calcScaleZeroPoint(min_val, max_val, num_bits=8, signed=False):
    if signed: # 举个例子，对于int8，qmin=-128. ,qmax=127.
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else: # 举个例子，对于uint8，qmin=0. ,qmax=255.
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin) # S=(rmax-rmin)/(qmax-qmin)，scale是浮点数

    zero_point = qmax - max_val / scale # Z=round(qmax-rmax/scale)，zeropoint也是浮点数

    if zero_point < qmin: # 意味着min_val是正数，0<rmin
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax: # 意味着max_val是正数，rmax<0
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)

    zero_point.round_()
    # 1.34---[1.34]<-{torch,tensor}->tensor([1.3400])<-{round_()}->tensor([1.])
    return scale, zero_point

# 2/11，量化：借助x，s,z，计算量化后的qx值，都是浮点
def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed: # 举个例子，对于int8，qmin=-128. ,qmax=127.
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else: # 举个例子，对于uint8，qmin=0. ,qmax=255.
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_() # q=round(r/S+Z)
    
    return q_x # 由于pytorch不支持int类型的运算，因此我们还是用float来表示整数
    # 尝试一番，目前结论是不支持int的除法

# 3/11，反量化：借助qx，s，z，计算反量化后的x值，都是浮点
def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point) # r=S(q-Z)

# 4/11，寻找符合误差范围的n与M0值:(S1*S2)/S3表示为M，M=M0*2^(-n)
def search(M):
    P = 7000
    n = 1
    while True: # 虽没有break，但满足条件时，return就可跳出循环
        Mo = int(round(2 ** n * M)) # int型
        # Mo 
        approx_result = Mo * P >> n # approx_result=Mo*P*2^(-n)，其中Mo=M*2^n并四舍五入
        result = int(round(M * P)) # result=M*P
        error = approx_result - result

        print("n=%d, Mo=%f, approx=%d, result=%d, error=%f" % \
            (n, Mo, approx_result, result, error))

        if math.fabs(error) < 1e-9 or n >= 22: # 最多让M左移22位或者误差小于10^(-9)
            return Mo, n
        n += 1

# 封装了量化反量化等，更新maxmin，量化，反量化，取出scale zeropoint min max，info串起来
class QParam(nn.Module):

    def __init__(self, num_bits=8):
        super(QParam, self).__init__() # 继承父类，且继承父类属性，不完全重写
        self.num_bits = num_bits
        # 声明scale zero_point min max， 不进行梯度更新
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    # 统计样本，更新min、max值，且计算scale和zeropoint
    def update(self, tensor):

        # 更新max且让max非负：nelment可以统计张量个数；如果没有最大值或者max不是最大值，就更新
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0) # 让max不为负：clamp_的作用是设定最大最小值，把框外拉进到框内
        # 更新min且让min非正：
        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits, False)

    # 量化：借助tensor，s,z，计算量化后的tensor值
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    # 反量化：借助 量化后的tensor值，s，z，计算反量化后的tensor值
    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    # 从统一的参数存储中，分别取出scale zeropoint min max
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_names = ['scale', 'zero_point', 'min', 'max']
        for key in key_names:
            value = getattr(self, key) # 得到属性值
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    # __str__会自动调用；info为一个字符串，包括scale zeropoint min max
    def __str__(self):
        info = 'scale: %.10f ' % self.scale # scale精确到小数点后10位
        info += 'zp: %d ' % self.zero_point # zero_point整体print
        info += 'min: %.6f ' % self.min # min精确到小数点后6位
        info += 'max: %.6f' % self.max # max精确到小数点后6位
        return info

# 下面实现基本网络模块的量化形式

# 没看懂-量化基类，减少重复代码，让代码结构更加清晰
class QModule(nn.Module):

    # 指定量化的位数；指定是否提供量化输入 (qi) 及输出参数 (qo)。不是每一网络模块都需要统计输入的 min、max，大部分中间层
    # 都是用上一层的 qo 来作为自己的 qi 的，另外有些中间层的激活函数也是直接用上一层的 qi 来作为自己的 qi 和 qo
    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits) # 若接收qi，则为info s,z,min,max
        if qo:
            self.qo = QParam(num_bits=num_bits) # 若接收qo，则为info s,z,min,max

    #　在统计完 min、max 后发挥作用。可把一些项提前固定下来，同时将网络的权重由浮点实数转化为定点整数
    def freeze(self):
        pass

    # 量化 inference 的时候会使用。实际 inference 的时候和正常的 forward 会有一些差异　
    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')

# 量化卷积层的实现
class QConv2d(QModule):

    # 传入一个 conv_module 模块对应全精度的卷积层，qw 统计weight 的 min、max 以及对 weight 进行量化用
    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits) # qw为info s,z,min,max

    # 计算M、qw、qb，对公式4进行加速
    def freeze(self, qi=None, qo=None):
        # 报错情况：有qi且qi非空；没有qi且qi为空
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        # 报错情况：有qo且qo非空；没有qo且qo为空
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')
        
        # qi非空，把qi传给self.qi；qo为空，把qo传给self.qo;随后计算M
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.M = self.qw.scale * self.qi.scale / self.qo.scale
        # 量化权重，(r/s+z)，后为r/s
        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        # 量化bias,量化 r/s 为float32，zp为零
        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data,
                                                     scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    # 同正常的forward一样在float进行，需要统计输入输出以及 weight 的 min、max
    def forward(self, x):
        
        # 有qi，则用x更新qi对应参数,FakeQuantize
        if hasattr(self, 'qi'):
            self.qi.update(x) # 统计样本x，更新qi：min、max值，且计算scale和zeropoint
            x = FakeQuantize.apply(x, self.qi) # 用qi对x进行假量化

        # 用conv_module.weight更新qw对应参数
        self.qw.update(self.conv_module.weight.data) # 统计样本权重，更新qw：min、max值，且计算scale和zeropoint

        # 定义卷积：x，bias不变，权重做假量化
        x = F.conv2d(x, FakeQuantize.apply(self.conv_module.weight, self.qw), self.conv_module.bias, 
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                     groups=self.conv_module.groups)

        # 有qo，则用卷积后的x更新qo,FakeQuantize
        if hasattr(self, 'qo'):
            self.qo.update(x) # 统计卷积后的x，更新qo：min、max值，且计算scale和zeropoint
            x = FakeQuantize.apply(x, self.qo) # 用qo进行假量化

        return x

    # 在实际 inference 的时候会被调用。卷积操作在int（float整数）上进行，对应公式7
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point        
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x


class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits) # 得到qw的参数

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                   zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw), self.fc_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x


class QReLU(QModule):

    def __init__(self, qi=False, num_bits=None): # qi、num_bits为QReLU的属性，具有默认值
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        # 有qi，则用x更新qi,FakeQuantize
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.relu(x)

        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x

class QMaxPooling2d(QModule):

    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def quantize_inference(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class QConvBNReLU(QModule):

    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias


    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        if self.training:
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
            # mean = y.mean(1)
            # var = y.var(1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var
        else:
            mean = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)

        std = torch.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.qw.update(weight.data)

        x = F.conv2d(x, FakeQuantize.apply(weight, self.qw), bias, 
                stride=self.conv_module.stride,
                padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                groups=self.conv_module.groups)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        weight, bias = self.fold_bn(self.bn_module.running_mean, self.bn_module.running_var)
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point        
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x
        

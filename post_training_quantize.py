from torch.serialization import load
from model_ptq import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp
import time

# ------------------------------普通训练之后得到预训练模型，在此进行训练后量化---------------------
# 用一些 测试数据 来估计 min、max
def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data) # 对测试数据集，用量化的模型推理
        if i % 200 == 0: # 当i=500时，跳出循环，i表示batch个数
            break
    print('direct quantization finish')

# 完全推理：用一些 测试数据 来进行测试，并输出准确度分数；与普通TEST一样，区别只有没有device和loss
def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data) # 普通的模型输出
        pred = output.argmax(dim=1, keepdim=True) # 第一维最大值的索引，即预测出的最有可能的结果的索引
        correct += pred.eq(target.view_as(pred)).sum().item() # 对于普通模型，all data，预测对了就加一
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

#　量化推理
def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        # print(data)
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    batch_size = 64
    # using_bn = True # 定义是否使用BN
    using_bn = False
    load_quant_model_file = "ckpt/mnist_cnnbn_ptq.pt"
    # load_model_file = None

    # -----------------加载训练与测试数据---------------------------
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # -----------------加载模型，确定保存路径---------------------------
    # 加载全精度模型的参数，选择普通训练的模型路径和确定量化后的模型保存路径
    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load('ckpt/mnist_cnnbn.pt', map_location='cpu'))
        save_file = "ckpt/mnist_cnnbn_ptq.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load('ckpt/mnist_cnn.pt', map_location='cpu'))
        save_file = "ckpt/mnist_cnn_ptq.pt"

    model.eval()
    start = time.clock()
    full_inference(model, test_loader) # 用测试集测试模型的预测准确度,打印Test set: Full Model Accuracy:
    end = time.clock()
    print('whole time:', round(end - start, 5))
    # print(model.state_dict())
    #print model's state_dict
    '''print('Model.state_dict:')
    for param_tensor in model.state_dict():
        #打印 key value字典
        print(param_tensor,'\t',model.state_dict()[param_tensor].size())'''


    num_bits = 8
    model.quantize(num_bits=num_bits) # 对网络进行量化，逐个量化每个模块
    model.eval()
    print('Quantization bit: %d' % num_bits) # 打印选择的量化位数

    # print(load_quant_model_file) ckpt/mnist_cnnbn_ptq.pt
    # 加载量化后的模型
    #if osp.exists('load_quant_model_file'):
    #if load_quant_model_file is not None:
    '''if osp.isfile(load_quant_model_file):
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)'''
    torch.save(model.state_dict(), save_file) # 保存量化后的模型
    # 用一些 测试数据 泡一下流程，来估计 min、max
    direct_quantize(model, train_loader)

    
    model.freeze() # 把量化参数都固定下来，并进行全量化推理

    # 量化推理
    start = time.clock()
    quantize_inference(model, test_loader)
    end = time.clock()
    print('whole time:', round(end - start, 5))
    # print(model.state_dict())
    '''print('Model.state_dict:')
    for param_tensor in model.state_dict():
        #打印 key value字典
        print(param_tensor,'\t',model.state_dict()[param_tensor].size())'''
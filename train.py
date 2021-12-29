from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

# ------------------------------普通训练【使用普通forward】与测试----------------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss() # 确定损失函数
    for batch_idx, (data, target) in enumerate(train_loader): # 按batchsize不断取batch训练
        data, target = data.to(device), target.to(device) # 待训练数据与标签
        optimizer.zero_grad()
        output = model(data) # 模型的输出
        loss = lossLayer(output, target) # 一个batch的训练损失
        loss.backward() # 损失反向传播
        optimizer.step() # 优化 

        if batch_idx % 50 == 0: # 每50个batch打印一次损失
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader: # 按batchsize不断取batch测试
        data, target = data.to(device), target.to(device) # 待测试数据与标签
        output = model(data)
        test_loss += lossLayer(output, target).item() # 之前batch的测试损失
        pred = output.argmax(dim=1, keepdim=True) # 因为MNIST为十分类，选择可能性最大的预测的 索引
        correct += pred.eq(target.view_as(pred)).sum().item() # 预测对了的样本总个数
    
    test_loss /= len(test_loader.dataset) # 每个样本的平均损失

     # 打印每一个epoch的测试的平均损失，和精确度分数
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 15
    lr = 0.01
    momentum = 0.5
    save_model = True
    using_bn = True

    torch.manual_seed(seed) # 设置CPU生成随机数的种子，方便下次复现实验结果,因为下次跟本次一样

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    ) # 加载训练数据集

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True
    ) # 加载测试数据集

    # 根据是否使用BN，决定要使用的模型
    if using_bn:
        model = NetBN().to(device)
    else:
        model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1): # 从1到15，先train，再test
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # 根据save_model，决定是否保存模型
    if save_model:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        if using_bn:
            torch.save(model.state_dict(), 'ckpt/mnist_cnnbn.pt')
        else:
            torch.save(model.state_dict(), 'ckpt/mnist_cnn.pt')
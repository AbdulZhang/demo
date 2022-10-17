%matplotlib inline
from torch.nn import functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import time
from torch.utils.data import DataLoader

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),#灰度化
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)#训练集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)#测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

model = torchvision.models.resnet50(pretrained=False)

f=open('cifar10log.txt','a')

EPOCH=200
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters())

for epoch in range(EPOCH):
    start_time = time.time() #记录当前时间
    for i, data in enumerate(trainloader, start=0):
        # data里面包含图像数据（inputs）(tensor类型的）和标签（labels）(tensor类型）。
        inputs, labels = data
        # 将数据加载到相应设备中
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失函数
        loss = criterion(outputs, labels)
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
    print('epoch{} loss:{:.4f} time:{:.4f}'.format(epoch+1, loss.item(), time.time()-start_time))
    f.write('epoch '+str(epoch+1)+' '+'loss '+str(loss.item())+' '+'time: '+str(time.time()-start_time))
    f.write('\n')
    
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 前向传播
        out = model(images)
        # 求出预测值索引  torch.max(input, dim) dim=1，行最大值
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('10000测试图像 准确率:{:.4f}%'.format(100 * correct / total))
    f.write('10000测试图像 准确率:{:.4f}%'.format(100 * correct / total))


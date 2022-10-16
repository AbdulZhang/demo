import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 卷积层1.输入是32*32*3，计算（32-5）/ 1 + 1 = 28，那么通过conv1输出的结果是28*28*6
        self.conv1 = nn.Conv2d(3,30,5)  # imput:3 output:6, kernel:5
        # 池化层， 输入时28*28*6， 窗口2*2，计算28 / 2 = 14，那么通过max_poll层输出的结果是14*14*6
        self.pool = nn.MaxPool2d(2,2) # kernel:2 stride:2
        # 卷积层2， 输入是14*14*6，计算（14-5）/ 1 + 1 = 10，那么通过conv2输出的结果是10*10*16
        self.conv2 = nn.Conv2d(30,60,5) # imput:6 output:16, kernel:5
        # 全连接层1
        self.fc1 = nn.Linear(60*5*5, 120)  # input：16*5*5，output：120
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)  # input：120，output：84
        # 全连接层3
        self.fc3 = nn.Linear(84, 10)  # input：84，output：10
        
    def forward(self,x):
        # 卷积1
        '''
        32x32x3 --> 28x28x6 -->14x14x6
        '''
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积2
        '''
        14x14x6 --> 10x10x16 --> 5x5x16
        '''
        x = self.pool(F.relu(self.conv2(x)))
        # 改变shape
        x = x.view(-1,60*5*5)
        # 全连接层1
        x = F.relu(self.fc1(x))
        # 全连接层2
        x = F.relu(self.fc2(x))
        # 全连接层3
        
        x = self.fc3(x)
        return x 
net =Net()                  

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

f=open('cifar10log.txt','a')

EPOCH=200
net.load_state_dict(torch.load('net.pkl'))
for epoch in range(EPOCH):
    timestart = time.time()
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data
        
        #inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 500 == 499:
            E=epoch + 1
            l=running_loss / 500
            f.write(E)
            f.write(':')
            f.write(l)
            f.write('\n')
            print('[%d ,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
    print('epoch %d cost %3f sec' % (epoch + 1, time.time()-timestart))
    
    
print('Finished Training')

torch.save(net.state_dict(), 'net.pkl')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(testloader)
images, labels = dataiter.__next__()


outputs = net(Variable(images))
_, predicted = torch.max(outputs.data,1)


correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
f.write('Accuracy of the network on the 10000 test images:')
f.write(100*correct/total)
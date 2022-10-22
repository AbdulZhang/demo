import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexLayers import ComplexDropout2d, NaiveComplexBatchNorm2d
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

batch_size = 64
n_train = 5000
n_test = 1000
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
train_set = Subset(train_set, torch.arange(n_train))
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)
test_set = Subset(test_set, torch.arange(n_test))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.bn2d = ComplexBatchNorm2d(10, track_running_stats=False)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(4 * 4 * 20, 500)
        self.dropout = ComplexDropout2d(p=0.3)
        self.bn1d = ComplexBatchNorm1d(500, track_running_stats=False)
        self.fc2 = ComplexLinear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.bn2d(x)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 20)
        x = self.fc1(x)
        x = self.dropout(x)
        x = complex_relu(x)
        x = self.bn1d(x)
        x = self.fc2(x)
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        return x


device = torch.device('cpu')
model = ComplexNet()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train\t Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )


def test(model, device, test_loader, optimizer, epoch):
    model.eval()
    cor_num=torch.tensor([0])
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        out_max = torch.max(output, -1)[1]
        cor_num+=(out_max == target).sum()
        if batch_idx % 15 == 14:
            print('Test\t Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(test_loader.dataset),
                100. * batch_idx / len(test_loader),
                loss.item())
            )
    print(cor_num)





model=torch.load('model')
# Run training on 4 epochs
EPOCH=2
# for epoch in range(EPOCH):
#     train(model, device, train_loader, optimizer, epoch)
#     if epoch==EPOCH-1:
#         test(model, device, test_loader, optimizer, epoch)

#torch.save(model,'model')
for i in iter(model.parameters()):
    print(i.shape)


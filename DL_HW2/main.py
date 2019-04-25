import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import CNN_Net


'''
# only use with jupyter notebook in vscode
import os
if 'DL_HW2' not in os.getcwd():
    os.chdir(os.getcwd()+'/DL_HW2')
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose(
    [transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0)),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.])])
train_dataset = ImageFolder('data/processed/train', transform=transforms)
valid_dataset = ImageFolder('data/processed/val', transform=transforms)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True, num_workers=3)
valid_dataloader = DataLoader(
    dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=3)

model = CNN_Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

nepoch = 10
loss_record = {'train_loss': [], 'train_acc': [], 'valid_acc': []}
for epoch in range(nepoch):
    # train
    model.train()
    running_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        running_loss += loss.item()*data.size(0)
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                (batch_idx + 1) * len(data), len(train_dataloader.dataset),
                100. * (batch_idx + 1) / len(train_dataloader), loss.item()))
    running_loss /= len(train_dataloader.dataset)
    loss_record['train_loss'].append(running_loss)
    acc = 100. * float(correct)/len(train_dataloader.dataset)
    loss_record['train_acc'].append(acc)
    # valid
    model.eval()
    loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(valid_dataloader):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(valid_dataloader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        loss, correct, len(valid_dataloader.dataset),
        100. * float(correct) / len(valid_dataloader.dataset)))
    acc = 100. * float(correct)/len(valid_dataloader.dataset)
    loss_record['valid_acc'].append(acc)

plt.figure()
plt.plot(loss_record['train_loss'])
plt.savefig('train_loss.png')
plt.figure()
plt.plot(loss_record['train_acc'])
plt.plot(loss_record['valid_acc'])
plt.savefig('acc.png')

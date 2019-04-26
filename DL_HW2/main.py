import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import Net_O, Net_K, Net_S, Net_F


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
    dataset=train_dataset, batch_size=256, shuffle=True, num_workers=6)
valid_dataloader = DataLoader(
    dataset=valid_dataset, batch_size=256, shuffle=False, num_workers=6)

nepoch = 300

model_class = [Net_O, Net_K, Net_S, Net_F]
for m in model_class:
    class_name = m.__name__
    model = m().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    loss_record = {'train_loss': [], 'train_acc': [], 'valid_acc': []}
    best_val_acc = 0
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
            if (batch_idx + 1) % 19 == 0:
                print('Train Epoch: {} [{}/{}] Loss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data),
                    len(train_dataloader.dataset), loss.item()))
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
        print('{}: Valid avg loss: {:.4f}, Acc: {}/{} ({:.3f}%)'.format(
            class_name, loss, correct, len(valid_dataloader.dataset),
            100. * float(correct) / len(valid_dataloader.dataset)))
        acc = 100. * float(correct)/len(valid_dataloader.dataset)
        loss_record['valid_acc'].append(acc)
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, '{}_best_model'.format(class_name))

    plt.figure()
    plt.plot(loss_record['train_loss'])
    plt.savefig('{}_train_loss.png'.format(class_name))
    plt.figure()
    plt.plot(loss_record['train_acc'])
    plt.plot(loss_record['valid_acc'])
    plt.savefig('{}_acc.png'.format(class_name))

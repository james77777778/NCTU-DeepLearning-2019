import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import VAE

# parameters
epochs = 100  # number of epochs of training
batchsize = 128  # size of the batches
cartoon_root = 'cartoon'  # root directory of the dataset
results_root = 'results'
lr = 1e-3  # initial learning rate
size = 32  # size of the data crop (squared assumed)

if not os.path.exists(results_root):
    os.makedirs(results_root)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Networks
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
# Dataset loader
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()])
cartoon_set = datasets.ImageFolder(cartoon_root, transform)
cartoon_loader = torch.utils.data.DataLoader(
    dataset=cartoon_set, batch_size=batchsize, shuffle=True, num_workers=2)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, train_loader, model, optimizer, loss_function):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def save(model):
    # Save models checkpoints
    torch.save(model.state_dict(), 'ckpt/VAE.pth')


if __name__ == "__main__":
    all_loss = []
    for epoch in range(1, epochs + 1):
        loss = train(epoch, cartoon_loader, model, optimizer, loss_function)
        all_loss.append(loss)
        save(model)
        with torch.no_grad():
            model.eval()
            sample = torch.randn(64, 64).to(device)
            model.batch_size = 64
            sample = model.vae_decode(sample).cpu()
            save_image(sample.view(64, 3, 32, 32),
                       'results/sample_' + str(epoch) + '.png')
    np.savetxt(
        "VAE_loss.csv", all_loss, delimiter=",", fmt='%f', header='VAE_loss')

    plt.style.use('classic')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.plot(all_loss)
    plt.savefig('DL_HW3/VAE_loss.png')
import os
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision
import numpy as np

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import weights_init_normal

import time
start_time = time.time()

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')

# parameters
# TODO : set up all the parameters
epochs = 100  # number of epochs of training
batchsize = 40  # size of the batches
animation_root = 'animation'  # root directory of the dataset
cartoon_root = 'cartoon'  # root directory of the dataset
lr = 0.0002  # initial learning rate
size = 32  # size of the data crop (squared assumed)
input_nc = 3  # number of channels of input data
output_nc = 3  # number of channels of output data
lambda_A = 10.0
lambda_B = 10.0
lambda_idt = 0.5

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Definition of variables
'''
# Networks
netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(output_nc)

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=lr,
    betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(
    netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(
    netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batchsize, input_nc, size, size)
input_B = Tensor(batchsize, output_nc, size, size)
target_real = Variable(Tensor(batchsize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchsize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
animation_set = torchvision.datasets.ImageFolder(animation_root, transform)
cartoon_set = torchvision.datasets.ImageFolder(cartoon_root, transform)
animation_loader = torch.utils.data.DataLoader(
    dataset=animation_set, batch_size=batchsize, shuffle=True, num_workers=2)
cartoon_loader = torch.utils.data.DataLoader(
    dataset=cartoon_set, batch_size=batchsize, shuffle=True, num_workers=2)
###################################
G_loss = []
DA_loss = []
DB_loss = []
# ##### Training ######
for epoch in range(1, epochs):
    i = 1
    print('epoch', epoch)
    for batch in zip(animation_loader, cartoon_loader):
        # Set model input
        A = torch.FloatTensor(batch[0][0])
        B = torch.FloatTensor(batch[1][0])
        real_A = Variable(input_A.copy_(A))
        real_B = Variable(input_B.copy_(B))

        '''
        Generators A2B and B2A
        '''
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # TODO : calculate the loss for the generators, and assign to loss_G
        # forward pass
        fake_B = netG_A2B(real_A)  # G_A(A)
        rec_A = netG_B2A(fake_B)   # G_B(G_A(A))
        fake_A = netG_B2A(real_B)  # G_B(B)
        rec_B = netG_A2B(fake_A)   # G_A(G_B(B))
        # Identity loss
        idt_A = netG_B2A(real_B)
        loss_idt_A = criterion_identity(idt_A, real_A) * lambda_A * lambda_idt
        idt_B = netG_A2B(real_A)
        loss_idt_B = criterion_identity(idt_B, real_B) * lambda_B * lambda_idt
        # GAN loss
        loss_G_A = criterion_GAN(netD_A(fake_A), target_real)
        loss_G_B = criterion_GAN(netD_B(fake_B), target_real)
        # Cycle loss
        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_A
        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_B

        loss_G = (
            loss_idt_A + loss_idt_B +
            loss_G_A + loss_G_B +
            loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()
        ###################################

        '''
        Discriminator A
        '''
        optimizer_D_A.zero_grad()

        # TODO : calculate the loss for a discriminator, and assign to loss_D_A
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_A_real = netD_A(real_A)  # Real
        loss_D_A_real = criterion_GAN(pred_A_real, target_real)
        pred_A_fake = netD_A(fake_A.detach())  # Fake
        loss_D_A_fake = criterion_GAN(pred_A_fake, target_fake)
        # Combined loss and calculate gradients
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        ###################################

        '''
        Discriminator B
        '''
        optimizer_D_B.zero_grad()

        # TODO : calculate the loss for the other discriminator,
        # and assign to loss_D_B
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_B_real = netD_B(real_B)  # Real
        loss_D_B_real = criterion_GAN(pred_B_real, target_real)
        pred_B_fake = netD_B(fake_B.detach())  # Fake
        loss_D_B_fake = criterion_GAN(pred_B_fake, target_fake)
        # Combined loss and calculate gradients
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        ###################################

        G_loss.append(loss_G.item())
        DA_loss.append(loss_D_A.item())
        DB_loss.append(loss_D_B.item())
        # Progress report
        if (i % 100 == 0):
            print(
                "loss_G : ", loss_G.data.cpu().numpy(),
                ",loss_D:", (
                    loss_D_A.data.cpu().numpy() + loss_D_B.data.cpu().numpy()))
            i = 0
        i = i+1
    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'ckpt/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'ckpt/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'ckpt/netD_A.pth')
    torch.save(netD_B.state_dict(), 'ckpt/netD_B.pth')

end_time = time.time()
print(
    'Total cost time',
    time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

# TODO : plot the figure

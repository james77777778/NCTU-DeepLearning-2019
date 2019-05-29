import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import VAE


# parameters
batchsize = 64
cartoon_root = 'cartoon'  # root directory of the dataset
results_root = 'results'
size = 32  # size of the data crop (squared assumed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model
model = VAE.to(device)
model.load_state_dict(torch.load('ckpt/VAE.pth'))

# Dataset loader
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor()])
cartoon_set = datasets.ImageFolder(cartoon_root, transform)
cartoon_loader = torch.utils.data.DataLoader(
    dataset=cartoon_set, batch_size=batchsize)
# Testing
model.eval()
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(cartoon_loader):
        if batch_idx > 0:
            break
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        save_image(
            data.view(batchsize, 3, 32, 32),
            'results/sample_ori.png')
        save_image(
            recon_batch.view(batchsize, 3, 32, 32),
            'results/sample_recon.png')

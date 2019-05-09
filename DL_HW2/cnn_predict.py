import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import Net_O


# create needed folder
needed_folder = ['results', 'models']
for f in needed_folder:
    if not os.path.exists(f):
        os.makedirs(f)
# run on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0)),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.])])
# dataset, dataloader
valid_dataset = ImageFolder('data/processed/val', transform=tf)
valid_dataloader = DataLoader(
    dataset=valid_dataset, batch_size=256, shuffle=False)
classes = valid_dataset.classes
model = Net_O().to(device)
best_model = torch.load('models/Net_O_best_model')
model.load_state_dict(best_model['model_state_dict'])
# valid
model.eval()
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
correct = 0
incorrect = 0
with torch.no_grad():
    for batch_idx, (datas, targets) in enumerate(valid_dataloader):
        datas = datas.to(device)
        targets = targets.to(device)

        outputs = model(datas)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == targets).squeeze()
        for i in range(targets.size(0)):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        if batch_idx < 3:
            for i, c_value in enumerate(c):
                if(c_value == 1):  # correct
                    correct = (datas[i], targets[i], predicted[i])
                else:  # incorrect
                    incorrect = (datas[i], targets[i], predicted[i])

for i in range(len(classes)):
    print('Accuracy of %9s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

correct_img = transforms.ToPILImage()(correct[0].cpu()).convert('RGB')
incorrect_img = transforms.ToPILImage()(incorrect[0].cpu()).convert('RGB')
correct_img.save('correct.png')
incorrect_img.save('incorrect.png')
print("correct: label={}, pred={}".format(
    classes[correct[1].cpu()], classes[correct[2].cpu()]))
print("incorrect: label={}, pred={}".format(
    classes[incorrect[1].cpu()], classes[incorrect[2].cpu()]))

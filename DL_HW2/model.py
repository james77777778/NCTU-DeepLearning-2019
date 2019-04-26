import torch.nn as nn


class Net_O(nn.Module):
    def __init__(self):
        super(Net_O, self).__init__()
        self.features = nn.Sequential(
            # 128
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 32
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 16
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# change kernel_size
class Net_K(nn.Module):
    def __init__(self):
        super(Net_K, self).__init__()
        self.features = nn.Sequential(
            # 128
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 63
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 31
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 15
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 15 * 15, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# change stride
class Net_S(nn.Module):
    def __init__(self):
        super(Net_S, self).__init__()
        self.features = nn.Sequential(
            # 128
            nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 43
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 15
            nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 5
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# change filter size
class Net_F(nn.Module):
    def __init__(self):
        super(Net_F, self).__init__()
        self.features = nn.Sequential(
            # 128
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

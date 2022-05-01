import torch
import torch.nn as nn
import config


class CNN(nn.Module):
    def __init__(self, num_classes=config.label_num):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            # 6
            # first linear equals to AvgPool * out_channel
            nn.Linear(128*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 7
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
from collections import OrderedDict

from torch import nn
from torchvision import models

# TODO: wie kann ich aus dem modell eine Klasse machen, die die Modell attribute und methoden besitzt,
# aber die Größe des letzten layers definiert werden kann bei init?

# def load_model(num_classes):

#     # model
#     model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

#     # # Unfreeze parameters
#     # for param in model.parameters():
#     #     param.requires_grad = True

#     # adapt output layer
#     fc = nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(1280, num_classes)),
#         ('output', nn.LogSoftmax(dim=1))
#         ]))

#     model.classifier[1] = fc

#     optimizer_layers = [model.classifier[1]]

#     return model, optimizer_layers


class Rateke_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Rateke_CNN, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )  # output is (32,64,64)
        # Weight initilization Layer 1
        # init.xavier_normal_(self.conv1[0].weight)
        # init.constant_(self.conv1[0].bias, 0.05)

        # Conv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )  # output is (32,32,32)

        # init.xavier_normal_(self.conv2[0].weight)
        # init.constant_(self.conv2[0].bias, 0.05)

        # Conv Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )  # output is (64, 16, 16)

        # init.xavier_normal_(self.conv3[0].weight)
        # init.constant_(self.conv3[0].bias, 0.05)

        self.flat = (
            nn.Flatten()
        )  # output (16384 when 128 image size, 65536 when 256 image size)

        self.fc1 = nn.Sequential(nn.Linear(65536, 256, bias=True), nn.ReLU())

        # init.xavier_normal_(self.fc1[0].weight)
        # init.constant_(self.fc1[0].bias, 0.05)

        self.fc2 = nn.Linear(256, num_classes, bias=True)
        # init.xavier_normal_(self.fc2.weight)
        # init.constant_(self.fc2.bias, 0.05)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def load(self):
        return self

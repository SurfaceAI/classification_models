import torch
from torch import nn, Tensor
from torchvision import models
from collections import OrderedDict


class CustomResnet50(nn.Module):
    def __init__(self, num_classes, avg_pool=1):
        super(CustomResnet50, self).__init__()

        model = models.resnet50(weights="IMAGENET1K_V1")
        # adapt output layer
        in_features = model.fc.in_features * (avg_pool * avg_pool)
        fc = nn.Linear(in_features, num_classes, bias=True)
        model.fc = nn.Sequential(OrderedDict([("fc", fc)]))

        feature_layers = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]
        self.features = nn.Sequential(
            OrderedDict([(layer, getattr(model, layer)) for layer in feature_layers])
        )

        self.avgpool = nn.AdaptiveAvgPool2d(avg_pool)
        self.classifier = model.fc
        if num_classes == 1:
            self.criterion = nn.MSELoss
        else:
            self.criterion = nn.CrossEntropyLoss

    @staticmethod
    def get_class_probabilies(x):
        return nn.functional.softmax(x, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def get_optimizer_layers(self):
        return self.classifier

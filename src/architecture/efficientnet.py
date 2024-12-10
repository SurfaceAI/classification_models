
import torch
from torch import nn, Tensor
from torchvision import models
from collections import OrderedDict

class CustomEfficientNetV2SLogsoftmax(nn.Module):
    def __init__(self, num_classes, avg_pool=1):
        super(CustomEfficientNetV2SLogsoftmax, self).__init__()

        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        # adapt output layer
        in_features = model.classifier[-1].in_features * (avg_pool * avg_pool)
        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, num_classes, bias=True)),
            ('output', nn.LogSoftmax(dim=1))   # criterion = nn.NLLLoss()
            ]))
        model.classifier[-1] = fc
        
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(avg_pool)
        self.classifier = model.classifier
        if num_classes == 1:
            self.criterion = nn.MSELoss
        else:
            self.criterion = nn.NLLLoss
        
    @ staticmethod
    def get_class_probabilies(x):
        return torch.exp(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
    
    def get_optimizer_layers(self):
        return self.classifier


class CustomEfficientNetV2SLinear(nn.Module):
    def __init__(self, num_classes, avg_pool=1):
        super(CustomEfficientNetV2SLinear, self).__init__()

        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        # adapt output layer
        in_features = model.classifier[-1].in_features * (avg_pool * avg_pool)
        fc = nn.Linear(in_features, num_classes, bias=True)
        model.classifier[-1] = fc
        
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(avg_pool)
        self.classifier = model.classifier
        if num_classes == 1:
            self.criterion = nn.MSELoss
        else:
            self.criterion = nn.CrossEntropyLoss
        
    @ staticmethod
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



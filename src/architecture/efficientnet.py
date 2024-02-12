import torch
from torch import nn, Tensor
from torchvision import models
from collections import OrderedDict

architecture = "Efficient Net"

class CustomEfficientNetV2SLogsoftmax(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetV2SLogsoftmax, self).__init__()

        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        # adapt output layer
        in_features = model.classifier[-1].in_features
        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, num_classes, bias=True)),
            ('output', nn.LogSoftmax(dim=1))   # criterion = nn.NLLLoss(), logits_to_prob = torch.exp()
            ]))
        model.classifier[-1] = fc
        
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
    
    def get_optimizer_layers(self):
        return self.classifier



import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

architecture = "VGG16"

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()

        # Load the pre-trained VGG16 model
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Freeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = False

        # Modify the classifier layer
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # select features in our last layer
        features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Save the modified model as a member variable
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
    
    def get_optimizer_layers(self):
        return self.classifier
        
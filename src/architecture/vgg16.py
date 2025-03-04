import torch
import torch.nn as nn
from torchvision import models

architecture = "VGG16"

class CustomVGG16(nn.Module):
    def __init__(self, num_classes, avg_pool):
        super(CustomVGG16, self).__init__()

        # Load the pre-trained VGG16 model
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Freeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = False

        # Modify the classifier layer
        num_features = model.classifier[6].in_features * (avg_pool * avg_pool)
        features = list(model.classifier.children())[:-1]  # select features in our last layer
        features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Save the modified model as a member variable
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

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
    
    def get_optimizer_layers(self):
        return self.classifier
        
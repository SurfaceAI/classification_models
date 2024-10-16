import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from src import constants as const
from coral_pytorch.losses import corn_loss
from multi_label.CLM import CLM

architecture = "VGG16"

class CustomVGG16(nn.Module):
    def __init__(self, num_classes, head):
        super(CustomVGG16, self).__init__()

        # Load the pre-trained VGG16 model
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Freeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True

        self.head = head
        self.num_classes = num_classes #this should always be the number of quality classes for each level
        # Define the common structure
# Shared classifier layers
        shared_classifier_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Define the final layer based on the head type
        if self.head == const.REGRESSION:
            top_layer = nn.Linear(1024, 1) 
            self.criterion = nn.MSELoss
            
        elif self.head == const.CLM:
            top_layer = nn.Sequential(
                nn.Linear(1024, 1),
                nn.BatchNorm1d(1),
                CLM(classes=num_classes, link_function="logit", min_distance=0.0, use_slope=False, fixed_thresholds=False)
            )
            self.criterion = nn.NLLLoss
            
        elif self.head == const.CORN:
            top_layer = nn.Linear(1024, num_classes - 1)
            self.criterion = corn_loss
            
        else:
            top_layer = nn.Linear(1024, num_classes)
            self.criterion = nn.CrossEntropyLoss

        # Combine the shared layers with the top layer
        self.classifier = nn.Sequential(
            shared_classifier_layers,
            top_layer
        )

        # Save the modified model as a member variable
        self.features = model.features
        self.avgpool = model.avgpool

    @ staticmethod
    def get_class_probabilities(x):
        return nn.functional.softmax(x, dim=1)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x) #TODO: decide whether to keep it in or not
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_optimizer_layers(self):
        return self.features, self.classifier
        
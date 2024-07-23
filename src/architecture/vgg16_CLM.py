import sys

sys.path.append(".")

import torch
import torch.nn as nn
from torchvision import models
from multi_label.CLM import CLM
from multi_label import QWK


class CustomVGG16_CLM(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16_CLM, self).__init__()

        # Load the pre-trained VGG16 model
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.features = torch.nn.Sequential(*(list(model.children())[:-1]))
        #self.features = self.features[:-1]
        # Freeze training for all layers in features
        # for param in model.features.parameters():
        #     param.requires_grad = False

        # Modify the classifier layer
        num_features = model.classifier[0].in_features
        #features = list(model.classifier.children())[:-1]  # select features in our last layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(),
            #nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(4096),
            #nn.Dropout(0.5),
            
            nn.Linear(4096, 1),
            nn.BatchNorm1d(1),
            CLM(classes=num_classes, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False),
        )
         # add layer with output size num_classes
        #model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Save the modified model as a member variable
        self.avgpool = model.avgpool
        self.criterion = nn.NLLLoss
        
        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)


    @staticmethod
    def get_class_probabilies(x):
        return nn.functional.softmax(x, dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_optimizer_layers(self):
        return self.features, self.classifier

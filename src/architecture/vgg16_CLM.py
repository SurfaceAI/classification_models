import sys

sys.path.append(".")

import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from multi_label import QWK
import math

class CustomVGG16_CLM(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16_CLM, self).__init__()

        # Load the pre-trained VGG16 model
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.features = torch.nn.Sequential(*(list(model.children())[:-1]))
        # Freeze training for all layers in features
        # for param in model.features.parameters():
        #     param.requires_grad = False

        # Modify the classifier layer
        num_features = model.classifier[0].in_features
        #features = list(model.classifier.children())[:-1]  # select features in our last layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            #nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            #nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 1),
            nn.BatchNorm1d(1, momentum=0.99, eps=0.001),
            CLM(classes=num_classes, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False),
        )
         # add layer with output size num_classes
        #model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Save the modified model as a member variable
        #self.avgpool = model.avgpool
        self.criterion = nn.CrossEntropyLoss
        
        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)


    @staticmethod
    def get_class_probabilies(x):
        return nn.functional.softmax(x, dim=1)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_optimizer_layers(self):
        return self.classifier


class CLM(nn.Module):
    def __init__(self, classes, link_function, min_distance=0.35, use_slope=False, fixed_thresholds=False):
        super(CLM, self).__init__()
        self.classes = classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.use_slope = use_slope
        self.fixed_thresholds = fixed_thresholds
        
        #if we dont have fixed thresholds, we initalize two trainable parameters 

        if not self.fixed_thresholds:
            if self.classes > 2:
                    # First threshold
                    self.thresholds_b = nn.Parameter(torch.rand(1) * 0.1)  # Random number between 0 and 0.1
                    # random_value = torch.rand(1) * -0.3
                    # random_value = random_value - 2
                    # self.thresholds_b = nn.Parameter(random_value)
                    # Squared distance
                    minval = math.sqrt((1.0 / (self.classes - 2)) / 2)
                    maxval = math.sqrt(1.0 / (self.classes - 2))
                    # minval = 1.9
                    # maxval = 2.3
                    self.thresholds_a = nn.Parameter(minval + (maxval - minval) * torch.rand(self.classes - 2))

            else: 
                raise ValueError("Number of classes must be greater than 2 for CLM.")
            #first threshold
            #self.thresholds_b = nn.Parameter(torch.rand(1) * 0.1) #random number between 0 and 1
            # self.thresholds_b = nn.Parameter(torch.rand(1) * 0.1)
            # #squared distance 
            # self.thresholds_a = nn.Parameter(
            #     torch.sqrt(torch.ones(num_classes - 2) / (num_classes - 2) / 2) * torch.rand(num_classes - 2)
            # )

        if self.use_slope:
            self.slope = nn.Parameter(2)
            
    def convert_thresholds(self, b, a, min_distance=0.35):
        a = a.pow(2) + min_distance
        thresholds_param = torch.cat([b, a], dim=0).float()
        th = torch.cumsum(thresholds_param, dim=0)
        return th
    
    def nnpom(self, projected, thresholds):
        projected = projected.view(-1).float()
        
        if self.use_slope:
            projected = projected * self.slope
            thresholds = thresholds * self.slope

        m = projected.shape[0]
        a = thresholds.repeat(m, 1)
        b = projected.repeat(self.classes - 1, 1).t()
        z3 = a - b

        if self.link_function == 'probit':
            a3T = torch.distributions.Normal(0, 1).cdf(z3)
        elif self.link_function == 'cloglog':
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:  # logit
            a3T = torch.sigmoid(z3)

        ones = torch.ones((m, 1))
        a3 = torch.cat([a3T, ones], dim=1)
        A3 = torch.cat([a3[:, :1], a3[:, 1:] - a3[:, :-1]], dim=1)

        return A3

    def forward(self, x):
        if self.fixed_thresholds:
            thresholds = torch.linspace(0, 1, 5, dtype=torch.float32)[1:-1]
        else:
            thresholds = self.convert_thresholds(self.thresholds_b, self.thresholds_a, self.min_distance)

        return self.nnpom(x, thresholds)

    def extra_repr(self):
        return 'num_classes={}, link_function={}, min_distance={}, use_slope={}, fixed_thresholds={}'.format(
            self.classes, self.link_function, self.min_distance, self.use_slope, self.fixed_thresholds
        )

architecture = "VGG16"
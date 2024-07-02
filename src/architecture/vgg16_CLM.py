import sys

sys.path.append(".")

import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from multi_label import QWK

class CLM(nn.Module):
    def __init__(self, num_classes, link_function, min_distance=0.35, use_slope=False, fixed_thresholds=False):
        super(CLM, self).__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.use_slope = use_slope
        self.fixed_thresholds = fixed_thresholds
        
        #if we dont have fixed thresholds, we initalize two trainable parameters 
        if not self.fixed_thresholds:
            #first threshold
            self.thresholds_b = nn.Parameter(torch.rand(1) * 0.1) #random number between 0 and 1
            #squared distance 
            self.thresholds_a = nn.Parameter(
                torch.sqrt(torch.ones(num_classes - 2) / (num_classes - 2) / 2) * torch.rand(num_classes - 2)
            )

        if self.use_slope:
            self.slope = nn.Parameter(torch.tensor(100.0))
            
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
        b = projected.repeat(self.num_classes - 1, 1).t()
        z3 = a - b

        if self.link_function == 'probit':
            a3T = torch.distributions.Normal(0, 1).cdf(z3)
        elif self.link_function == 'cloglog':
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:  # logit
            a3T = torch.sigmoid(z3)

        ones = torch.ones((m, 1))
        a3 = torch.cat([a3T, ones], dim=1)
        a3 = torch.cat([a3[:, :1], a3[:, 1:] - a3[:, :-1]], dim=1)

        return a3

    def forward(self, x):
        if self.fixed_thresholds:
            thresholds = torch.linspace(0, 1, self.num_classes, dtype=torch.float32)[1:-1]
        else:
            thresholds = self.convert_thresholds(self.thresholds_b, self.thresholds_a, self.min_distance)

        return self.nnpom(x, thresholds)

    def extra_repr(self):
        return 'num_classes={}, link_function={}, min_distance={}, use_slope={}, fixed_thresholds={}'.format(
            self.num_classes, self.link_function, self.min_distance, self.use_slope, self.fixed_thresholds
        )

architecture = "VGG16"

class CustomVGG16_CLM(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16_CLM, self).__init__()

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
        #self.criterion = nn.CrossEntropyLoss
        if num_classes == 1:
            self.criterion = QWK.qwk_loss_base
            #self.criterion = nn.MSELoss
        else:
            #cost_matrix = QWK.make_cost_matrix(num_classes)
            #self.criterion = QWK.qwk_loss(cost_matrix, num_classes)
            self.criterion = QWK.qwk_loss_base
        # else:
        #     self.criterion = nn.CrossEntropyLoss
            
        
        self.CLM = CLM(num_classes = 4, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False)
        

    @ staticmethod
    def get_class_probabilies(x):
        return nn.functional.softmax(x, dim=1)

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        
        x = self.CLM(x)

        return x
    
    def get_optimizer_layers(self):
        return self.classifier, self.CLM
    
        
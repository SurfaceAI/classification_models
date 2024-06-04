import torch
import torch.nn as nn
from torchvision import models
from src.utils.helper import NonNegUnitNorm


class Condition_CNN_PRE(nn.Module):
    def __init__(self, num_c, num_classes):
        super(Condition_CNN_PRE, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
            
        
        ### Block 1
        self.block1 = model.features[:5]
        self.block2 = model.features[5:10]
        self.block3 = model.features[10:17]
        self.block4 = model.features[17:24]
        self.block5 = model.features[24:]
        
        #Coarse prediction branch
        self.coarse_classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_c)
        )
        
        ### Fine prediction branch
        num_features = model.classifier[6].in_features
        model.classifier[0] = nn.Linear(in_features=32768, out_features=4096, bias=True)
        features = list(model.classifier.children())[:-1]  # select features in our last layer
        features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        model.classifier = nn.Sequential(*features)  # Replace the model classifier
        
        ### Condition part
        self.coarse_condition = nn.Linear(num_c, num_classes, bias=False)
        self.coarse_condition.weight.data.fill_(0)  # Initialize weights to zero
        self.constraint = NonNegUnitNorm(axis=0) 
        
        # Save the modified model as a member variable
        self.features = model.features
        self.fine_classifier = model.classifier
        
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if num_classes == 1:
            self.fine_criterion = nn.MSELoss
        else:
            self.fine_criterion = nn.CrossEntropyLoss
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse = inputs
        
        x = self.block1(images) #[128, 64, 128, 128]
        x = self.block2(x) #([128, 128, 64, 64])
        x = self.block3(x) #e([128, 256, 32, 32])
        
        x = self.block4(x) # [128, 512, 16, 16])
        x = self.block5(x) # [128, 512, 8, 8])
        
       # x = self.avgpool(x)
        flat = x.reshape(x.size(0), -1) #([128, 32768])
        coarse_output = self.coarse_classifier(flat)
        fine_raw_output = self.fine_classifier(flat) #[128, 18])
        
        if self.training:
            coarse_condition = self.coarse_condition(true_coarse) 
        else:
            coarse_condition = self.coarse_condition(coarse_output) 
            
    
        #Adding the conditional probabilities to the dense features
        fine_output = coarse_condition + fine_raw_output
        self.coarse_condition.weight.data = self.constraint(self.coarse_condition.weight.data)
        
        return coarse_output, fine_output
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

architecture = "VGG16"

class B_CNN_VGG16(nn.Module):
    def __init__(self, num_c, num_classes):
        super(B_CNN_VGG16, self).__init__()

        # Load the pre-trained VGG16 model
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Freeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = False
            
            
        #Divide the model in blocks like in B_CNN paper
        # self.block_1 = model.features[:5]
        # self.block_2 = model.features[5:10]
        # self.block_3 = model.features[10:17]
        # self.block_4 = model.features[17:24]
        # self.block_5 = model.features[24:]
        
        #Build branch 1
        self.b1_classifier = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_c)
        )
        

        # Modify the classifier layer
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # select features in our last layer
        features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        model.classifier = nn.Sequential(*features)  # Replace the model classifier
        
        model.classifier[0].in_features = ((model.block_5[-3].out_channels) * 16 * 16)

        # Save the modified model as a member variable
        self.features = model.features
        #self.avgpool = model.avgpool
        self.classifier = model.classifier
        if num_classes == 1:
            self.criterion = nn.MSELoss
        else:
            self.criterion = nn.CrossEntropyLoss

    @ staticmethod
    def get_class_probabilies(x):
        return nn.functional.softmax(x, dim=1)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        
        flat = x.reshape(x.size(0), -1) 
        coarse_output = self.b1_classifier(flat)

        x = self.block_4(x)
        x = self.block_5(x)

        flat = x.reshape(x.size(0), -1) 
        fine_output = self.classifier(x)
        
        return coarse_output, fine_output
    
    def get_optimizer_layers(self):
        return self.classifier, self.b1_classifier
        


class VGG16_B_CNN(nn.Module):
    def __init__(self, num_c, num_classes):
        super(VGG16_B_CNN, self).__init__()
        
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Freeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = False
            
        
        ### Block 1
        self.block1 = model.features[:5]
        self.block2 = model.features[5:10]
        self.block3 = model.features[10:17]
        self.block4 = model.features[17:24]
        self.block5 = model.features[24:]
        
        #Coarse prediction branch
        self.b1_classifier = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_c)
        )
        
        ### Fine prediction branch
        num_features = model.classifier[6].in_features
        model.classifier[0] = nn.Linear(in_features=32768, out_features=4096, bias=True)
        features = list(model.classifier.children())[:-1]  # select features in our last layer
        #features[0].in_features = ((self.block5[-3].out_channels) * 16 * 16)
        features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        model.classifier = nn.Sequential(*features)  # Replace the model classifier
        #model.classifier[0].in_features = ((self.block5[-3].out_channels) * 16 * 16) #do we need this?
        
        # Save the modified model as a member variable
        self.features = model.features
        #self.avgpool = model.avgpool
        self.classifier = model.classifier
        if num_classes == 1:
            self.criterion = nn.MSELoss
        else:
            self.criterion = nn.CrossEntropyLoss
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, x):
        x = self.block1(x) #[128, 64, 128, 128]
        x = self.block2(x) #([128, 128, 64, 64])
        x = self.block3(x) #e([128, 256, 32, 32])
        
        flat = x.reshape(x.size(0), -1) #[128, 262144])
        coarse_output = self.b1_classifier(flat)
        
        x = self.block4(x) # [128, 512, 16, 16])
        x = self.block5(x) 
        
        #x = self.avgpool(x)
        flat = x.reshape(x.size(0), -1) #([128, 131072])
        fine_output = self.classifier(flat) #[48, 18])
        
        return coarse_output, fine_output
import torch
import torch.nn as nn
from torchvision import models

class CustomMultLayer(nn.Module):
    def __init__(self):
        super(CustomMultLayer, self).__init__()
        
    def forward(self, tensor_1, tensor_2):
        return torch.mul(tensor_1, tensor_2)

class HierarchyNet_Pre(nn.Module):
    def __init__(self, num_c, num_classes):
        super(HierarchyNet_Pre, self).__init__()
        
        self.custom_layer = CustomMultLayer()
        
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
        
        #Here comes the flatten layer and then the dense ones for the coarse classes
        # self.coarse_classifier = nn.Sequential(
        #     nn.Linear(256 * 8 * 8, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, num_c)
        # )
        
        
        #model.classifier[0] = nn.Linear(in_features=32768, out_features=4096, bias=True)
        #features = list(model.classifier.children())[:-1]  # select features in our last layer
        #features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        #model.classifier = nn.Sequential(*features)  # Replace the model classifier


        num_features = model.classifier[6].in_features
        # Modify the first fully connected layer to accept the correct input size
        model.classifier[0] = nn.Linear(in_features=512*8*8, out_features=4096, bias=True)

        # Save the modified classifier layers as member variables
        self.fc = nn.Sequential(*list(model.classifier.children())[0:3])
        self.fc_1 = nn.Sequential(*list(model.classifier.children())[3:6])
        
        self.fc_2_coarse = nn.Linear(num_features, num_c)
        self.fc_2_fine = nn.Linear(num_features, num_classes) 

        
        self.coarse_criterion = nn.CrossEntropyLoss()
        
        if num_classes == 1:
            self.fine_criterion = nn.MSELoss()
        else:
            self.fine_criterion = nn.CrossEntropyLoss()
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
     
    def crop(self, x, dimension, start, end):
        slices = [slice(None)] * x.dim()
        slices[dimension] = slice(start, end)
        return x[tuple(slices)]

    
    def forward(self, x):
        x = self.block1(x)   #128, 64, 128, 128    
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x) #128, 512, 8, 8]
        
        flat = x.reshape(x.size(0), -1) #torch.Size([16, 131072])
        
        branch_output = self.fc(flat)
        branch_output = self.fc_1(branch_output)
        
        coarse_output = self.fc_2_coarse(branch_output)
        
        #cropping coarse outputs
        coarse_1 = self.crop(coarse_output, 1, 0, 1)
        coarse_2 = self.crop(coarse_output, 1, 1, 2)
        coarse_3 = self.crop(coarse_output, 1, 2, 3)
        coarse_4 = self.crop(coarse_output, 1, 3, 4)
        coarse_5 = self.crop(coarse_output, 1, 4, 5)
        
        #coarse_pred = F.softmax(coarse_output, dim=1)
        
        raw_fine_output = self.fc_2_fine(branch_output)
        
        fine_1 = self.crop(raw_fine_output, 1, 0, 4)
        fine_2 = self.crop(raw_fine_output, 1, 4, 8)
        fine_3 = self.crop(raw_fine_output, 1, 8, 12)
        fine_4 = self.crop(raw_fine_output, 1, 12, 15)
        fine_5 = self.crop(raw_fine_output, 1, 15, 18)
        
        fine_1 = self.custom_layer(coarse_1, fine_1)
        fine_2 = self.custom_layer(coarse_2, fine_2)
        fine_3 = self.custom_layer(coarse_3, fine_3)
        fine_4 = self.custom_layer(coarse_4, fine_4)
        fine_5 = self.custom_layer(coarse_5, fine_5)
        
        fine_output = torch.cat([fine_1, fine_2, fine_3, fine_4, fine_5], dim=1)
        
        return coarse_output, fine_output
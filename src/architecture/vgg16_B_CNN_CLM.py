import sys

sys.path.append(".")

import torch
import torch.nn as nn
from torchvision import models
from multi_label.CLM import CLM
from multi_label.QWK import QWK

class B_CNN_CLM(nn.Module):
    def __init__(self, num_c, num_classes):
        super(B_CNN_CLM, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        

        ### Block 1
        self.block1_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.block1_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        ### Block 2
        self.block2_layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.block2_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        

        ### Block 3
        self.block3_layer1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.block3_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.block3_layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        ### Coarse branch
        #self.c_flat = nn.Flatten() 
        
        self.surface_fc = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_c)
        )
        
        
        ### Block 4
        self.block4_layer1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.block4_layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.block4_layer3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        ### Block 5
        self.block5_layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.block5_layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.block5_layer3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.quality_fc_asphalt = self._create_quality_fc(num_classes=4)
        self.quality_fc_concrete = self._create_quality_fc(num_classes=4)
        self.quality_fc_sett = self._create_quality_fc(num_classes=4)
        self.quality_fc_paving_stones = self._create_quality_fc(num_classes=3)
        self.quality_fc_unpaved = self._create_quality_fc(num_classes=3)
        
        
        if num_classes == 1:
            self.fine_criterion = nn.MSELoss
        else:
            self.fine_criterion = nn.NLLLoss

    def _create_quality_fc(self, num_classes=4):
        return nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.BatchNorm1d(1),
            CLM(classes=num_classes)
        )
    
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse, hierarchy_method = inputs
        
        x = self.block1_layer1(images) #[batch_size, 64, 256, 256]
        x = self.block1_layer2(x) #[batch_size, 64, 128, 128]
        
        x = self.block2_layer1(x)#[batch_size, 64, 128, 128] 
        x = self.block2_layer2(x) #(batch_size, 128, 64, 64)
        
        x = self.block3_layer1(x)
        x = self.block3_layer2(x)
        x = self.block3_layer3(x)
        
        flat = x.reshape(x.size(0), -1) 
        coarse_output = self.surface_fc(flat)
        coarse_probs = self.get_class_probabilies(coarse_output)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        
        
        x = self.block4_layer1(x)
        x = self.block4_layer2(x) # output: [batch_size, 512 #channels, 16, 16 #height&width]
        x = self.block4_layer3(x)
        
        x = self.block5_layer1(x)
        x = self.block5_layer2(x)
        x = self.block5_layer3(x)
        
        flat = x.reshape(x.size(0), -1) #([48, 131072])
        
        #in this part the ground truth is used to guide to the correct regression quality head
        
        fine_output_asphalt = self.quality_fc_asphalt(flat)
    
        fine_output_concrete = self.quality_fc_concrete(flat)
        
        fine_output_sett = self.quality_fc_sett(flat)
        
        fine_output_paving_stones = self.quality_fc_paving_stones(flat)
        
        fine_output_unpaved = self.quality_fc_unpaved(flat)
        
        all_fine_clm_outputs = torch.cat([fine_output_asphalt, fine_output_concrete, fine_output_sett, fine_output_paving_stones, fine_output_unpaved], dim=1)
        
        return coarse_output, all_fine_clm_outputs

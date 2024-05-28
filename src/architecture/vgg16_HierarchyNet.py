import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        
    def forward(self, tensor_1, tensor_2):
        return torch.mul(tensor_1, tensor_2)

class HierarchyNet(nn.Module):
    def __init__(self, num_c, num_classes):
        super(HierarchyNet, self).__init__()
        
        self.custom_layer = CustomLayer()
        
        
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
        
        #Here comes the flatten layer and then the dense ones for the coarse classes
        self.c_fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c_fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5))
        
        self.coarse_branch = (
            nn.Linear(128, num_c) #output layer for coarse prediction
        )    
        
        self.fine_branch = (
            nn.Linear(128, num_classes) #output layer for coarse prediction
        )          
        
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
     
    def crop(x, dimension, start, end):
        slices = [slice(None)] * x.dim()
        slices[dimension] = slice(start, end)
        return x[tuple(slices)]

     
    
    def forward(self, x):
        x = self.block1_layer1(x)
        x = self.block1_layer2(x)
        
        x = self.block2_layer1(x)
        x = self.block2_layer2(x) 
        
        x = self.block3_layer1(x)
        x = self.block3_layer2(x)
        x = self.block3_layer3(x)
        
        x = self.block4_layer1(x)
        x = self.block4_layer2(x) 
        x = self.block4_layer3(x)
        
        x = self.block5_layer1(x)
        x = self.block5_layer2(x) 
        x = self.block5_layer3(x)
        
        flat = x.reshape(x.size(0), -1) #torch.Size([16, 131072])
        
        branch_output = self.c_fc(flat)
        branch_output = self.c_fc1(branch_output)
        
        coarse_output = self.coarse_branch(branch_output)
        
        #cropping coarse outputs
        coarse_1 = self.crop(coarse_output, 1, 0, 1)
        coarse_2 = self.crop(coarse_output, 1, 1, 2)
        coarse_3 = self.crop(coarse_output, 1, 2, 3)
        coarse_4 = self.crop(coarse_output, 1, 3, 4)
        coarse_5 = self.crop(coarse_output, 1, 4, 5)
        
        #coarse_pred = F.softmax(coarse_output, dim=1)
        
        raw_fine_output = self.fine_branch(branch_output)
        
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
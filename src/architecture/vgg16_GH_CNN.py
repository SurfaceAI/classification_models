import torch
import torch.nn as nn

class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        
    def forward(self, parent_prob, subclass_probs):
        #iterate through all subclasses of one parent class
        y_subclass = []
        for i in range(0, len(subclass_probs)):
            y_i_sublcass = torch.mul(parent_prob, subclass_probs[i]) / sum(subclass_probs)
            y_subclass.append(y_i_sublcass)
            return y_subclass
    
    
class GH_CNN(nn.Module):
    def __init__(self, num_c, num_classes):
        super(GH_CNN, self).__init__()
        
        
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
        
        z_1 = self.coarse_branch(branch_output)
        
        z_2 = self.fine_branch(branch_output)
        
        #cropping coarse outputs: z_i_j, i=1: coarse branch
        z_1_1 = self.crop(z_1, 1, 0, 1) #raw prob asphalt
        z_1_2 = self.crop(z_1, 1, 1, 2)
        z_1_3 = self.crop(z_1, 1, 2, 3)
        z_1_4 = self.crop(z_1, 1, 3, 4)
        z_1_5 = self.crop(z_1, 1, 4, 5)
        
        #cropping fine output: z_i_j, i=2: fine branch
        z_2_1 = self.crop(z_2, 1, 0, 4) #raw prob all asphalt subclasses (asphalt_excellent, asphalt_good, asphalt_intermediate, asphalt_bad)
        z_2_2 = self.crop(z_2, 1, 4, 8)
        z_2_3 = self.crop(z_2, 1, 8, 12)
        z_2_4 = self.crop(z_2, 1, 12, 15)
        z_2_5 = self.crop(z_2, 1, 15, 18)
        
        #FAFO
        y_2_1 = self.custom_layer(z_1_1, z_2_1)
        y_2_2 = self.custom_layer(z_1_1, z_2_2)
        y_2_3 = self.custom_layer(z_1_1, z_2_3)
        y_2_4 = self.custom_layer(z_1_1, z_2_4)
        y_2_5 = self.custom_layer(z_1_1, z_2_5)

        coarse_output = z_1
        fine_output = torch.cat([y_2_1, y_2_2, y_2_3, y_2_4, y_2_5], dim=1)
        
        return coarse_output, fine_output

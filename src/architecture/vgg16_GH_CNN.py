import torch
import torch.nn as nn
from experiments.config import train_config
config = train_config.GH_CNN


class CustomBayesLayer(nn.Module):
    def __init__(self):
        super(CustomBayesLayer, self).__init__()
        
    def forward(self, parent_prob, subclass_probs):
        y_subclass = torch.mul(parent_prob, subclass_probs) / torch.sum(subclass_probs, dim=1, keepdim=True)
        return y_subclass
        
        #iterate through all subclasses of one parent class
        # y_subclass = []
        # for i in range(subclass_probs.shape[1]):
        #     y_i_subclass = torch.mul(parent_prob, subclass_probs[:, i]) / torch.sum(subclass_probs, dim=1, keepdim=True)
        #     y_subclass.append(y_i_subclass.unsqueeze(1))  # Keep the dimension for concatenation
        # return torch.cat(y_subclass, dim=1)
    
# class CustomMultLayer(nn.Module):
#     def __init__(self):
#         super(CustomMultLayer, self).__init__()
        
#     def forward(self, tensor_1, tensor_2):
#         return torch.mul(tensor_1, tensor_2)   
    
class GH_CNN(nn.Module):
    def __init__(self, num_c, num_classes):
        super(GH_CNN, self).__init__()
        
        #Custom layer
        self.custom_bayes_layer = CustomBayesLayer()
        #self.custom_mult_layer = CustomMultLayer()
        
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
            nn.Linear(512 * 8 * 8, 256),
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
     
    def crop(self, x, dimension, start, end):
        slices = [slice(None)] * x.dim()
        slices[dimension] = slice(start, end)
        return x[tuple(slices)]

    
    def forward(self, inputs):

        x = self.block1_layer1(inputs)
        x = self.block1_layer2(x)
        
        x = self.block2_layer1(x)
        x = self.block2_layer2(x) #([16, 128, 64, 64])
        
        x = self.block3_layer1(x)
        x = self.block3_layer2(x)
        x = self.block3_layer3(x) #([16, 256, 32, 32])
        
        x = self.block4_layer1(x)
        x = self.block4_layer2(x) 
        x = self.block4_layer3(x) #([16, 512, 16, 16])
        
        x = self.block5_layer1(x)
        x = self.block5_layer2(x) 
        x = self.block5_layer3(x) #([16, 512, 8, 8])
        
        flat = x.reshape(x.size(0), -1) #torch.Size([16, 32.768])
        
        branch_output = self.c_fc(flat)
        branch_output = self.c_fc1(branch_output)
        
        z_1 = self.coarse_branch(branch_output) #[16,5]
        
        z_2 = self.fine_branch(branch_output) #[16,18]
        
        return z_1, z_2
        
    def teacher_forcing(self, z_1, z_2, true_coarse):
        
        true_coarse_1 = self.crop(true_coarse, 1, 0, 1)
        true_coarse_2 = self.crop(true_coarse, 1, 1, 2)
        true_coarse_3 = self.crop(true_coarse, 1, 2, 3)
        true_coarse_4 = self.crop(true_coarse, 1, 3, 4)
        true_coarse_5 = self.crop(true_coarse, 1, 4, 5)
        
        raw_fine_1 = self.crop(z_2, 1, 0, 4) #raw prob all asphalt subclasses (asphalt_excellent, asphalt_good, asphalt_intermediate, asphalt_bad)
        raw_fine_2 = self.crop(z_2, 1, 4, 8)
        raw_fine_3 = self.crop(z_2, 1, 8, 12)
        raw_fine_4 = self.crop(z_2, 1, 12, 15)
        raw_fine_5 = self.crop(z_2, 1, 15, 18)
        
        fine_1 = torch.mul(true_coarse_1, raw_fine_1)
        fine_2 = torch.mul(true_coarse_2, raw_fine_2)
        fine_3 = torch.mul(true_coarse_3, raw_fine_3)
        fine_4 = torch.mul(true_coarse_4, raw_fine_4)
        fine_5 = torch.mul(true_coarse_5, raw_fine_5)
        
        fine_output = torch.cat([fine_1, fine_2, fine_3, fine_4, fine_5], dim=1)
        
        return z_1, fine_output
        
    def bayesian_adjustment(self, z_1, z_2):
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
        y_2_1 = self.custom_bayes_layer(z_1_1, z_2_1)
        y_2_2 = self.custom_bayes_layer(z_1_2, z_2_2)
        y_2_3 = self.custom_bayes_layer(z_1_3, z_2_3)
        y_2_4 = self.custom_bayes_layer(z_1_4, z_2_4)
        y_2_5 = self.custom_bayes_layer(z_1_5, z_2_5)

        coarse_output = z_1
        fine_output = torch.cat([y_2_1, y_2_2, y_2_3, y_2_4, y_2_5], dim=1)
        
        return coarse_output, fine_output 

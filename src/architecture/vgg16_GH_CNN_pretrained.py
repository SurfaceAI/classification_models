import torch
import torch.nn as nn
from experiments.config import train_config#
from torchvision import models
config = train_config.GH_CNN_PRE


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
    
class GH_CNN_PRE(nn.Module):
    def __init__(self, num_c, num_classes):
        super(GH_CNN_PRE, self).__init__()
        
        #Custom layer
        self.custom_bayes_layer = CustomBayesLayer()
        #self.custom_mult_layer = CustomMultLayer()
        
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
            
              
        ### Block 1
        self.features = model.features
        # self.block1 = model.features[:5]
        # self.block2 = model.features[5:10]
        # self.block3 = model.features[10:17]
        # self.block4 = model.features[17:24]
        # self.block5 = model.features[24:]
        
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

    
    def forward(self, inputs):

        x = self.features(inputs) #128, 512, 8, 8]
        
        flat = x.reshape(x.size(0), -1)#torch.Size([16, 32.768])
        
        branch_output = self.fc(flat)
        branch_output = self.fc_1(branch_output)
        
        z_1 = self.fc_2_coarse(branch_output) #[16,5]
        
        z_2 = self.fc_2_fine(branch_output) #[16,18]
        
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
    
    def get_optimizer_layers(self):
        return self.features, self.coarse_classifier, self.fc_1, self.fc_2_coarse, self.fc_2_fine
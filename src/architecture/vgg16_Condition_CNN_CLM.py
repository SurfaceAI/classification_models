import torch
import torch.nn as nn
from src.utils.helper import NonNegUnitNorm
from multi_label.CLM import CLM

class Condition_CNN_CLM(nn.Module):
    def __init__(self, num_c, num_classes):
        super(Condition_CNN_CLM, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        
        # self.x = torch.randn(3, 256, 256)
        
        # self.true_coarse = torch.tensor([self.num_c]) #empty vector with dimensions of the true_label vector
                
        
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
            nn.ReLU())

#--------------------------coarse--------------------------------------       
         
        ### Coarse branch prediction
        self.c_fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c_fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c_fc2 = (
            nn.Linear(256, num_c)
        )
        
#--------------------------fine--------------------------------------       
        
        self.CLM_4 = CLM(num_classes = 4, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False)
        self.CLM_3 = CLM(num_classes = 3, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False)

        
        self.quality_fc_asphalt = self.create_quality_fc()
        self.quality_fc_concrete = self.create_quality_fc()
        self.quality_fc_sett = self.create_quality_fc()
        self.quality_fc_paving_stones = self.create_quality_fc()
        self.quality_fc_unpaved = self.create_quality_fc()
        

        
#-------------------condition fine pred on coarse labels -------------------       
        
        self.coarse_condition = nn.Linear(num_c, 18, bias=False)
        self.coarse_condition.weight.data.fill_(0)  # Initialize weights to zero
        self.constraint = NonNegUnitNorm(axis=0)  # Define the constraint
        #self.coarse_condition.weight.requires_grad = True  Gradient=False -> no gradients for this layer
        
        # self.fine_raw_layer =  nn.Sequential(
        #     nn.Linear(1024, self.num_classes),
        #     nn.ReLU()
        # )
        
    def create_quality_fc(self):
        return nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
  
        
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse = inputs
        
        x = self.block1_layer1(images) #[batch_size, 64, 256, 256]
        x = self.block1_layer2(x) #[batch_size, 64, 128, 128]
        
        x = self.block2_layer1(x)#[batch_size, 64, 128, 128] 
        x = self.block2_layer2(x) #(batch_size, 128, 64, 64)
        
        x = self.block3_layer1(x)
        x = self.block3_layer2(x)
        x = self.block3_layer3(x)
        
        x = self.block4_layer1(x)
        x = self.block4_layer2(x)
        x = self.block4_layer3(x)
        
        x = self.block5_layer1(x)
        x = self.block5_layer2(x)
        x = self.block5_layer3(x) #[batch_size, 512, 16, 16])
        
        #--- flatten layer
        
        flat = x.view(x.size(0), -1)
        
        #--- coarse branch
         
        coarse_output = self.c_fc(flat) 
        coarse_output = self.c_fc1(coarse_output)
        coarse_output = self.c_fc2(coarse_output)
        
        #---fine
        fine_raw_asphalt = self.quality_fc_asphalt(flat) #([batch_size, 1024])
        clm_fine_output_asphalt = self.CLM_4(fine_raw_asphalt)
        
        fine_raw_concrete = self.quality_fc_concrete(flat)
        clm_fine_output_concrete = self.CLM_4(fine_raw_concrete)
        
        fine_raw_sett = self.quality_fc_sett(flat)
        clm_fine_output_sett = self.CLM_4(fine_raw_sett)
        
        fine_raw_paving_stones = self.quality_fc_paving_stones(flat)
        clm_fine_output_paving_stones = self.CLM_3(fine_raw_paving_stones)
        
        fine_raw_unpaved = self.quality_fc_unpaved(flat)
        clm_fine_output_unpaved = self.CLM_3(fine_raw_unpaved)

        
        if self.training:
            coarse_condition = self.coarse_condition(true_coarse) 
        else:
            coarse_condition = self.coarse_condition(coarse_output) 
            
        fine_clm_combined = torch.cat([clm_fine_output_asphalt, 
                                       clm_fine_output_concrete, 
                                       clm_fine_output_sett, 
                                       clm_fine_output_paving_stones, 
                                       clm_fine_output_unpaved], 
                                      dim=1)
        #features = torch.add(coarse_condition, fine_raw)#
        #Adding the conditional probabilities to the dense features
        fine_output = coarse_condition + fine_clm_combined
        self.coarse_condition.weight.data = self.constraint(self.coarse_condition.weight.data)
        
        return coarse_output, fine_output

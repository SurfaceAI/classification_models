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
        
        self.quality_fc_asphalt = self._create_quality_fc(num_classes=4)
        self.quality_fc_concrete = self._create_quality_fc(num_classes=4)
        self.quality_fc_paving_stones = self._create_quality_fc(num_classes=4)
        self.quality_fc_sett = self._create_quality_fc(num_classes=3)
        self.quality_fc_unpaved = self._create_quality_fc(num_classes=3)
        

        
#-------------------condition fine pred on coarse labels -------------------       
        
        self.coarse_condition = nn.Linear(num_c, 18, bias=False)
        self.coarse_condition.weight.data.fill_(0)  # Initialize weights to zero
        self.constraint = NonNegUnitNorm(axis=0)  # Define the constraint
        #self.coarse_condition.weight.requires_grad = True  Gradient=False -> no gradients for this layer
        
        # self.fine_raw_layer =  nn.Sequential(
        #     nn.Linear(1024, self.num_classes),
        #     nn.ReLU()
        # )
        
    def _create_quality_fc(self, num_classes=4):
        return nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.BatchNorm1d(1),
            CLM(classes=num_classes, link_function="logit", min_distance=0.0, use_slope=False, fixed_thresholds=False)
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
        
        if hierarchy_method == 'use_condition_layer':
        #---fine
            fine_output_asphalt = self.quality_fc_asphalt(flat) #([batch_size, 1024])
            
            fine_output_concrete = self.quality_fc_concrete(flat)
            
            fine_output_paving_stones = self.quality_fc_paving_stones(flat)
                        
            fine_output_sett = self.quality_fc_sett(flat)
            
            fine_output_unpaved = self.quality_fc_unpaved(flat)

            
            if self.training:
                coarse_condition = self.coarse_condition(true_coarse) 
        
                
            else:
                coarse_condition = self.coarse_condition(coarse_output) 
                
            fine_clm_combined = torch.cat([fine_output_asphalt, 
                                        fine_output_concrete, 
                                        fine_output_sett, 
                                        fine_output_paving_stones, 
                                        fine_output_unpaved], 
                                        dim=1)
        #features = torch.add(coarse_condition, fine_raw)#
        #Adding the conditional probabilities to the dense features
            fine_output = coarse_condition + fine_clm_combined
            self.coarse_condition.weight.data = self.constraint(self.coarse_condition.weight.data)
            
        elif hierarchy_method == 'use_ground_truth': 
            
            if self.training:
                #fine_output = torch.zeros(x.size(0), 1, device=x.device)
                fine_predictions = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
                
                #we seperate out one-hot encoded tensor to seperate one hot tensors 1=belonging to surface type, 0 not 

                fine_output_asphalt = self.quality_fc_asphalt(flat[true_coarse[:, 0].bool()])
                #fine_pred_asphalt = torch.argmax(self.CLM_4(fine_output_asphalt), dim=1)
                #fine_predictions[true_coarse[:, 0].bool()] = fine_pred_asphalt
                
                fine_output_concrete = self.quality_fc_concrete(flat[true_coarse[:, 1].bool()])
                #fine_pred_concrete = torch.argmax(self.CLM_4(fine_output_concrete), dim=1)
                #fine_predictions[true_coarse[:, 1].bool()] = fine_pred_concrete   
                          
                fine_output_paving_stones = self.quality_fc_paving_stones(flat[true_coarse[:, 3].bool()])
                #fine_pred_paving_stones = torch.argmax(self.CLM_3(fine_output_paving_stones), dim=1)
                #fine_predictions[true_coarse[:, 3].bool()] = fine_pred_paving_stones   

                fine_output_sett = self.quality_fc_sett(flat[true_coarse[:, 2].bool()])
                #fine_pred_sett = torch.argmax(self.CLM_4(fine_output_sett), dim=1)
                #fine_predictions[true_coarse[:, 2].bool()] = fine_pred_sett                       


                fine_output_unpaved = self.quality_fc_unpaved(flat[true_coarse[:, 4].bool()])
                #fine_pred_unpaved = torch.argmax(self.CLM_3(fine_output_unpaved), dim=1)
                #fine_predictions[true_coarse[:, 3].bool()] = fine_pred_unpaved   
                
            else:
                fine_output_asphalt = self.quality_fc_asphalt(flat)
                
                fine_output_concrete = self.quality_fc_concrete(flat)
                
                fine_output_paving_stones = self.quality_fc_paving_stones(flat)   
                             
                fine_output_sett = self.quality_fc_sett(flat)
                
                fine_output_unpaved = self.quality_fc_unpaved(flat)
        
        return coarse_output, fine_output_asphalt, fine_output_concrete, fine_output_paving_stones, fine_output_sett, fine_output_unpaved

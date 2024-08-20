import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torchvision import models
#from src.utils.helper import *
from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss


class NonNegUnitNorm:
    '''Enforces all weight elements to be non-negative and each column/row to be unit norm'''
    def __init__(self, axis=1):
        self.axis = axis
    
    def __call__(self, w):
        w = w * (w >= 0).float()  # Set negative weights to zero
        norm = torch.sqrt(torch.sum(w ** 2, dim=self.axis, keepdim=True))
        w = w / (norm + 1e-8)  # Normalize each column/row to unit norm
        return w


class Condition_CNN_CLM_PRE(nn.Module):
    def __init__(self, num_c, num_classes, head):
        super(Condition_CNN_CLM_PRE, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        self.head = head
        
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
        
        self.features = model.features
        
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
        
        #Individual fine prediction branches  
        if head == 'clm':      
            self.classifier_asphalt = self._create_quality_fc_clm(num_classes=4)
            self.classifier_concrete = self._create_quality_fc_clm(num_classes=4)
            self.classifier_paving_stones = self._create_quality_fc_clm(num_classes=4)
            self.classifier_sett = self._create_quality_fc_clm(num_classes=3)
            self.classifier_unpaved = self._create_quality_fc_clm(num_classes=3)
            
        elif head == 'regression':
            self.classifier_asphalt = self._create_quality_fc_regression()
            self.classifier_concrete = self._create_quality_fc_regression()
            self.classifier_paving_stones = self._create_quality_fc_regression()
            self.classifier_sett = self._create_quality_fc_regression()
            self.classifier_unpaved = self._create_quality_fc_regression()
            
        elif head == 'corn':
            self.classifier_asphalt = self._create_quality_fc_corn(num_classes=4)
            self.classifier_concrete = self._create_quality_fc_corn(num_classes=4)
            self.classifier_paving_stones = self._create_quality_fc_corn(num_classes=4)
            self.classifier_sett = self._create_quality_fc_corn(num_classes=3)
            self.classifier_unpaved = self._create_quality_fc_corn(num_classes=3)
            
        elif head == 'single':
            self.fine_classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
                
        elif head == 'classification':
            self.fine_classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes) 
            )
            
        ### Condition part
        if head == 'regression' or head == 'single':
            self.coarse_condition = nn.Linear(num_c, num_c, bias=False)
        if head == 'corn':
            self.coarse_condition = nn.Linear(num_c, num_classes - 5, bias=False)
        else: 
            self.coarse_condition = nn.Linear(num_c, num_classes, bias=False)
        self.coarse_condition.weight.data.fill_(0)  # Initialize weights to zero
        self.constraint = NonNegUnitNorm(axis=0) 

        # criteria for different heads          
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if head == 'regression' or head == 'single':
            self.fine_criterion = nn.MSELoss
        elif head == 'clm':
            self.fine_criterion = nn.NLLLoss
        elif head == 'corn':
            self.fine_criterion = corn_loss
        else:
            self.fine_criterion = nn.CrossEntropyLoss
                     
    def _create_quality_fc_clm(self, num_classes=4):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.BatchNorm1d(1),
            CLM(classes=num_classes, link_function="logit", min_distance=0.0, use_slope=False, fixed_thresholds=False)
        )
        return layers
    
    def _create_quality_fc_regression(self):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
        return layers
    
    def _create_quality_fc_corn(self, num_classes=4):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes - 1),
        )
        return layers

        
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse, hierarchy_method = inputs
        
        x = self.features(images) 
        #x = self.avgpool(x)
        flat = x.reshape(x.size(0), -1) #([128, 32768])
        
        coarse_output = self.coarse_classifier(flat)
        coarse_probs = self.get_class_probabilies(coarse_output)
            
        if self.head == 'clm' or self.head == 'regression' or self.head == 'corn':
            fine_output_asphalt = self.classifier_asphalt(flat) #([batch_size, 1024])  
            fine_output_concrete = self.classifier_concrete(flat)
            fine_output_paving_stones = self.classifier_paving_stones(flat)      
            fine_output_sett = self.classifier_sett(flat)
            fine_output_unpaved = self.classifier_unpaved(flat)
                
            fine_output = torch.cat([fine_output_asphalt, 
                                            fine_output_concrete, 
                                            fine_output_paving_stones, 
                                            fine_output_sett, 
                                            fine_output_unpaved], 
                                            dim=1)
                    
                
        elif self.head == 'single':
            fine_output = self.fine_classifier(flat)
        
        elif self.head == 'classification':
            fine_output = self.fine_classifier(flat)
                
            
        if hierarchy_method == 'use_model_structure':
    
            if self.training:
                coarse_condition = self.coarse_condition(true_coarse)  
            else:
                coarse_condition = self.coarse_condition(coarse_probs) 
            
            if self.head == 'regression' or self.head == 'corn':    
                fine_output = torch.sum(fine_output * coarse_condition, dim=1)
            else:
                fine_output = coarse_condition + fine_output
            
        return coarse_output, fine_output     
               
    
    def get_optimizer_layers(self):
        if self.head == 'classification' or self.head == 'single':
            return self.features, self.coarse_classifier, self.fine_classifier, self.coarse_condition
        else:
            return self.features, self.coarse_classifier, self.classifier_asphalt, self.classifier_concrete, self.classifier_paving_stones, self.classifier_sett, self.classifier_unpaved, self.coarse_condition
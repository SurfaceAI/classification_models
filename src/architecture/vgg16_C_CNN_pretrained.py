import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torchvision import models
#from src.utils.helper import *
from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss
from src import constants as const
import copy
from multi_label.QWK import QWK_Loss


class NonNegUnitNorm:
    '''Enforces all weight elements to be non-negative and each column/row to be unit norm'''
    def __init__(self, axis=1):
        self.axis = axis
    
    def __call__(self, w):
        w = w * (w >= 0).float()  # Set negative weights to zero
        norm = torch.sqrt(torch.sum(w ** 2, dim=self.axis, keepdim=True))
        w = w / (norm + 1e-8)  # Normalize each column/row to unit norm
        return w


class C_CNN(nn.Module):
    def __init__(self, num_c, num_classes, head, hierarchy_method):
        super(C_CNN, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        self.head = head
        self.hierarchy_method = hierarchy_method
               
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.features = model.features
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
                
        self.block1_to_5 = model.features[:24]


        #Coarse prediction branch
        self.coarse_classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_c)
        )
        
        #Individual fine prediction branches 
        self.coarse_criterion = nn.CrossEntropyLoss
         
        if head == 'clm' or head == 'clm_qwk':      
            self.classifier_asphalt = self._create_quality_fc_clm(num_classes=4)
            self.classifier_concrete = self._create_quality_fc_clm(num_classes=4)
            self.classifier_paving_stones = self._create_quality_fc_clm(num_classes=4)
            self.classifier_sett = self._create_quality_fc_clm(num_classes=3)
            self.classifier_unpaved = self._create_quality_fc_clm(num_classes=3)
            
            if head == 'clm':
                self.fine_criterion = nn.NLLLoss
            elif head == 'clm_qwk':
                self.fine_criterion = QWK_Loss
            
        elif head == 'regression':
            self.classifier_asphalt = self._create_quality_fc_regression()
            self.classifier_concrete = self._create_quality_fc_regression()
            self.classifier_paving_stones = self._create_quality_fc_regression()
            self.classifier_sett = self._create_quality_fc_regression()
            self.classifier_unpaved = self._create_quality_fc_regression()
            
            self.fine_criterion = nn.MSELoss
            
        elif head == 'corn':
            self.classifier_asphalt = self._create_quality_fc_corn(num_classes=4)
            self.classifier_concrete = self._create_quality_fc_corn(num_classes=4)
            self.classifier_paving_stones = self._create_quality_fc_corn(num_classes=4)
            self.classifier_sett = self._create_quality_fc_corn(num_classes=3)
            self.classifier_unpaved = self._create_quality_fc_corn(num_classes=3)
            
            self.fine_criterion = corn_loss
            
        elif head == 'single':
            self.fine_classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
                
        elif head == 'classification' or head == 'classification_qwk':
            self.fine_classifier = nn.Sequential(
                nn.Linear(512 * 16 * 16, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes) 
            )
            
        if head == 'classification':
            self.fine_criterion = nn.CrossEntropyLoss
        elif head == 'classification_qwk':
            self.fine_criterion = QWK_Loss
            
        ### Condition part
        if head == 'regression' or head == 'single':
            self.coarse_condition = nn.Linear(num_c, num_c, bias=False)
        elif head == 'corn':
            self.coarse_condition = nn.Linear(num_c, num_classes - 5, bias=False)
        else: 
            self.coarse_condition = nn.Linear(num_c, num_classes, bias=False)
        self.coarse_condition.weight.data.fill_(0)  # Initialize weights to zero
        self.constraint = NonNegUnitNorm(axis=0) 

                     
    def _create_quality_fc_clm(self, num_classes=4):
        layers = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
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
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        return layers
    
    def _create_quality_fc_corn(self, num_classes=4):
        layers = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes - 1),
        )
        return layers

        
    @ staticmethod
    def get_class_probabilities(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse = inputs
        
        x = self.block1_to_5(images) 
        #x = self.avgpool(x)
        
        coarse_flat = x.reshape(x.size(0), -1) #([128, 32768])
        coarse_output = self.coarse_classifier(coarse_flat)
        coarse_probs = self.get_class_probabilities(coarse_output)
        
        fine_flat = x.reshape(x.size(0), -1)  
        
        if self.head == 'clm' or self.head == 'clm_qwk' or self.head == 'regression' or self.head == 'corn':
            fine_output_asphalt = self.classifier_asphalt(fine_flat) #([batch_size, 1024])  
            fine_output_concrete = self.classifier_concrete(fine_flat)
            fine_output_paving_stones = self.classifier_paving_stones(fine_flat)      
            fine_output_sett = self.classifier_sett(fine_flat)
            fine_output_unpaved = self.classifier_unpaved(fine_flat)
                
            fine_output = torch.cat([fine_output_asphalt, 
                                            fine_output_concrete, 
                                            fine_output_paving_stones, 
                                            fine_output_sett, 
                                            fine_output_unpaved], 
                                            dim=1)
        
        elif self.head == const.CLASSIFICATION or self.head == const.CLASSIFICATION_QWK:
            fine_output = self.fine_classifier(fine_flat)
    
        if self.training:
            coarse_condition = self.coarse_condition(true_coarse)  
        else:
            coarse_condition = self.coarse_condition(coarse_probs) 
        
        if self.hierarchy_method == const.MODELSTRUCTURE:   
            if self.head == 'regression' or self.head == 'corn':    
                fine_output = torch.sum(fine_output * coarse_condition, dim=1)
            else:
                fine_output = coarse_condition + fine_output
                self.coarse_condition.weight.data = self.constraint(self.coarse_condition.weight.data)
                
        return coarse_output, fine_output     
               
    
    def get_optimizer_layers(self):
        if self.head == const.CLASSIFICATION or self.head == const.CLASSIFICATION_QWK or self.head == 'single':
            return self.block1_to_5, self.coarse_classifier, self.fine_classifier, self.coarse_condition
        else:
            return self.block1_to_5, self.coarse_classifier, self.classifier_asphalt, self.classifier_concrete, self.classifier_paving_stones, self.classifier_sett, self.classifier_unpaved, self.coarse_condition
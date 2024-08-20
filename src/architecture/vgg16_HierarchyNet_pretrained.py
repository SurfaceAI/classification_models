import torch
import torch.nn as nn
from torchvision import models

from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss


class CustomMultLayer(nn.Module):
    def __init__(self):
        super(CustomMultLayer, self).__init__()
        
    def forward(self, tensor_1, tensor_2):
        return torch.mul(tensor_1, tensor_2)

class HierarchyNet_Pre(nn.Module):
    def __init__(self, num_c, num_classes, head):
        super(HierarchyNet_Pre, self).__init__()
        
        self.custom_layer = CustomMultLayer()
        self.head = head
        
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
              
        ### Features
        self.features = model.features
        
        self.coarse_classifier = nn.Sequential(
            nn.Linear(32768, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_c)
        )
        
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
     
    def crop(self, x, dimension, start, end):
        slices = [slice(None)] * x.dim()
        slices[dimension] = slice(start, end)
        return x[tuple(slices)]

    
    def forward(self, inputs):
        
        images, true_coarse = inputs
        
        x = self.features(images)
        
        flat = x.reshape(x.size(0), -1) 
        
        coarse_output = self.coarse_classifier(flat)
        coarse_probs = self.get_class_probabilies(coarse_output)
        
        #cropping coarse outputs
        if self.head == 'clm':
            coarse_1 = self.crop(coarse_probs, 1, 0, 1)
            coarse_2 = self.crop(coarse_probs, 1, 1, 2)
            coarse_3 = self.crop(coarse_probs, 1, 2, 3)
            coarse_4 = self.crop(coarse_probs, 1, 3, 4)
            coarse_5 = self.crop(coarse_probs, 1, 4, 5)
        else:       
            coarse_1 = self.crop(coarse_output, 1, 0, 1)
            coarse_2 = self.crop(coarse_output, 1, 1, 2)
            coarse_3 = self.crop(coarse_output, 1, 2, 3)
            coarse_4 = self.crop(coarse_output, 1, 3, 4)
            coarse_5 = self.crop(coarse_output, 1, 4, 5)
            
        #coarse_pred = F.softmax(coarse_output, dim=1)
        if self.head == 'classification':
            raw_fine_output = self.fine_classifier(flat)
            
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
           
        if self.head == 'regression' or self.head == 'corn':
            fine_1 = self.classifier_asphalt(flat) #([batch_size, 1024])
            fine_2 = self.classifier_concrete(flat)
            fine_3 = self.classifier_paving_stones(flat)           
            fine_4 = self.classifier_sett(flat)
            fine_5 = self.classifier_unpaved(flat)
                   
        else:
            raw_fine_asphalt = self.classifier_asphalt(flat) #([batch_size, 1024])
            raw_fine_concrete = self.classifier_concrete(flat)
            raw_fine_paving_stones = self.classifier_paving_stones(flat)           
            raw_fine_sett = self.classifier_sett(flat)
            raw_fine_unpaved = self.classifier_unpaved(flat)
            
            fine_1 = self.custom_layer(coarse_1, raw_fine_asphalt)
            fine_2 = self.custom_layer(coarse_2, raw_fine_concrete)
            fine_3 = self.custom_layer(coarse_3, raw_fine_paving_stones)
            fine_4 = self.custom_layer(coarse_4, raw_fine_sett)
            fine_5 = self.custom_layer(coarse_5, raw_fine_unpaved)
            
        fine_output = torch.cat([fine_1, fine_2, fine_3, fine_4, fine_5], dim=1)
        
        return coarse_output, fine_output
    
    def get_optimizer_layers(self):
        if self.head == 'classification':
            return self.features, self.coarse_classifier, self.fine_classifier
        else:
            return self.features, self.coarse_classifier, self.classifier_asphalt, self.classifier_concrete, self.classifier_paving_stones, self.classifier_sett, self.classifier_unpaved, 
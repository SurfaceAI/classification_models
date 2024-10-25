import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss
from multi_label.QWK import QWK_Loss


class VGG_16_test(nn.Module):
    def __init__(self, num_classes, head, hierarchy_method,):
        super(VGG_16_test, self).__init__()
        
        self.num_classes = num_classes
        self.head = head     
        self.hierarchy_method = hierarchy_method
           
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
            
        self.features = model.features
        
        
            
        if head == 'classification' or head == 'classification_qwk':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, 1024),
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
            
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images = inputs
        
        x = self.features(images) #[128, 64, 128, 128]
        flat = x.reshape(x.size(0), -1) #([128, 131072])
        
        if self.head == 'classification' or self.head == 'classification_qwk':
            output = self.classifier(flat)
            return output
        
        # else:
        #     fine_output_asphalt = self.classifier_asphalt(flat) #([batch_size, 1024])  
        #     fine_output_concrete = self.classifier_concrete(flat)
        #     fine_output_paving_stones = self.classifier_paving_stones(flat)      
        #     fine_output_sett = self.classifier_sett(flat)
        #     fine_output_unpaved = self.classifier_unpaved(flat)    
        
               
        #     fine_output_combined = torch.cat([fine_output_asphalt, 
        #                                     fine_output_concrete, 
        #                                     fine_output_paving_stones, 
        #                                     fine_output_sett, 
        #                                     fine_output_unpaved], 
        #                                     dim=1)
            
        #     return fine_output_combined
    
    def get_optimizer_layers(self):
        if self.head == 'classification' or self.head == 'single' or self.head == 'classification_qwk':
            return self.features, self.classifier
        # else:
        #     return self.features, self.coarse_classifier, self.classifier_asphalt, self.classifier_concrete, self.classifier_paving_stones, self.classifier_sett, self.classifier_unpaved
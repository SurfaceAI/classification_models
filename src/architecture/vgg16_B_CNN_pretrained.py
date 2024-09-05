import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss


class B_CNN(nn.Module):
    def __init__(self, num_c, num_classes, head, hierarchy_method, fc_neurons):
        super(B_CNN, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        self.head = head     
        self.fc_neurons = fc_neurons
           
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
            
        self.features = model.features
        
        #Coarse prediction branch
        self.coarse_classifier = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_c)
        )
        
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if head == 'clm':      
            self.classifier_asphalt = self._create_quality_fc_clm(num_classes=4)
            self.classifier_concrete = self._create_quality_fc_clm(num_classes=4)
            self.classifier_paving_stones = self._create_quality_fc_clm(num_classes=4)
            self.classifier_sett = self._create_quality_fc_clm(num_classes=3)
            self.classifier_unpaved = self._create_quality_fc_clm(num_classes=3)
            
            self.fine_criterion = nn.NLLLoss
            
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
            
        elif head == 'classification':
            self.fine_classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, fc_neurons),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(fc_neurons, fc_neurons),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(fc_neurons, num_classes) 
            )
            self.fine_criterion = nn.CrossEntropyLoss
            
            
    def _create_quality_fc_clm(self, num_classes=4):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, 1),
            nn.BatchNorm1d(1),
            CLM(classes=num_classes, link_function="logit", min_distance=0.0, use_slope=False, fixed_thresholds=False)
        )
        return layers
    
    def _create_quality_fc_regression(self):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, 1),
        )
        return layers
    
    def _create_quality_fc_corn(self, num_classes):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, num_classes - 1),
        )
        return layers
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse = inputs
        
        x = self.features[:17](images) #[128, 64, 128, 128]
        
        flat = x.reshape(x.size(0), -1) #[128, 262144])
        coarse_output = self.coarse_classifier(flat)
        coarse_probs = self.get_class_probabilies(coarse_output)
        
        x = self.features[17:](x) # [128, 512, 16, 16])
        
       # x = self.avgpool(x)
        flat = x.reshape(x.size(0), -1) #([128, 131072])
        
        if self.head == 'classification':
            fine_output = self.fine_classifier(flat)
            return coarse_output, fine_output
        
        else:
            fine_output_asphalt = self.classifier_asphalt(flat) #([batch_size, 1024])  
            fine_output_concrete = self.classifier_concrete(flat)
            fine_output_paving_stones = self.classifier_paving_stones(flat)      
            fine_output_sett = self.classifier_sett(flat)
            fine_output_unpaved = self.classifier_unpaved(flat)    
        
               
            fine_output_combined = torch.cat([fine_output_asphalt, 
                                            fine_output_concrete, 
                                            fine_output_paving_stones, 
                                            fine_output_sett, 
                                            fine_output_unpaved], 
                                            dim=1)
            
            return coarse_output, fine_output_combined
    
    def get_optimizer_layers(self):
        if self.head == 'classification' or self.head == 'single':
            return self.features, self.coarse_classifier, self.fine_classifier
        else:
            return self.features, self.coarse_classifier, self.classifier_asphalt, self.classifier_concrete, self.classifier_paving_stones, self.classifier_sett, self.classifier_unpaved
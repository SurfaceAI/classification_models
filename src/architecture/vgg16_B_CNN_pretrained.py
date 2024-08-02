import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss


class VGG16_B_CNN_PRE(nn.Module):
    def __init__(self, num_c, num_classes, head):
        super(VGG16_B_CNN_PRE, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        self.head = head     
           
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = False
            
        ### Block 1
        # self.block1 = model.features[:5]
        # self.block2 = model.features[5:10]
        # self.block3 = model.features[10:17]
        # self.block4 = model.features[17:24]
        # self.block5 = model.features[24:]
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
        
        ### Fine prediction branch
        # num_features = model.classifier[6].in_features
        # model.classifier[0] = nn.Linear(in_features=32768, out_features=4096, bias=True)
        # features = list(model.classifier.children())[:-1]  # select features in our last layer
        # features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        # model.classifier = nn.Sequential(*features)  # Replace the model classifier
        
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
            
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if head == 'regression':
            self.fine_criterion = nn.MSELoss
        elif head == 'clm':
            self.fine_criterion = nn.NLLLoss
        elif head == 'corn':
            self.fine_criterion = corn_loss
        else:
            self.fine_criterion = nn.CrossEntropyLoss
            
    def _create_quality_fc_clm(self, num_classes=4):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
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
    
    def _create_quality_fc_corn(self, num_classes):
        layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes - 1),
        )
        return layers
        
        # Save the modified model as a member variable
        #self.avgpool = model.avgpool #brauch ich nicht, da ich die Input feature für den Classifier angepasst habe.
        
        # self.coarse_criterion = nn.CrossEntropyLoss()
        
        # if num_classes == 1:
        #     self.fine_criterion = nn.MSELoss()
        # else:
        #     self.fine_criterion = nn.CrossEntropyLoss()
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse, hierarchy_method = inputs
        
        x = self.features[:17](images) #[128, 64, 128, 128]
        
        flat = x.reshape(x.size(0), -1) #[128, 262144])
        coarse_output = self.coarse_classifier(flat)
        coarse_probs = self.get_class_probabilies(coarse_output)
        
        x = self.features[17:](x) # [128, 512, 16, 16])
        
       # x = self.avgpool(x)
        flat = x.reshape(x.size(0), -1) #([128, 131072])
        
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
    
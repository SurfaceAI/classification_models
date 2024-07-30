import torch
import torch.nn as nn
from torchvision import models
#from src.utils.helper import *
from multi_label.CLM import CLM

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
            
                
        ### Fine prediction branch
        # num_features = model.classifier[6].in_features
        # model.classifier[0] = nn.Linear(in_features=32768, out_features=4096, bias=True)
        # features = list(model.classifier.children())[:-1]  # select features in our last layer
        # features.extend([nn.Linear(num_features, num_classes)])  # add layer with output size num_classes
        # model.classifier = nn.Sequential(*features)  # Replace the model classifier
        
        ### Condition part
        if head == 'regression':
            self.coarse_condition = nn.Linear(num_c, num_c, bias=False)
        else: 
            self.coarse_condition = nn.Linear(num_c, num_classes, bias=False)
        self.coarse_condition.weight.data.fill_(0)  # Initialize weights to zero
        self.constraint = NonNegUnitNorm(axis=0) 
        
        # Save the modified model as a member variable               
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if head == 'regression':
            self.fine_criterion = nn.MSELoss
        elif head == 'clm':
            self.fine_criterion = nn.NLLLoss
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
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        return layers

        
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
    
    def forward(self, inputs):
        
        images, true_coarse, hierarchy_method = inputs
        
        x = self.features(images) #[128, 64, 128, 128]
        # x = self.features[5:10](x) #([128, 128, 64, 64])
        # x = self.model.features[10:17] #e([128, 256, 32, 32])
        
        # x = self.block4(x) # [128, 512, 16, 16])
        # x = self.block5(x) # [128, 512, 8, 8])
        
       # x = self.avgpool(x)
        flat = x.reshape(x.size(0), -1) #([128, 32768])
        
        coarse_output = self.coarse_classifier(flat)
        
        if hierarchy_method == 'use_condition_layer':
            
        #---fine
            fine_output_asphalt = self.classifier_asphalt(flat) #([batch_size, 1024])  
            fine_output_concrete = self.classifier_concrete(flat)
            fine_output_paving_stones = self.classifier_paving_stones(flat)      
            fine_output_sett = self.classifier_sett(flat)
            fine_output_unpaved = self.classifier_unpaved(flat)

            if self.training:
                coarse_condition = self.coarse_condition(true_coarse)  
            else:
                coarse_condition = self.coarse_condition(self.get_class_probabilies(coarse_output)) 
                
            fine_output_combined = torch.cat([fine_output_asphalt, 
                                            fine_output_concrete, 
                                            fine_output_paving_stones, 
                                            fine_output_sett, 
                                            fine_output_unpaved], 
                                            dim=1)
            
            if self.head == 'regression':
                fine_output = torch.sum(fine_output_combined * coarse_condition, dim=1)
        #features = torch.add(coarse_condition, fine_raw)#
        #Adding the conditional probabilities to the dense features
            else:
                fine_output = coarse_condition + fine_output_combined
                self.coarse_condition.weight.data = self.constraint(self.coarse_condition.weight.data)
        
            return coarse_output, fine_output
        
        
        elif hierarchy_method == 'use_ground_truth': 
            
            if self.training:
                #fine_predictions = torch.zeros(x.size(0), device=x.device, dtype=torch.long)                
                #we seperate out one-hot encoded tensor to seperate one hot tensors 1=belonging to surface type, 0 not 

                fine_output_asphalt = self.classifier_asphalt(flat[true_coarse[:, 0].bool()])
                fine_output_concrete = self.classifier_concrete(flat[true_coarse[:, 1].bool()])
                fine_output_paving_stones = self.classifier_paving_stones(flat[true_coarse[:, 2].bool()])
                fine_output_sett = self.classifier_sett(flat[true_coarse[:, 3].bool()])
                fine_output_unpaved = self.classifier_unpaved(flat[true_coarse[:, 4].bool()])
                
                # if head == 'clm':
                #     fine_predictions[true_coarse[:, 0].bool()] = torch.argmax(fine_output_asphalt, dim=1)
                #     fine_predictions[true_coarse[:, 1].bool()] = torch.argmax(fine_output_concrete, dim=1)   
                #     fine_predictions[true_coarse[:, 2].bool()] = torch.argmax(fine_output_paving_stones, dim=1)    
                #     fine_predictions[true_coarse[:, 3].bool()] = torch.argmax(fine_output_sett, dim=1)                          
                #     fine_predictions[true_coarse[:, 4].bool()] = torch.argmax(fine_output_unpaved, dim=1)  
                    
                # elif head == 'regression':
                #     fine_predictions[true_coarse[:, 0].bool()] = fine_output_asphalt.round()
                #     fine_predictions[true_coarse[:, 0].bool()] = fine_output_concrete.round()
                #     fine_predictions[true_coarse[:, 0].bool()] = fine_output_paving_stones.round()
                #     fine_predictions[true_coarse[:, 0].bool()] = fine_output_sett.round()
                #     fine_predictions[true_coarse[:, 0].bool()] = fine_output_unpaved.round()
                    
                
                return coarse_output, fine_output_asphalt, fine_output_concrete, fine_output_paving_stones, fine_output_sett, fine_output_unpaved    
                
            else:
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
                    
                return coarse_output, fine_output    
    
    def get_optimizer_layers(self):
        return self.features, self.coarse_classifier, self.classifier_asphalt, self.classifier_concrete, self.classifier_paving_stones, self.classifier_sett, self.classifier_unpaved, self.coarse_condition
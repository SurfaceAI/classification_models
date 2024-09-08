import torch
import torch.nn as nn
from experiments.config import train_config
from torchvision import models
from multi_label.QWK import QWK_Loss
from multi_label.CLM import CLM
from coral_pytorch.losses import corn_loss


class CustomBayesLayer(nn.Module):
    def __init__(self):
        super(CustomBayesLayer, self).__init__()
        
    def forward(self, parent_prob, subclass_probs):
        y_subclass = torch.mul(parent_prob, subclass_probs) / torch.sum(subclass_probs, dim=1, keepdim=True)
        return y_subclass
    
class GH_CNN(nn.Module):
    def __init__(self, num_c, num_classes, head, hierarchy_method, fc_neurons):
        super(GH_CNN, self).__init__()
        
        #Custom layer
        self.custom_bayes_layer = CustomBayesLayer()
        self.fc_neurons = fc_neurons
        self.hierarchy_method = hierarchy_method
        self.head = head
        self.num_c = num_c
        self.num_classes = num_classes
        #self.custom_mult_layer = CustomMultLayer()
        
        #Load pretrained weights
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        
        # Unfreeze training for all layers in features
        for param in model.features.parameters():
            param.requires_grad = True
            
              
        ### Block 1
        self.features = model.features
        
        self.coarse_classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, num_c)
        )
        
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if head == 'clm' or head == 'clm_qwk':
            self._create_fine_classifiers_clm()
            if head == 'clm':
                self.fine_criterion = nn.NLLLoss
            elif head == 'clm_qwk':
                self.fine_criterion = QWK_Loss
            
        elif head == 'regression':
            self._create_fine_classifiers_regression()
            self.fine_criterion = nn.MSELoss
            
        elif head == 'corn':
            self._create_fine_classifiers_corn()
            self.fine_criterion = corn_loss
            
        elif head == 'classification' or head == 'classification_qwk':
            self.fine_classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, self.fc_neurons),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.fc_neurons, self.fc_neurons),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.fc_neurons, num_classes)
            )
            if head == 'classification':
                self.fine_criterion = nn.CrossEntropyLoss
            elif head == 'classification_qwk':
                self.fine_criterion = QWK_Loss
        
        #num_features = model.classifier[6].in_features
        # Modify the first fully connected layer to accept the correct input size
        # model.classifier[0] = nn.Linear(in_features=512*8*8, out_features=self.fc_neurons, bias=True)

        # # Modify other parts of the classifier if needed
        # classifier_layers = list(model.classifier.children())

        # # Save different parts of the classifier
        # self.fc = nn.Sequential(*classifier_layers[0:3])
        # self.fc_1 = nn.Sequential(*classifier_layers[3:6])

        # # Output layers for coarse and fine classification
        # self.fc_2_coarse = nn.Linear(self.fc_neurons, num_c)
        # self.fc_2_fine = nn.Linear(self.fc_neurons, num_classes)

        
        
    def _create_fine_classifiers_clm(self):
        self.classifier_asphalt = self._create_quality_fc_clm(4)
        self.classifier_concrete = self._create_quality_fc_clm(4)
        self.classifier_paving_stones = self._create_quality_fc_clm(4)
        self.classifier_sett = self._create_quality_fc_clm(3)
        self.classifier_unpaved = self._create_quality_fc_clm(3)
    
    def _create_fine_classifiers_regression(self):
        self.classifier_asphalt = self._create_quality_fc_regression()
        self.classifier_concrete = self._create_quality_fc_regression()
        self.classifier_paving_stones = self._create_quality_fc_regression()
        self.classifier_sett = self._create_quality_fc_regression()
        self.classifier_unpaved = self._create_quality_fc_regression()

    def _create_fine_classifiers_corn(self):
        self.classifier_asphalt = self._create_quality_fc_corn(4)
        self.classifier_concrete = self._create_quality_fc_corn(4)
        self.classifier_paving_stones = self._create_quality_fc_corn(4)
        self.classifier_sett = self._create_quality_fc_corn(3)
        self.classifier_unpaved = self._create_quality_fc_corn(3)

    def _create_quality_fc_clm(self, num_classes=4):
        return nn.Sequential(
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

    def _create_quality_fc_regression(self):
        return nn.Sequential(
            nn.Linear(512 * 8 * 8, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, 1)
        )

    def _create_quality_fc_corn(self, num_classes):
        return nn.Sequential(
            nn.Linear(512 * 8 * 8, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, self.fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_neurons, num_classes - 1)
        )
                
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
     
    def crop(self, x, dimension, start, end):
        slices = [slice(None)] * x.dim()
        slices[dimension] = slice(start, end)
        return x[tuple(slices)]

    
    def forward(self, inputs):
        
        images, true_coarse = inputs

        x = self.features(images) #128, 512, 8, 8]
        
        flat = x.reshape(x.size(0), -1)#torch.Size([16, 32.768])
        
        # branch_output = self.fc(flat)
        # branch_output = self.fc_1(branch_output)
        
        z_1 = self.coarse_classifier(flat) #[16,5]
        
        if self.head == 'classification' or self.head == 'classification_qwk':
            z_2 = self.fine_classifier(flat)
        else:
            fine_output_asphalt = self.classifier_asphalt(flat)
            fine_output_concrete = self.classifier_concrete(flat)
            fine_output_paving_stones = self.classifier_paving_stones(flat)
            fine_output_sett = self.classifier_sett(flat)
            fine_output_unpaved = self.classifier_unpaved(flat)
            z_2 = torch.cat([fine_output_asphalt, fine_output_concrete, fine_output_paving_stones, 
                            fine_output_sett, fine_output_unpaved], dim=1)

        z_1, z_2 = self.teacher_forcing(z_1, z_2, true_coarse)
        coarse_output, fine_output = self.bayesian_adjustment(z_1, z_2)
        
        return coarse_output, fine_output
        
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
        if self.head in ['classification', 'classification_qwk']:
            return self.features, self.coarse_classifier, self.fine_classifier
        else:
            return (self.features, self.coarse_classifier, self.classifier_asphalt, 
                    self.classifier_concrete, self.classifier_paving_stones, 
                    self.classifier_sett, self.classifier_unpaved)
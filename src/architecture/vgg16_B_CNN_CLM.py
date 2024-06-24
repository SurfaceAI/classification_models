import torch
import torch.nn as nn


class CLM(nn.Module):
    def __init__(self, num_classes, link_function, min_distance=0.35, use_slope=False, fixed_thresholds=False):
        super(CLM, self).__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.use_slope = use_slope
        self.fixed_thresholds = fixed_thresholds
        
        #if we dont have fixed thresholds, we initalize two trainable parameters 
        if not self.fixed_thresholds:
            #first threshold
            self.thresholds_b = nn.Parameter(torch.rand(1) * 0.1) #random number between 0 and 1
            #squared distance 
            self.thresholds_a = nn.Parameter(
                torch.sqrt(torch.ones(num_classes - 2) / (num_classes - 2) / 2) * torch.rand(num_classes - 2)
            )

        if self.use_slope:
            self.slope = nn.Parameter(torch.tensor(100.0))
            
    def convert_thresholds(self, b, a, min_distance=0.35):
        a = a.pow(2) + min_distance
        thresholds_param = torch.cat([b, a], dim=0).float()
        th = torch.cumsum(thresholds_param, dim=0)
        return th
    
    def nnpom(self, projected, thresholds):
        projected = projected.view(-1).float()
        
        if self.use_slope:
            projected = projected * self.slope
            thresholds = thresholds * self.slope

        m = projected.shape[0]
        a = thresholds.repeat(m, 1)
        b = projected.repeat(self.num_classes - 1, 1).t()
        z3 = a - b

        if self.link_function == 'probit':
            a3T = torch.distributions.Normal(0, 1).cdf(z3)
        elif self.link_function == 'cloglog':
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:  # logit
            a3T = torch.sigmoid(z3)

        ones = torch.ones((m, 1))
        a3 = torch.cat([a3T, ones], dim=1)
        a3 = torch.cat([a3[:, :1], a3[:, 1:] - a3[:, :-1]], dim=1)

        return a3

    def forward(self, x):
        if self.fixed_thresholds:
            thresholds = torch.linspace(0, 1, self.num_classes, dtype=torch.float32)[1:-1]
        else:
            thresholds = self.convert_thresholds(self.thresholds_b, self.thresholds_a, self.min_distance)

        return self.nnpom(x, thresholds)

    def extra_repr(self):
        return 'num_classes={}, link_function={}, min_distance={}, use_slope={}, fixed_thresholds={}'.format(
            self.num_classes, self.link_function, self.min_distance, self.use_slope, self.fixed_thresholds
        )


class B_CNN_CLM(nn.Module):
    def __init__(self, num_c, num_classes):
        super(B_CNN_CLM, self).__init__()
        
        self.num_c = num_c
        self.num_classes = num_classes
        
        
        
        
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
        
        ### Coarse branch
        #self.c_flat = nn.Flatten() 
        
        self.surface_fc = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_c)
        )
        
        
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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        
        
        self.CLM_4 = CLM(num_classes = 4, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False)
        self.CLM_3 = CLM(num_classes = 3, link_function='logit', min_distance=0.0, use_slope=False, fixed_thresholds=False)

        
        self.quality_fc_asphalt = self._create_quality_fc()
        self.quality_fc_concrete = self._create_quality_fc()
        self.quality_fc_sett = self._create_quality_fc()
        self.quality_fc_paving_stones = self._create_quality_fc()
        self.quality_fc_unpaved = self._create_quality_fc()

        
        
        # self.coarse_criterion = nn.CrossEntropyLoss()
        
        # if num_classes == 1:
        #     self.fine_criterion = nn.MSELoss()
        # else:
        #     self.fine_criterion = nn.CrossEntropyLoss()
            
    def _create_quality_fc(self):
        return nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
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
        
        images, true_coarse, hierarchy_method = inputs
        
        x = self.block1_layer1(images) #[batch_size, 64, 256, 256]
        x = self.block1_layer2(x) #[batch_size, 64, 128, 128]
        
        x = self.block2_layer1(x)#[batch_size, 64, 128, 128] 
        x = self.block2_layer2(x) #(batch_size, 128, 64, 64)
        
        x = self.block3_layer1(x)
        x = self.block3_layer2(x)
        x = self.block3_layer3(x)
        
        flat = x.reshape(x.size(0), -1) 
        coarse_output = self.surface_fc(flat)
        coarse_probs = self.get_class_probabilies(coarse_output)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        
        
        x = self.block4_layer1(x)
        x = self.block4_layer2(x) # output: [batch_size, 512 #channels, 16, 16 #height&width]
        x = self.block4_layer3(x)
        
        x = self.block5_layer1(x)
        x = self.block5_layer2(x)
        x = self.block5_layer3(x)
        
        flat = x.reshape(x.size(0), -1) #([48, 131072])
        
        #in this part the ground truth is used to guide to the correct regression quality head
        
        fine_output_asphalt = self.quality_fc_asphalt(flat)
        clm_fine_output_asphalt = self.CLM_4(fine_output_asphalt)
    
        fine_output_concrete = self.quality_fc_concrete(flat)
        clm_fine_output_concrete = self.CLM_4(fine_output_concrete)
        
        fine_output_sett = self.quality_fc_sett(flat)
        clm_fine_output_sett = self.CLM_4(fine_output_sett)
        
        fine_output_paving_stones = self.quality_fc_paving_stones(flat)
        clm_fine_output_paving_stones = self.CLM_3(fine_output_paving_stones)
        
        fine_output_unpaved = self.quality_fc_unpaved(flat)
        clm_fine_output_unpaved = self.CLM_3(fine_output_unpaved)
        
        all_fine_clm_outputs = torch.cat([clm_fine_output_asphalt, clm_fine_output_concrete, clm_fine_output_sett, clm_fine_output_paving_stones, clm_fine_output_unpaved], dim=1)
    

        
        return coarse_output, all_fine_clm_outputs

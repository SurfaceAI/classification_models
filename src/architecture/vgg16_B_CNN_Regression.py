import torch
import torch.nn as nn

class B_CNN_Regression(nn.Module):
    def __init__(self, num_c, num_classes):
        super(B_CNN_Regression, self).__init__()
        
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
        
        self.quality_fc_asphalt = self._create_quality_fc()
        self.quality_fc_concrete = self._create_quality_fc()
        self.quality_fc_sett = self._create_quality_fc()
        self.quality_fc_paving_stones = self._create_quality_fc()
        self.quality_fc_unpaved = self._create_quality_fc()

        
        self.coarse_criterion = nn.CrossEntropyLoss
        
        if num_classes == 1:
            self.fine_criterion = nn.MSELoss
        else:
            self.fine_criterion = nn.CrossEntropyLoss
            
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
        if self.training and true_coarse is not None:
            coarse_predictions = true_coarse


            fine_output = torch.zeros(x.size(0), 1, device=x.device)
            
            #we seperate out one-hot encoded tensor to seperate one hot tensors 1=belonging to surface type, 0 not 
            if coarse_predictions[:, 0].any():
                fine_output[coarse_predictions[:, 0].bool()] = self.quality_fc_asphalt(flat[coarse_predictions[:, 0].bool()])
            if coarse_predictions[:, 1].any():
                fine_output[coarse_predictions[:, 1].bool()] = self.quality_fc_concrete(flat[coarse_predictions[:, 1].bool()])
            if coarse_predictions[:, 2].any():
                fine_output[coarse_predictions[:, 2].bool()] = self.quality_fc_sett(flat[coarse_predictions[:, 2].bool()])
            if coarse_predictions[:, 3].any():
                fine_output[coarse_predictions[:, 3].bool()] = self.quality_fc_paving_stones(flat[coarse_predictions[:, 3].bool()])
            if coarse_predictions[:, 4].any():
                fine_output[coarse_predictions[:, 4].bool()] = self.quality_fc_unpaved(flat[coarse_predictions[:, 4].bool()])
            
        
        else:
            fine_output = torch.zeros(x.size(0), 1, device=x.device)
    
            if hierarchy_method == 'filter_by_probs':
                mask_asphalt = (coarse_predictions == 0)
                mask_concrete = (coarse_predictions == 1)
                mask_sett = (coarse_predictions == 2)
                mask_paving_stones = (coarse_predictions == 3)
                mask_unpaved = (coarse_predictions == 4)

                # Apply corresponding regression heads based on coarse predictions
                if mask_asphalt.any():
                    fine_output[mask_asphalt] = self.quality_fc_asphalt(flat[mask_asphalt])
                if mask_concrete.any():
                    fine_output[mask_concrete] = self.quality_fc_concrete(flat[mask_concrete])
                if mask_sett.any():
                    fine_output[mask_sett] = self.quality_fc_sett(flat[mask_sett])
                if mask_paving_stones.any():
                    fine_output[mask_paving_stones] = self.quality_fc_paving_stones(flat[mask_paving_stones])
                if mask_unpaved.any():
                    fine_output[mask_unpaved] = self.quality_fc_unpaved(flat[mask_unpaved])
                    
            if hierarchy_method == 'weighted_sum': 
                asphalt_fine = self.quality_fc_asphalt(flat)
                concrete_fine = self.quality_fc_concrete(flat)
                sett_fine = self.quality_fc_sett(flat)
                paving_stones_fine = self.quality_fc_paving_stones(flat)
                unpaved_fine = self.quality_fc_unpaved(flat)
                
                fine_output = (
                    coarse_probs[:, 0].unsqueeze(1) * asphalt_fine +
                    coarse_probs[:, 1].unsqueeze(1) * concrete_fine +
                    coarse_probs[:, 2].unsqueeze(1) * sett_fine +
                    coarse_probs[:, 3].unsqueeze(1) * paving_stones_fine +
                    coarse_probs[:, 4].unsqueeze(1) * unpaved_fine
                )
        
        return coarse_output, fine_output

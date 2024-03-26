import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config
from src.utils import preprocessing
from src import constants



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
#from torchtnt.framework.callback import Callback

import numpy as np
import os


config = train_config.rateke_flatten_params
device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

print(device)

def proba_product(tensors):
    '''
    Returns the element-wise product of two tensors.
    '''
    return torch.mul(tensors[0], tensors[1])


def to_one_hot_tensor(y, num_classes):
    y = torch.tensor(y)
    return F.one_hot(y, num_classes)

#--- coarse classes ---
num_c = 5

#--- fine classes ---
num_classes  = 18


# other parameters

#--- file paths ---

weights_store_filepath = './HierarchyNet_weights/'
train_id = '1'
model_name = 'weights_HierarchyNet_surfaceai'+train_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)


#Define custom multiplicative layer
class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, tensor_1, tensor_2):
        return torch.mul(tensor_1, tensor_2)

# Define the neural network model
class HierarchyNet(nn.Module):
    def __init__(self, num_c, num_classes):
        super(HierarchyNet, self).__init__()
        
        self.custom_layer = CustomLayer()
        
        
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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        ### Block 4
        self.block4_layer1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.block4_layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        #Here comes the flatten layer and then the dense ones for the coarse classes
        
        self.c_fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5))
        self.c_fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5))
        self.c_fc2 = (
            nn.Linear(1024, num_c) #output layer for coarse prediction
        )    
        
        # Now we create the flat layers for the fine classes
        self.fc = nn.Sequential(
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5))
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5))
        self.fc2 = (
            nn.Linear(1024, num_classes) #output layer for coarse prediction
        )   
        
        
    
    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)
     
    def crop(self, x, dimension, start, end):
        slices = [slice(None)] * x.dim()
        slices[dimension] = slice(start, end)
        return x[tuple(slices)]
    
    
    def forward(self, x):
        x = self.block1_layer1(x)
        x = self.block1_layer2(x)
        
        x = self.block2_layer1(x)
        x = self.block2_layer2(x) #(12, 128, 64, 64)
        
        x = self.block3_layer1(x)
        x = self.block3_layer2(x)
        
        x = self.block4_layer1(x)
        x = self.block4_layer2(x)
        
        flat = x.reshape(x.size(0), -1)
        
        coarse_output = self.c_fc(flat)
        coarse_output = self.c_fc1(coarse_output)
        coarse_output = self.c_fc2(coarse_output)
        
        c_1 = self.crop(coarse_output, 1, 0, 1)
        c_2 = self.crop(coarse_output, 1, 1, 2)
        c_3 = self.crop(coarse_output, 1, 2, 3)
        c_4 = self.crop(coarse_output, 1, 3, 4)
        c_5 = self.crop(coarse_output, 1, 4, 5)
        
        pre_fine_output = self.fc(flat)
        pre_fine_output = self.fc1(pre_fine_output)
        pre_fine_output = self.fc2(pre_fine_output)
        
        sub_1 = self.crop(pre_fine_output, 1, 0, 4)
        sub_2 = self.crop(pre_fine_output, 1, 4, 8)
        sub_3 = self.crop(pre_fine_output, 1, 8, 12)
        sub_4 = self.crop(pre_fine_output, 1, 12, 15)
        sub_5 = self.crop(pre_fine_output, 1, 15, 18)
        
        fine_1 = self.custom_layer(c_1, sub_1)
        fine_2 = self.custom_layer(c_2, sub_2)
        fine_3 = self.custom_layer(c_3, sub_3)
        fine_4 = self.custom_layer(c_4, sub_4)
        fine_5 = self.custom_layer(c_5, sub_5)
        
        fine_output = torch.cat([fine_1, fine_2, fine_3, fine_4, fine_5], dim=1)
        
        return coarse_output, fine_output


#loss weights for label layers

alpha = 0.3
beta = 0.7



#learning rate scheduler manual, it returns the multiplier for our initial learning rate
def lr_lambda(epoch):
  learning_rate_multi = 1.0
  if epoch > 2:
    learning_rate_multi = (1/6) # 0.003/6 to get lr = 0.0005
  if epoch > 6:
    learning_rate_multi = (1/30) # 0.003/30 to get lr = 0.0001
  return learning_rate_multi

# Loss weights modifier
class LossWeightsModifier():
    def __init__(self, alpha, beta):
        super(LossWeightsModifier, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch):
        if epoch >= 3:
            self.alpha = torch.tensor(0.5)
            self.beta = torch.tensor(0.5)
        # elif epoch >= 5:
        #     self.alpha = torch.tensor(0.2)
        #     self.beta = torch.tensor(0.8)
        # elif epoch >= 8:
        #     self.alpha = torch.tensor(0.0)
        #     self.beta = torch.tensor(1.0)
            
        return self.alpha, self.beta

# Define the data loaders and transformations

train_data, valid_data = preprocessing.create_train_validation_datasets(data_root=config.get('root_data'),
                                                                        dataset=config.get('dataset'),
                                                                        selected_classes=config.get('selected_classes'),
                                                                        validation_size=config.get('validation_size'),
                                                                        general_transform=config.get('transform'),
                                                                        augmentation=config.get('augment'),
                                                                        random_state=config.get('random_seed'),
                                                                        is_regression=config.get('is_regression'),
                                                                        level=config.get('level'),
                                                                        )


#create train and valid loader
train_loader = DataLoader(train_data, batch_size=config.get('batch_size'), shuffle=True)
valid_loader = DataLoader(train_data, batch_size=config.get('batch_size'), shuffle=False)

#create one-hot encoded tensors with the fine class labels
y_train = to_one_hot_tensor(train_data.targets, num_classes)
y_valid = to_one_hot_tensor(valid_data.targets, num_classes)


#here we define the label tree, left is the fine class (e.g. asphalt-excellent) and right the coarse (e.g.asphalt)
parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])


y_c_train = torch.zeros(y_train.size(0), num_c, dtype=torch.float32)
y_c_valid = torch.zeros(y_train.size(0), num_c, dtype=torch.float32)

# y_c_train = torch.tensor((y_train.shape[0], num_c))
# y_c_valid = torch.tensor((y_valid.shape[0], num_c))


# Transform labels for coarse level
for i in range(y_train.shape[0]):
    y_c_train[i][parent[torch.argmax(y_train[i])]] = 1.0

for j in range(y_valid.shape[0]):
    y_c_valid[j][parent[torch.argmax(y_valid[j])]] = 1.0


# Initialize the loss weights

alpha = torch.tensor(0.98)
beta = torch.tensor(0.02)

# Initialize the model, loss function, and optimizer
model = HierarchyNet(num_c=5, num_classes=18)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# Set up learning rate scheduler
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_weights_modifier = LossWeightsModifier(alpha, beta)

# Train the model
num_epochs = 10
writer = SummaryWriter('logs')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    
    for batch_index, (inputs, fine_labels) in enumerate(train_loader):
    
        inputs, labels = inputs.to(device), fine_labels.to(device)
        
        optimizer.zero_grad()
        coarse_labels = parent[fine_labels]
        
        coarse_outputs, fine_outputs = model.forward(inputs)
        coarse_loss = criterion(coarse_outputs, coarse_labels)
        fine_loss = criterion(fine_outputs, fine_labels)
        loss = alpha * coarse_loss + beta * fine_loss  #weighted loss functions for different levels
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        
        coarse_probs = model.get_class_probabilies(coarse_outputs)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
        fine_probs = model.get_class_probabilies(fine_outputs)
        fine_predictions = torch.argmax(fine_probs, dim=1)
        fine_correct += (fine_predictions == fine_labels).sum().item()
        
        # if batch_index == 0:
        #     break
    
    #learning rate step        
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    
    #loss weights step
    alpha, beta = loss_weights_modifier.on_epoch_end(epoch)
    
    # epoch_loss = running_loss /  len(inputs) * (batch_index + 1) 
    # epoch_coarse_accuracy = 100 * coarse_correct / (len(inputs) * (batch_index + 1))
    # epoch_fine_accuracy = 100 * fine_correct / (len(inputs) * (batch_index + 1))
    epoch_loss = running_loss /  len(train_loader.sampler)
    epoch_coarse_accuracy = 100 *coarse_correct / len(train_loader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(train_loader.sampler)
    
    #writer.add_scalar('Training Loss', epoch_loss, epoch)
    
    # Validation
    model.eval()
    val_running_loss = 0.0
    val_coarse_correct = 0
    val_fine_correct = 0
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(valid_loader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            
            coarse_outputs, fine_outputs = model.forward(inputs)
            
            coarse_loss = criterion(coarse_outputs, coarse_labels)
            fine_loss = criterion(fine_outputs, fine_labels)
            
            loss = alpha * coarse_loss + beta * fine_loss
            val_running_loss += loss.item() 
            
            coarse_probs = model.get_class_probabilies(coarse_outputs)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            val_coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
            fine_probs = model.get_class_probabilies(fine_outputs)
            fine_predictions = torch.argmax(fine_probs, dim=1)
            val_fine_correct += (fine_predictions == fine_labels).sum().item()
            
    #         if batch_index == 1:
    #             break
    
    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    val_epoch_loss = val_running_loss /  len(valid_loader.dataset)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(valid_loader.dataset)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(valid_loader.dataset)
    
    print(f"""
        Epoch: {epoch+1}: 
        Learning Rate: {before_lr} ->  {after_lr},
        Loss Weights: [alpha, beta] = [{alpha}, {beta}],
        Train loss: {epoch_loss:.3f}, 
        Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
        Train fine accuracy: {epoch_fine_accuracy:.3f}%,
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
        Validation fine accuracy: {val_epoch_fine_accuracy:.3f}% """)

#writer.close()

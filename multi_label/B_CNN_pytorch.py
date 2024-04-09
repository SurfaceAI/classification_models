import sys
sys.path.append('.')

from experiments.config import train_config
from src.utils import preprocessing
from src import constants
from src.architecture.vgg16_B_CNN_pretrained import VGG16_B_CNN
from src.architecture.vgg16_B_CNN import B_CNN



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import torchvision.utils as vutils
#from torchtnt.framework.callback import Callback

import wandb
import numpy as np
import os




config = train_config.B_CNN_multilabel

device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

print(device)

if config.get('wandb_on'):
    run = wandb.init(
        project=config.get('project'),
        name=config.get('name'),
        config = config
    )


def to_one_hot_tensor(y, num_classes):
    y = torch.tensor(y)
    return F.one_hot(y, num_classes)

#--- coarse classes ---
num_c = 5

#--- fine classes ---
num_classes = 18



# other parameters

#--- file paths ---

weights_store_filepath = './B_CNN_weights/'
train_id = '1'
model_name = 'weights_B_CNN_surfaceai'+train_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)


#functions
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#learning rate scheduler manual, it returns the multiplier for our initial learning rate
def lr_lambda(epoch):
  learning_rate_multi = 1.0
  if epoch > 22:
    learning_rate_multi = (1/6) # 0.003/6 to get lr = 0.0005
  if epoch > 32:
    learning_rate_multi = (1/30) # 0.003/30 to get lr = 0.0001
  return learning_rate_multi

# Loss weights modifier
class LossWeightsModifier():
    def __init__(self, alpha, beta):
        super(LossWeightsModifier, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch):
        if epoch >= 10:
            self.alpha = torch.tensor(0.6)
            self.beta = torch.tensor(0.4)
        elif epoch >= 20:
            self.alpha = torch.tensor(0.2)
            self.beta = torch.tensor(0.8)
        elif epoch >= 30:
            self.alpha = torch.tensor(0.0)
            self.beta = torch.tensor(1.0)
            
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


# mapping = {
#     1: 0, 2: 1, 3: 2, 0: 3, 5: 4, 6: 5, 7: 6, 4: 7,
#     9: 8, 10: 9, 11: 10, 8: 11, 13: 12, 14: 13, 12: 14,
#     16: 15, 15: 16, 17: 17
# }


# train_data.targets = [mapping[target] for target in train_data.targets]


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

num_classes=1

# Initialize the model, loss function, and optimizer
model = B_CNN(num_c=num_c, num_classes=num_classes)
fine_criterion = model.fine_criterion(reduction="sum")
coarse_criterion = model.coarse_criterion(reduction="sum")

optimizer = optim.SGD(model.parameters(), lr=config.get('learning_rate'), momentum=0.9)

# Set up learning rate scheduler
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_weights_modifier = LossWeightsModifier(alpha, beta)

# Train the model
#writer = SummaryWriter('logs')

for epoch in range(config.get('epochs')):
    model.train()
    running_loss = 0.0
    coarse_running_loss = 0.0
    fine_running_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    eval_metric_value_fine = 0
    
    for batch_index, (inputs, fine_labels) in enumerate(train_loader):
        
        
        # if batch_index == 0:  # Print only the first batch
        #     print("Batch Images:")
        #     images_grid = vutils.make_grid(inputs, nrow=8, padding=2, normalize=True)  # Assuming batch size is 64
        #     plt.figure(figsize=(16, 16))
        #     plt.imshow(np.transpose(images_grid, (1, 2, 0)))
        #     plt.axis('off')
        #     plt.show()

        inputs, labels = inputs.to(device), fine_labels.to(device)
        
        optimizer.zero_grad()
        
        coarse_labels = parent[fine_labels]
        
        coarse_outputs, fine_outputs = model.forward(inputs)
        coarse_loss = coarse_criterion(coarse_outputs, coarse_labels)
        fine_loss = fine_criterion(fine_outputs, fine_labels)
        loss = alpha * coarse_loss + beta * fine_loss  #weighted loss functions for different levels
        
        loss.backward()
        optimizer.step()
        
        coarse_running_loss += coarse_loss.item()
        fine_running_loss += fine_loss.item()
        running_loss += loss.item()
        
        coarse_probs = model.get_class_probabilies(coarse_outputs)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
        if config.get('eval_metric' == constants.EVAL_METRIC_ACCURACY):
            if isinstance(fine_criterion, nn.MSELoss): 
                fine_predictions = fine_outputs.round()
            else:
                fine_probs = model.get_class_probabilies(fine_outputs)
                fine_predictions = torch.argmax(fine_probs, dim=1)
            eval_metric_value_fine += (fine_predictions == labels).sum().item()

        elif config.get('eval_metric' == constants.EVAL_METRIC_MSE):
            if not isinstance(fine_criterion, nn.MSELoss): 
                raise ValueError(
                    f"Criterion must be nn.MSELoss for eval_metric {config.get('eval_metric')}"
                )
            eval_metric_value_fine = fine_running_loss
        else:
            raise ValueError(f"Unknown eval_metric: {config.get('eval_metric')}")
        
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
    epoch_loss = running_loss /  len(train_loader)
    epoch_coarse_accuracy = 100 * coarse_correct / len(train_loader.sampler)
    epoch_fine_metric = eval_metric_value_fine / len(train_loader) #TODOshould be .sampler when accuracy, else without
    
    #writer.add_scalar('Training Loss', epoch_loss, epoch)
    
    # Validation
    model.eval()
    loss = 0.0
    val_coarse_running_loss = 0.0
    val_fine_running_loss = 0.0
    val_running_loss = 0.0
    
    val_coarse_correct = 0
    val_eval_metric_fine = 0
   
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(valid_loader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            
            coarse_outputs, fine_outputs = model.forward(inputs)
            
            if isinstance(fine_criterion, nn.MSELoss):
                fine_outputs = fine_outputs.flatten()
                fine_labels = fine_labels.float()
            
            coarse_loss = coarse_criterion(coarse_outputs, coarse_labels)
            fine_loss = fine_criterion(fine_outputs, fine_labels)
            
            val_coarse_running_loss += coarse_loss.item()
            val_fine_running_loss += fine_loss.item()
            
            loss = (coarse_loss + fine_loss) / 2
            val_running_loss += loss.item() 
            
            
            coarse_probs = model.get_class_probabilies(coarse_outputs)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            val_coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
        
            if config.get('eval_metric' == constants.EVAL_METRIC_ACCURACY):
                if isinstance(fine_criterion, nn.MSELoss): 
                    fine_predictions = fine_outputs.round()
                else:
                    fine_probs = model.get_class_probabilies(fine_outputs)
                    fine_predictions = torch.argmax(fine_probs, dim=1)
                val_eval_metric_fine += (fine_predictions == labels).sum().item()

            elif config.get('eval_metric' == constants.EVAL_METRIC_MSE):
                if not isinstance(fine_criterion, nn.MSELoss): 
                    raise ValueError(
                        f"Criterion must be nn.MSELoss for eval_metric {config.get('eval_metric')}"
                    )
                val_eval_metric_fine = val_fine_running_loss
            else:
                raise ValueError(f"Unknown eval_metric: {config.get('eval_metric')}")

            
            # if batch_index == 1:
            #     break
    
    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    val_epoch_loss = val_running_loss /  len(valid_loader)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(valid_loader.sampler)
    val_epoch_fine_metric = 100 * val_eval_metric_fine / len(valid_loader) #.sampler if accuracy Todo
    
    if config.get('wandb_on'):
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "dataset": config.get('dataset'),
                    "train/loss": epoch_loss,
                    "train/accuracy/coarse": epoch_coarse_accuracy,
                    "train/metric/fine": epoch_fine_metric, 
                    "eval/loss": val_epoch_loss,
                    "eval/accuracy/coarse": val_epoch_coarse_accuracy,
                    "eval/metric/fine": val_epoch_fine_metric,
                }
            )
    
    
    print(f"""
        Epoch: {epoch+1}: 
        Learning Rate: {before_lr} ->  {after_lr},
        Loss Weights: [alpha, beta] = [{alpha}, {beta}],
        Train loss: {epoch_loss:.3f}, 
        Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
        Train fine accuracy: {epoch_fine_metric:.3f}%,
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
        Validation fine accuracy: {val_epoch_fine_metric:.3f}% """)

if config.get('wandb_on'):
        wandb.finish()
#writer.close()

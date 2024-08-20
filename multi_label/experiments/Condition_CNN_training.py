import sys
sys.path.append('.')

from experiments.config import train_config
from src.utils import preprocessing, checkpointing
from src.utils.helper import *
from src import constants

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import Counter
from datetime import datetime
import time
import wandb
import numpy as np
import os

from Archive.architectures.vgg16_Condition_CNN import Condition_CNN
from src.architecture.vgg16_C_CNN_pretrained import Condition_CNN_PRE


config = train_config.C_CNN
torch.manual_seed(config.get("seed"))
np.random.seed(config.get("seed"))



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
    

    
#--- file paths ---

level = config.get("level").split("/", 1)
type_class = None
if len(level) == 2:
    type_class = level[-1]

start_time = datetime.fromtimestamp(
        time.time() if not config.get('wandb_on') else run.start_time
    ).strftime("%Y%m%d_%H%M%S")
id = "" if not config.get('wandb_on') else "-" + run.id
saving_name = (
    config.get('level') + "-" + config.get("model") + "-" + start_time + id + ".pt"
)



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

num_classes = len(train_data.classes)
#counting the coarse classes
num_c = len(Counter([entry.split('__')[0] for entry in train_data.classes]))

# mapping = {
#     1: 0, 2: 1, 3: 2, 0: 3, 5: 4, 6: 5, 7: 6, 4: 7,
#     9: 8, 10: 9, 11: 10, 8: 11, 13: 12, 14: 13, 12: 14,
#     16: 15, 15: 16, 17: 17
# }


# train_data.targets = [mapping[target] for target in train_data.targets]


#create train and valid loader
train_loader = DataLoader(train_data, batch_size=config.get('batch_size'), shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config.get('batch_size'), shuffle=False)



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





# Initialize the model, loss function, and optimizer
model = Condition_CNN(num_c=5, num_classes=18)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.get('learning_rate'), momentum=0.9)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

checkpointer = checkpointing.CheckpointSaver(
        dirpath=config.get("root_model"),
        saving_name=saving_name,
        decreasing=True,
        config=config,
        dataset=valid_loader.dataset,
        top_n=config.get("checkpoint_top_n", const.CHECKPOINT_DEFAULT_TOP_N),
        early_stop_thresh=config.get("early_stop_thresh", const.EARLY_STOPPING_DEFAULT),
        save_state=config.get("save_state", True),
    )


for epoch in range(config.get('epochs')):
    model.train()
    running_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    
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
        coarse_one_hot = to_one_hot_tensor(coarse_labels, num_c).to(device)
        #coarse_one_hot = coarse_one_hot.type(torch.LongTensor)
        #, dtype=torch.float32
        
        #we give the coarse true labels for the conditional prob weights matrix as input to the model
        model_inputs = (inputs, coarse_one_hot)
        
        coarse_outputs, fine_outputs = model.forward(model_inputs)
        coarse_loss = criterion(coarse_outputs, coarse_labels)
        fine_loss = criterion(fine_outputs, fine_labels)
        loss = coarse_loss + fine_loss  #weighted loss functions for different levels
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        
        # if eval_metric == const.EVAL_METRIC_ACCURACY:
        #     if isinstance(criterion, nn.MSELoss): # compare with is_regression for generalization?
        #         predictions = outputs.round()
        #     else:
        #         probs = model.get_class_probabilies(outputs)
        #         predictions = torch.argmax(probs, dim=1)
        #     eval_metric_value += (predictions == labels).sum().item()

        # elif eval_metric == const.EVAL_METRIC_MSE:
        #     if not isinstance(criterion, nn.MSELoss): # compare with is_regression for generalization?
        #         raise ValueError(
        #             f"Criterion must be nn.MSELoss for eval_metric {eval_metric}"
        #         )
        #     eval_metric_value = running_loss
        # else:
        #     raise ValueError(f"Unknown eval_metric: {eval_metric}")
        
        coarse_probs = model.get_class_probabilies(coarse_outputs)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
        fine_probs = model.get_class_probabilies(fine_outputs)
        fine_predictions = torch.argmax(fine_probs, dim=1)
        fine_correct += (fine_predictions == fine_labels).sum().item()
        
        # if batch_index == 0:
        #     break
            
    # #learning rate step        
    # before_lr = optimizer.param_groups[0]["lr"]
    # scheduler.step()
    # after_lr = optimizer.param_groups[0]["lr"]
    
    # #loss weights step
    # alpha, beta = loss_weights_modifier.on_epoch_end(epoch)
    
    # epoch_loss = running_loss /  len(inputs) * (batch_index + 1) 
    # epoch_coarse_accuracy = 100 * coarse_correct / (len(inputs) * (batch_index + 1))
    # epoch_fine_accuracy = 100 * fine_correct / (len(inputs) * (batch_index + 1))
    epoch_loss = running_loss /  len(train_loader)
    epoch_coarse_accuracy = 100 * coarse_correct / len(train_loader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(train_loader.sampler)
    
    
    # Validation
    model.eval()
    loss = 0.0
    val_running_loss = 0.0
    val_coarse_correct = 0
    val_fine_correct = 0
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(valid_loader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            coarse_one_hot = to_one_hot_tensor(coarse_labels, num_c).to(device)
            
            model_inputs = (inputs, coarse_one_hot)           
            coarse_outputs, fine_outputs = model.forward(model_inputs)
            
            # if isinstance(criterion, nn.MSELoss):
            #     coarse_outputs = coarse_outputs.flatten()
            #     fine_outputs = fine_outputs.flatten()
                
            #     fine_labels = fine_labels.float()
            #     coarse_labels = coarse_labels.float()
            
            
            coarse_loss = criterion(coarse_outputs, coarse_labels)
            fine_loss = criterion(fine_outputs, fine_labels)
            
            loss = coarse_loss + fine_loss
            val_running_loss += loss.item() 
            
            coarse_probs = model.get_class_probabilies(coarse_outputs)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            val_coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
            fine_probs = model.get_class_probabilies(fine_outputs)
            fine_predictions = torch.argmax(fine_probs, dim=1)
            val_fine_correct += (fine_predictions == fine_labels).sum().item()
            
            # if batch_index == 1:
            #     break
    
    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    val_epoch_loss = val_running_loss /  len(valid_loader)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(valid_loader.sampler)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(valid_loader.sampler)
    
    early_stop = checkpointer(
            model=model, epoch=epoch, metric_val=val_epoch_loss, optimizer=optimizer
        )
    
    if config.get('wandb_on'):
        wandb.log(
            {
                "epoch": epoch + 1,
                "dataset": config.get('dataset'),
                "train/loss": epoch_loss,
                "train/accuracy/coarse": epoch_coarse_accuracy,
                "train/accuracy/fine": epoch_fine_accuracy , 
                "eval/loss": val_epoch_loss,
                "eval/accuracy/coarse": val_epoch_coarse_accuracy,
                "eval/accuracy/fine": val_epoch_fine_accuracy,
                "trainable_params": trainable_params
            }
        )
    
    print(f"""
        Epoch: {epoch+1}:,
        Train loss: {epoch_loss:.3f}, 
        Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
        Train fine accuracy: {epoch_fine_accuracy:.3f}%,
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
        Validation fine accuracy: {val_epoch_fine_accuracy:.3f}% """)
    
    if early_stop:
        print(f"Early stopped training at epoch {epoch}")
        break
    

if config.get('wandb_on'):
        wandb.finish()


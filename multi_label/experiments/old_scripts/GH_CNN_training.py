import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config
from src.utils import preprocessing, checkpointing
from src.utils.helper import * 
from src import constants
import wandb
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler

from src.architecture.vgg16_GH_CNN_pretrained import GH_CNN_PRE

from datetime import datetime
import time
import numpy as np
import os

from torch.optim.lr_scheduler import StepLR

print('hi')


config = train_config.GH_CNN_PRE
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
model_cls = string_to_object(config.get("model"))
optimizer_cls = string_to_object(config.get("optimizer"))

train_data, valid_data, trainloader, validloader, model, optimizer = training.prepare_train(model_cls=model_cls,
                optimizer_cls=optimizer_cls,
                transform=config.get("transform"),
                augment=config.get("augment"),
                dataset=config.get("dataset"),
                data_root=config.get("root_data"),
                level=level[0],
                type_class=type_class,
                selected_classes=config.get("selected_classes"),
                validation_size=config.get("validation_size"),
                batch_size=config.get("batch_size"),
                valid_batch_size=config.get("valid_batch_size"),
                learning_rate=config.get("learning_rate"),
                random_seed=config.get("seed"),
                is_regression=config.get("is_regression"),
                is_hierarchical=config.get("is_hierarchical"),
                head=config.get("head"),
                max_class_size=config.get("max_class_size"),
                freeze_convs=config.get("freeze_convs"),
                )

if config.get('lr_scheduler'):
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
    
if config.get('lw_modifier'):
    alpha = torch.tensor(0.98)
    beta = torch.tensor(0.02)
    loss_weights_modifier = LossWeightsModifier(alpha, beta)
    
#fine classes
num_classes = len(train_data.classes)
#coarse classes
num_c = len(Counter([entry.split('__')[0] for entry in train_data.classes]))

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

class LossWeightsModifier_GH():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch):
        if 0.15 * config.get('epochs') <= epoch < 0.25 * config.get('epochs'):
            self.alpha = torch.tensor(0.5)
            self.beta = torch.tensor(0.5)
        elif epoch >= 0.25 * config.get('epochs'):
            self.alpha = torch.tensor(0.0)
            self.beta = torch.tensor(1.0)
            
        return self.alpha, self.beta

# Initialize the loss weights




trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# Set up learning rate scheduler
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
if config.get('lw_modifier'):
    loss_weights_modifier = LossWeightsModifier_GH(alpha, beta)

# Train the model
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
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    if config.get('head') == 'corn':
        pass
    else:
        fine_criterion = model.fine_criterion(reduction="sum")
    
    
    running_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    
    for batch_index, (inputs, fine_labels) in enumerate(train_loader):
    
        inputs, labels = inputs.to(device), fine_labels.to(device)
        
        optimizer.zero_grad()
        coarse_labels = parent[fine_labels].to(device)
        coarse_one_hot = to_one_hot_tensor(coarse_labels, num_c).to(device)
        
        #basic model
        raw_coarse, raw_fine = model.forward(inputs)
        
        #3 different training phases
        if epoch < 0.15 * config.get('epochs'):
            coarse_outputs, fine_outputs = raw_coarse, raw_fine
            
        elif 0.15 * config.get('epochs') <= epoch < 0.25 * config.get('epochs'): 
            coarse_outputs, fine_outputs = model.teacher_forcing(raw_coarse, raw_fine, coarse_one_hot)
            
        else:
            coarse_outputs, fine_outputs = model.bayesian_adjustment(raw_coarse, raw_fine)

        coarse_loss = coarse_criterion(coarse_outputs, coarse_labels)
        fine_loss = fine_criterion(fine_outputs, fine_labels)
        
        coarse_probs = model.get_class_probabilies(coarse_outputs)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
        fine_probs = model.get_class_probabilies(fine_outputs)
        fine_predictions = torch.argmax(fine_probs, dim=1)
        fine_correct += (fine_predictions == fine_labels).sum().item()
        
        if config.get('lw_modifier'):
            loss_h = torch.sum(alpha * coarse_loss + beta * fine_loss)
        else:
            loss_h = coarse_loss + fine_loss
        
        #coarse only, weights should be (1,0)
        if epoch < 0.15 * config.get('epochs'):
            loss = loss_h   
           
        #teacher forcing 
        elif 0.15 * config.get('epochs') <= epoch < 0.25 * config.get('epochs'):
            loss = loss_h 
            
        #added calculating the loss_v (greatest error on a prediction where coarse and subclass prediction dont match)
        else:
            try:
                mismatched_indices = (coarse_predictions != parent[fine_predictions])
                max_mismatched_coarse_loss = coarse_loss[mismatched_indices].max()
                max_mismatched_fine_loss = fine_loss[mismatched_indices].max()
                loss_v = max(max_mismatched_coarse_loss, max_mismatched_fine_loss)
                loss = loss_h + loss_v
            except IndexError as e:
                print(f"IndexError encountered: {e}. Skipping this iteration and continuing.")
                loss = loss_h  # Optionally, handle the loss in some other way or set it to a default
            except Exception as e:
                print(f"An error occurred: {e}. Skipping this iteration and continuing.")
                loss = loss_h  # Optionally, handle the loss in some other way or set it to a default

        #backward step
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        
        # if batch_index == 0:
        #     break
    
    #learning rate step        
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    
    #loss weights step
    if config.get('lw_modifier'):
        alpha, beta = loss_weights_modifier.on_epoch_end(epoch)
    
    # epoch_loss = running_loss /  len(inputs) * (batch_index + 1) 
    # epoch_coarse_accuracy = 100 * coarse_correct / (len(inputs) * (batch_index + 1))
    # epoch_fine_accuracy = 100 * fine_correct / (len(inputs) * (batch_index + 1))
    epoch_loss = running_loss /  len(train_loader.sampler)
    epoch_coarse_accuracy = 100 * coarse_correct / len(train_loader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(train_loader.sampler)
    
    #writer.add_scalar('Training Loss', epoch_loss, epoch)
    
    # Validation
    model.eval()
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    if config.get('head') == 'corn':
        fine_criterion = model.fine_criterion
    else:
        fine_criterion = model.fine_criterion(reduction="sum")
    
    
    val_running_loss = 0.0
    val_coarse_correct = 0
    val_fine_correct = 0
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(valid_loader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            
            coarse_outputs, fine_outputs = model.forward(inputs)
            
            coarse_loss = coarse_criterion(coarse_outputs, coarse_labels)
            fine_loss = fine_criterion(fine_outputs, fine_labels)
            
            if config.get('lw_modifier'):
                loss = torch.sum(alpha * coarse_loss + beta * fine_loss)
            else:
                loss = coarse_loss + fine_loss
            val_running_loss += loss.item() 
            
            coarse_probs = model.get_class_probabilies(coarse_outputs)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            val_coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
            fine_probs = model.get_class_probabilies(fine_outputs)
            fine_predictions = torch.argmax(fine_probs, dim=1)
            val_fine_correct += (fine_predictions == fine_labels).sum().item()
            
            # if batch_index == 0:
            #     break
    
    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    val_epoch_loss = val_running_loss /  len(valid_loader.dataset)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(valid_loader.dataset)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(valid_loader.dataset)
    
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
                "trainable_paramater": trainable_params
            }
        )

    print(f"""
        Epoch: {epoch+1}: 
        Train loss: {epoch_loss:.3f}, 
        Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
        Train fine accuracy: {epoch_fine_accuracy:.3f}%,
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
        Validation fine accuracy: {val_epoch_fine_accuracy:.3f}% """)
    
    #Loss Weights: [alpha, beta] = [{alpha}, {beta}],
    
    if early_stop:
        print(f"Early stopped training at epoch {epoch}")
        break

if config.get('wandb_on'):
    wandb.finish()


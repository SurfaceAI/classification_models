import sys
sys.path.append('.')

from experiments.config import train_config
from src.utils import preprocessing, checkpointing
from src.utils import helper
from src import constants
from src.models import training


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from collections import Counter
from datetime import datetime
import time
import wandb
import numpy as np
import os

from torch.optim.lr_scheduler import StepLR




config = train_config.C_CNN_CLM
torch.manual_seed(config.get("seed"))
np.random.seed(config.get("seed"))



lr_scheduler = config.get("lr_scheduler")
head = config.get("head")
epsilon = 1e-9



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


model_cls = helper.string_to_object(config.get("model"))
optimizer_cls = helper.string_to_object(config.get("optimizer"))

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

# Define the data loaders and transformations

num_classes = 18
#counting the coarse classes
num_c = len(train_data.selected_classes)


#create one-hot encoded tensors with the fine class labels
y_train = helper.to_one_hot_tensor(train_data.targets, num_classes)
y_valid = helper.to_one_hot_tensor(valid_data.targets, num_classes)


#here we define the label tree, left is the fine class (e.g. asphalt-excellent) and right the coarse (e.g.asphalt)
parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])


y_c_train = torch.zeros(y_train.size(0), num_c, dtype=torch.float32)
y_c_valid = torch.zeros(y_train.size(0), num_c, dtype=torch.float32)


# Transform labels for coarse level
for i in range(y_train.shape[0]):
    y_c_train[i][parent[torch.argmax(y_train[i])]] = 1.0

for j in range(y_valid.shape[0]):
    y_c_valid[j][parent[torch.argmax(y_valid[j])]] = 1.0


# Initialize the model, loss function, and optimizer
#model = Condition_CNN_CLM(num_c=num_c, num_classes=num_classes)

# if num_classes == 1:
#     if config.get('ordinal_method') == "clm":
#         fine_criterion = nn.CrossEntropyLoss()
#     else:
#         fine_criterion = nn.MSELoss()
# else:
#     fine_criterion = nn.CrossEntropyLoss()


#optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate'), momentum=0.9)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)



if lr_scheduler:
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

checkpointer = checkpointing.CheckpointSaver(
        dirpath=config.get("root_model"),
        saving_name=saving_name,
        decreasing=True,
        config=config,
        dataset=validloader.dataset,
        top_n=config.get("checkpoint_top_n", constants.CHECKPOINT_DEFAULT_TOP_N),
        early_stop_thresh=config.get("early_stop_thresh", constants.EARLY_STOPPING_DEFAULT),
        save_state=config.get("save_state", True),
    )


for epoch in range(config.get('epochs')):
    model.train()
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    fine_criterion = model.fine_criterion(reduction="sum")

    running_loss = 0.0
    coarse_loss_total = 0.0
    fine_loss_total = 0.0
    
    fine_loss_asphalt_total = 0.0
    fine_loss_concrete_total = 0.0
    fine_loss_sett_total = 0.0
    fine_loss_paving_stones_total = 0.0
    fine_loss_unpaved_total = 0.0
    
    coarse_correct = 0
    fine_correct = 0
    
    fine_correct_asphalt = 0
    fine_correct_concrete = 0
    fine_correct_sett = 0
    fine_correct_paving_stones = 0
    fine_correct_unpaved = 0
    
    
    for batch_index, (inputs, fine_labels) in enumerate(trainloader):
              
        
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
        coarse_one_hot = helper.to_one_hot_tensor(coarse_labels, num_c).to(device)
        #coarse_one_hot = coarse_one_hot.type(torch.LongTensor)
        #, dtype=torch.float32

        #we give the coarse true labels for the conditional prob weights matrix as input to the model
        model_inputs = (inputs, coarse_one_hot, config.get('hierarchy_method'))
        
        if config.get('hierarchy_method') == 'use_ground_truth':
            
            fine_labels_mapped = torch.tensor([helper.map_quality_to_continuous(label) for label in fine_labels], dtype=torch.long).to(device)
            
            asphalt_mask = (coarse_labels == 0)
            concrete_mask = (coarse_labels == 1)
            paving_stones_mask = (coarse_labels == 2)
            sett_mask = (coarse_labels == 3)
            unpaved_mask = (coarse_labels == 4)
            
            fine_labels_mapped_aspahlt = fine_labels_mapped[asphalt_mask]
            fine_labels_mapped_concrete = fine_labels_mapped[concrete_mask]
            fine_labels_mapped_paving_stones = fine_labels_mapped[paving_stones_mask]
            fine_labels_mapped_sett = fine_labels_mapped[sett_mask]
            fine_labels_mapped_unpaved = fine_labels_mapped[unpaved_mask]
            
            
            coarse_output, fine_output_asphalt, fine_output_concrete, fine_output_paving_stones, fine_output_sett, fine_output_unpaved = model.forward(model_inputs)
                             
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            
            fine_loss_asphalt = fine_criterion(torch.log(fine_output_asphalt), fine_labels_mapped_aspahlt)
            fine_loss_concrete = fine_criterion(torch.log(fine_output_concrete), fine_labels_mapped_concrete)
            fine_loss_paving_stones = fine_criterion(torch.log(fine_output_paving_stones), fine_labels_mapped_paving_stones)
            fine_loss_sett = fine_criterion(torch.log(fine_output_sett), fine_labels_mapped_sett)
            fine_loss_unpaved = fine_criterion(torch.log(fine_output_unpaved), fine_labels_mapped_unpaved)
            
            fine_loss_asphalt = torch.nan_to_num(fine_loss_asphalt, nan=0.0)
            fine_loss_concrete = torch.nan_to_num(fine_loss_concrete, nan=0.0)
            fine_loss_paving_stones = torch.nan_to_num(fine_loss_paving_stones, nan=0.0)
            fine_loss_sett = torch.nan_to_num(fine_loss_sett, nan=0.0)
            fine_loss_unpaved = torch.nan_to_num(fine_loss_unpaved, nan=0.0)
        
            fine_loss = 1/5 * fine_loss_asphalt + 1/5 * fine_loss_concrete + 1/5 * fine_loss_sett + 1/5 * fine_loss_paving_stones + 1/5 * fine_loss_unpaved
            
            loss = coarse_loss + fine_loss  #weighted loss functions for different levels
        
            loss.backward()
            #plot_grad_flow(model.named_parameters())
            optimizer.step()
            running_loss += loss.item()
            
            coarse_loss_total += coarse_loss.item()
            fine_loss_total += fine_loss.item()
            
            fine_loss_asphalt_total += fine_loss_asphalt.item()
            fine_loss_concrete_total += fine_loss_concrete.item()
            fine_loss_paving_stones_total += fine_loss_paving_stones.item()
            fine_loss_sett_total += fine_loss_sett.item()
            fine_loss_unpaved_total += fine_loss_unpaved.item()
            
            coarse_output = model.get_class_probabilies(coarse_output)
            coarse_predictions = torch.argmax(coarse_output, dim=1)
            coarse_correct += (coarse_predictions == coarse_labels).sum().item()
            
            if head == 'clm':
                fine_predictions = torch.argmax(fine_output, dim=1)
            # if head == 'regression':
            #     fine_predictions = = fine_output.round()
            # else:
            #     probs = model.get_class_probabilies(outputs)
            #     predictions = torch.argmax(probs, dim=1)
            if head == 'clm':
                fine_predictions_asphalt = torch.argmax(fine_output_asphalt, dim=1)                
                fine_predictions_concrete = torch.argmax(fine_output_concrete, dim=1)
                fine_predictions_sett = torch.argmax(fine_output_sett, dim=1)
                fine_predictions_paving_stones = torch.argmax(fine_output_paving_stones, dim=1)
                fine_predictions_unpaved = torch.argmax(fine_output_unpaved, dim=1)
                                
            if head == 'regression':
                fine_predictions_asphalt = fine_output_asphalt.round()
                fine_predictions_concrete = fine_output_concrete.round()
                fine_predictions_sett = fine_output_sett.round()
                fine_predictions_paving_stones = fine_output_paving_stones.round()
                fine_predictions_unpaved = fine_output_unpaved.round()
            
            fine_correct_asphalt += (fine_predictions_asphalt == fine_labels_mapped_aspahlt).sum().item()
            fine_correct_concrete += (fine_predictions_concrete == fine_labels_mapped_concrete).sum().item()
            fine_correct_sett += (fine_predictions_sett == fine_labels_mapped_sett).sum().item()
            fine_correct_paving_stones += (fine_predictions_paving_stones == fine_labels_mapped_paving_stones).sum().item()
            fine_correct_unpaved += (fine_predictions_unpaved == fine_labels_mapped_unpaved).sum().item()
            fine_correct = fine_correct_asphalt + fine_correct_concrete + fine_correct_sett + fine_correct_paving_stones + fine_correct_unpaved
            
            asphalt_fine_epoch_loss = fine_loss_asphalt_total / len(trainloader)
            concrete_fine_epoch_loss = fine_loss_concrete_total / len(trainloader)
            sett_fine_epoch_loss = fine_loss_sett_total / len(trainloader)
            paving_stones_fine_epoch_loss = fine_loss_paving_stones_total / len(trainloader)
            unpaved_fine_epoch_loss = fine_loss_unpaved_total / len(trainloader)
            
            # if batch_index == 0:
            #     break
        
        
        elif config.get('hierarchy_method') == 'use_condition_layer':
            
            coarse_output, fine_output = model.forward(model_inputs)
            
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            fine_loss = fine_criterion(torch.log(fine_output + epsilon), fine_labels)
        
            loss = coarse_loss + fine_loss  #weighted loss functions for different levels
            
            loss.backward()
            #plot_grad_flow(model.named_parameters())
            print(f'CPWM before optimizer step: {model.coarse_condition.weight.data}')
            print("Gradients:", model.coarse_condition.weight.grad)
            optimizer.step()
            print(f'CPWM after optimizer step: {model.coarse_condition.weight.data}')
            
            running_loss += loss.item()
            
            coarse_loss_total += coarse_loss.item()
            fine_loss_total += fine_loss.item()
        
            coarse_output = model.get_class_probabilies(coarse_output)
            coarse_predictions = torch.argmax(coarse_output, dim=1)
            coarse_correct += (coarse_predictions == coarse_labels).sum().item()
            
            if head == 'clm':
                fine_predictions = torch.argmax(fine_output, dim=1)
            if head == 'regression':
                fine_predictions = fine_output.round()
            else:
                probs = model.get_class_probabilies(fine_output)
                predictions = torch.argmax(probs, dim=1)
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
    
    epoch_loss = running_loss /  len(trainloader)
    epoch_coarse_accuracy = 100 * coarse_correct / len(trainloader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(trainloader.sampler)
    
    coarse_epoch_loss = coarse_loss_total / len(trainloader)
    fine_epoch_loss = fine_loss_total / len(trainloader)



    # Validation
    model.eval()
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    fine_criterion = model.fine_criterion(reduction="sum")
    
    
    val_running_loss = 0.0
    val_coarse_loss_total = 0.0
    val_fine_loss_total = 0.0
    
    val_coarse_correct = 0
    val_fine_correct = 0
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(validloader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            coarse_one_hot = helper.to_one_hot_tensor(coarse_labels, num_c).to(device)
            
            model_inputs = (inputs, coarse_one_hot, config.get('hierarchy_method'))   
            
            coarse_output, fine_output = model.forward(model_inputs)
            
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            fine_loss = fine_criterion(torch.log(fine_output + epsilon), fine_labels)
                        
            loss = coarse_loss + fine_loss  #weighted loss functions for different levels
        
            val_running_loss += loss.item()
            
            val_coarse_loss_total += coarse_loss.item()
            val_fine_loss_total += fine_loss.item()
            
            val_coarse_output = model.get_class_probabilies(coarse_output)
            val_coarse_predictions = torch.argmax(val_coarse_output, dim=1)
            val_coarse_correct += (val_coarse_predictions == coarse_labels).sum().item()
            
            if head == 'clm':
                val_fine_predictions = torch.argmax(fine_output, dim=1)
            if head == 'regression':
                val_fine_predictions = fine_output.round()
            else:
                probs = model.get_class_probabilies(fine_output)
                predictions = torch.argmax(probs, dim=1)
            val_fine_correct += (val_fine_predictions == fine_labels).sum().item()

            # if batch_index == 0:
            #     break
            
            # if isinstance(criterion, nn.MSELoss):
            #     coarse_output = coarse_output.flatten()
            #     fine_output = fine_output.flatten()
                
            #     fine_labels = fine_labels.float()
            #     coarse_labels = coarse_labels.float()
    
    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    
    val_epoch_loss = val_running_loss /  len(validloader)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(validloader.sampler)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(validloader.sampler)
    
    val_coarse_epoch_loss = val_coarse_loss_total / len(validloader)
    val_fine_epoch_loss = val_fine_loss_total / len(validloader)

    if lr_scheduler:
        scheduler.step()


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
                "trainable_params": trainable_params,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
    
    print(f"""
        Epoch: {epoch+1}:,
        
        Train loss: {epoch_loss:.3f},
        
        Coarse train loss: {coarse_epoch_loss:.3f},
        Fine train loss: {fine_epoch_loss:.3f}, 
        
        Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
        Train fine accuracy: {epoch_fine_accuracy:.3f}%,

        
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
        Validation fine accuracy: {val_epoch_fine_accuracy:.3f}% 
        
        Learning_rate: {scheduler.get_last_lr()[0]}
        
        """)
    
    if early_stop:
        print(f"Early stopped training at epoch {epoch}")
        break
    

if config.get('wandb_on'):
        wandb.finish()    


        # Fine asphalt train loss: {asphalt_fine_epoch_loss:.3f},
        # Fine asphalt train loss: {concrete_fine_epoch_loss:.3f},
        # Fine asphalt train loss: {sett_fine_epoch_loss:.3f},
        # Fine asphalt train loss: {paving_stones_fine_epoch_loss:.3f},
        # Fine asphalt train loss: {unpaved_fine_epoch_loss:.3f},
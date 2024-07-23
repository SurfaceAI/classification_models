import sys
sys.path.append('.')

from experiments.config import train_config
from src.utils import preprocessing, checkpointing
from src.utils.helper import *
from src import constants

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

from src.architecture.vgg16_Condition_CNN import Condition_CNN
from src.architecture.vgg16_Condition_CNN_pretrained import Condition_CNN_PRE
from src.architecture.vgg16_Condition_CNN_CLM import Condition_CNN_CLM


config = train_config.C_CNN_CLM
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

#limit class size
    # limit max class size
if config.get('max_class_size') is not None:
    # define indices with max number of class size
    indices = []
    class_counts = {}
    # TODO: randomize sample picking?
    for i, label in enumerate(train_data.targets):
        if label not in class_counts:
            class_counts[label] = 0
        if class_counts[label] < config.get('max_class_size'):
            indices.append(i)
            class_counts[label] += 1
        # stop if all classes are filled
        if all(count >= config.get('max_class_size') for count in class_counts.values()):
            break

    # create a) (Subset with indices + WeightedRandomSampler) or b) (SubsetRandomSampler) (no weighting, if max class size larger than smallest class size!)
    # b) SubsetRandomSampler ? 
    #    Samples elements randomly from a given list of indices, without replacement.
    # a):
    train_data = Custom_Subset(train_data, indices)


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


if config.get('is_regression'):
    num_classes = 1
    
    if config.get('ordinal_method') == "clm":
        num_classes = 4
    
else:
    num_classes = len(train_data.classes)


# Initialize the model, loss function, and optimizer
model = Condition_CNN_CLM(num_c=num_c, num_classes=num_classes)
criterion = nn.NLLLoss(reduction="sum")

# if num_classes == 1:
#     if config.get('ordinal_method') == "clm":
#         fine_criterion = nn.CrossEntropyLoss()
#     else:
#         fine_criterion = nn.MSELoss()
# else:
#     fine_criterion = nn.CrossEntropyLoss()


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
    
    fine_correct_asphalt = 0
    fine_correct_concrete = 0
    fine_correct_sett = 0
    fine_correct_paving_stones = 0
    fine_correct_unpaved = 0
        
    fine_loss_total = 0.0
    coarse_loss_total = 0.0
    
    fine_loss_asphalt_total = 0.0
    fine_loss_concrete_total = 0.0
    fine_loss_sett_total = 0.0
    fine_loss_paving_stones_total = 0.0
    fine_loss_unpaved_total = 0.0
    
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
        
        if config.get('clm'):
            fine_labels_mapped = torch.tensor([map_quality_to_continuous(label) for label in fine_labels], dtype=torch.long).to(device)
            
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
        
        #we give the coarse true labels for the conditional prob weights matrix as input to the model
        model_inputs = (inputs, coarse_one_hot, config.get('hierarchy_method'))
        
        coarse_outputs, fine_probs_asphalt, fine_probs_concrete, fine_probs_sett, fine_probs_paving_stones, fine_probs_unpaved = model.forward(model_inputs)
        
        coarse_loss = criterion(coarse_outputs, coarse_labels)
        
        
        fine_loss_asphalt = criterion(torch.log(fine_probs_asphalt), fine_labels_mapped_aspahlt)
        fine_loss_concrete = criterion(torch.log(fine_probs_concrete), fine_labels_mapped_concrete)
        fine_loss_paving_stones = criterion(torch.log(fine_probs_paving_stones), fine_labels_mapped_paving_stones)
        fine_loss_sett = criterion(torch.log(fine_probs_sett), fine_labels_mapped_sett)
        fine_loss_unpaved = criterion(torch.log(fine_probs_unpaved), fine_labels_mapped_unpaved)
        
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
        fine_loss_sett_total += fine_loss_sett.item()
        fine_loss_paving_stones_total += fine_loss_paving_stones.item()
        fine_loss_unpaved_total += fine_loss_unpaved.item()
                
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
        
        #fine_probs = model.get_class_probabilies(fine_outputs)
        fine_predictions_asphalt = torch.argmax(fine_probs_asphalt, dim=1)
        fine_correct_asphalt += (fine_predictions_asphalt == fine_labels_mapped_aspahlt).sum().item()
        
        fine_predictions_concrete = torch.argmax(fine_probs_concrete, dim=1)
        fine_correct_concrete += (fine_predictions_concrete == fine_labels_mapped_concrete).sum().item()

        fine_predictions_sett = torch.argmax(fine_probs_sett, dim=1)
        fine_correct_sett += (fine_predictions_sett == fine_labels_mapped_sett).sum().item()

        fine_predictions_paving_stones = torch.argmax(fine_probs_paving_stones, dim=1)
        fine_correct_paving_stones += (fine_predictions_paving_stones == fine_labels_mapped_paving_stones).sum().item()

        fine_predictions_unpaved = torch.argmax(fine_probs_unpaved, dim=1)
        fine_correct_unpaved += (fine_predictions_unpaved == fine_labels_mapped_unpaved).sum().item()

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
    fine_correct = fine_correct_asphalt + fine_correct_concrete + fine_correct_sett + fine_correct_paving_stones + fine_correct_unpaved
    
    epoch_loss = running_loss /  len(train_loader)
    epoch_coarse_accuracy = 100 * coarse_correct / len(train_loader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(train_loader.sampler)
    
    coarse_epoch_loss = coarse_loss_total / len(train_loader)
    fine_epoch_loss = fine_loss_total / len(train_loader)
    
    asphalt_fine_epoch_loss = fine_loss_asphalt_total / len(train_loader)
    concrete_fine_epoch_loss = fine_loss_concrete_total / len(train_loader)
    sett_fine_epoch_loss = fine_loss_sett_total / len(train_loader)
    paving_stones_fine_epoch_loss = fine_loss_paving_stones_total / len(train_loader)
    unpaved_fine_epoch_loss = fine_loss_unpaved_total / len(train_loader)


    # Validation
    model.eval()
    loss = 0.0
    val_running_loss = 0.0
    val_coarse_correct = 0
    val_fine_correct = 0
    
    # val_fine_correct_asphalt = 0
    # val_fine_correct_concrete = 0
    # val_fine_correct_sett = 0
    # val_fine_correct_paving_stones = 0
    # val_fine_correct_unpaved = 0
    
    val_fine_loss_total = 0.0
    val_coarse_loss_total = 0.0
    
    val_fine_loss_asphalt_total = 0.0
    val_fine_loss_concrete_total = 0.0
    val_fine_loss_sett_total = 0.0
    val_fine_loss_paving_stones_total = 0.0
    val_fine_loss_unpaved_total = 0.0
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(valid_loader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            coarse_one_hot = to_one_hot_tensor(coarse_labels, num_c).to(device)
            
            model_inputs = (inputs, coarse_one_hot, config.get('hierarchy_method'))           
            coarse_outputs, fine_probs_asphalt, fine_probs_concrete, fine_probs_sett, fine_probs_paving_stones, fine_probs_unpaved = model.forward(model_inputs)
            
            fine_outputs = torch.cat([fine_probs_asphalt, fine_probs_concrete, fine_probs_sett, fine_probs_paving_stones, fine_probs_unpaved], dim=1)
            
            # if isinstance(criterion, nn.MSELoss):
            #     coarse_outputs = coarse_outputs.flatten()
            #     fine_outputs = fine_outputs.flatten()
                
            #     fine_labels = fine_labels.float()
            #     coarse_labels = coarse_labels.float()
            
            
            coarse_loss = criterion(coarse_outputs, coarse_labels)
            
            fine_loss = criterion(fine_outputs, fine_labels)
            
            # fine_loss_asphalt = criterion(fine_probs_asphalt, fine_labels_mapped_aspahlt)
            # fine_loss_concrete = criterion(fine_probs_concrete, fine_labels_mapped_concrete)
            # fine_loss_sett = criterion(fine_probs_sett, fine_labels_mapped_sett)
            # fine_loss_paving_stones = criterion(fine_probs_paving_stones, fine_labels_mapped_paving_stones)
            # fine_loss_unpaved = criterion(fine_probs_unpaved, fine_labels_mapped_unpaved)
            
            # fine_loss_asphalt = torch.nan_to_num(fine_loss_asphalt, nan=0.0)
            # fine_loss_concrete = torch.nan_to_num(fine_loss_concrete, nan=0.0)
            # fine_loss_sett = torch.nan_to_num(fine_loss_sett, nan=0.0)
            # fine_loss_paving_stones = torch.nan_to_num(fine_loss_paving_stones, nan=0.0)
            # fine_loss_unpaved = torch.nan_to_num(fine_loss_unpaved, nan=0.0)
            
            #fine_loss = 1/5 * fine_loss_asphalt + 1/5 * fine_loss_concrete + 1/5 * fine_loss_sett + 1/5 * fine_loss_paving_stones + 1/5 * fine_loss_unpaved
            
            loss = coarse_loss + fine_loss
            val_running_loss += loss.item() 
            
            val_coarse_loss_total += coarse_loss.item()
            val_fine_loss_total += fine_loss.item()
            
            # val_fine_loss_asphalt_total += fine_loss_asphalt.item()
            # val_fine_loss_concrete_total += fine_loss_concrete.item()
            # val_fine_loss_sett_total += fine_loss_sett.item()
            # val_fine_loss_paving_stones_total += fine_loss_paving_stones.item()
            # val_fine_loss_unpaved_total += fine_loss_unpaved.item()
                
            coarse_probs = model.get_class_probabilies(coarse_outputs)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            val_coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
            #fine_probs = model.get_class_probabilies(fine_outputs)
            fine_predictions = torch.argmax(fine_outputs, dim=1)
            val_fine_correct += (fine_predictions == fine_labels).sum().item()
            # fine_predictions_asphalt = torch.argmax(fine_probs_asphalt, dim=1)
            # val_fine_correct_asphalt += (fine_predictions_asphalt == fine_labels_mapped_aspahlt).sum().item()
            
            # fine_predictions_concrete = torch.argmax(fine_probs_concrete, dim=1)
            # val_fine_correct_concrete += (fine_predictions_concrete == fine_labels_mapped_concrete).sum().item()

            # fine_predictions_sett = torch.argmax(fine_probs_sett, dim=1)
            # val_fine_correct_sett += (fine_predictions_sett == fine_labels_mapped_sett).sum().item()

            # fine_predictions_paving_stones = torch.argmax(fine_probs_paving_stones, dim=1)
            # val_fine_correct_paving_stones += (fine_predictions_paving_stones == fine_labels_mapped_paving_stones).sum().item()

            # fine_predictions_unpaved = torch.argmax(fine_probs_unpaved, dim=1)
            # val_fine_correct_unpaved += (fine_predictions_unpaved == fine_labels_mapped_unpaved).sum().item()

            
            # if batch_index == 0:
            #     break
    
    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    #val_fine_correct = val_fine_correct_asphalt + val_fine_correct_concrete + val_fine_correct_sett + val_fine_correct_paving_stones + val_fine_correct_unpaved
    
    val_epoch_loss = val_running_loss /  len(valid_loader)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(valid_loader.sampler)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(valid_loader.sampler)
    
    val_coarse_epoch_loss = val_coarse_loss_total / len(valid_loader)
    val_fine_epoch_loss = val_fine_loss_total / len(valid_loader)
    
    val_asphalt_fine_epoch_loss = val_fine_loss_asphalt_total / len(valid_loader)
    val_concrete_fine_epoch_loss = val_fine_loss_concrete_total / len(valid_loader)
    val_sett_fine_epoch_loss = val_fine_loss_sett_total / len(valid_loader)
    val_paving_stones_fine_epoch_loss = val_fine_loss_paving_stones_total / len(valid_loader)
    val_unpaved_fine_epoch_loss = val_fine_loss_unpaved_total / len(valid_loader)

    
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


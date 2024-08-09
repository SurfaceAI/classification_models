import sys
sys.path.append('.')

from experiments.config import train_config
from src.utils import preprocessing, checkpointing
from src.utils import helper
from src import constants
from src.models import training
import matplotlib.pyplot as plt
from multi_label.ordinal_metrics import accuracy_off1, minimum_sensitivity

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms as transforms

from collections import Counter
from datetime import datetime
import time
import wandb
import numpy as np
import os
import torchvision.utils as vutils

from torch.optim.lr_scheduler import StepLR

from coral_pytorch.dataset import corn_label_from_logits

from torchcam.methods import SmoothGradCAMpp

#config = train_config.C_CNN_CLM
config = train_config.H_NET_PRE
torch.manual_seed(config.get("seed"))
np.random.seed(config.get("seed"))
#
lr_scheduler = config.get("lr_scheduler")
head = config.get("head")
epsilon = 1e-9
lw_modifier = config.get("lw_modifier")

#to_pil_image = transforms.ToPILImage()

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
    
def map_predictions_to_quality(predictions, surface_type):
    quality_mapping = {
        "asphalt": [0, 1, 2, 3],  # Modify as needed
        "concrete": [4, 5, 6, 7],
        "paving_stones": [8, 9, 10, 11],
        "sett": [12, 13, 14],
        "unpaved": [15, 16, 17]
    }
    return torch.tensor([quality_mapping[surface_type][pred] for pred in predictions], dtype=torch.long)


def compute_fine_losses(fine_output, fine_labels_mapped, masks, head):
    fine_loss = 0.0
    
    if head == 'regression':
        fine_output_asphalt = fine_output[:, 0:1].float()
        fine_output_concrete = fine_output[:, 1:2].float()
        fine_output_paving_stones = fine_output[:, 2:3].float()
        fine_output_sett = fine_output[:, 3:4].float()
        fine_output_unpaved = fine_output[:, 4:5].float()
    
    elif head == 'corn':
        fine_output_asphalt = fine_output[:, 0:3]
        fine_output_concrete = fine_output[:, 3:6]
        fine_output_paving_stones = fine_output[:, 6:9]
        fine_output_sett = fine_output[:, 9:11]
        fine_output_unpaved = fine_output[:, 11:13]
                # Separate the fine outputs
    else:
        fine_output_asphalt = fine_output[:, 0:4]
        fine_output_concrete = fine_output[:, 4:8]
        fine_output_paving_stones = fine_output[:, 8:12]
        fine_output_sett = fine_output[:, 12:15]
        fine_output_unpaved = fine_output[:, 15:18]
        
    
    # Extract the masks
    asphalt_mask, concrete_mask, paving_stones_mask, sett_mask, unpaved_mask = masks
    
    # Get the labels for each surface type
    fine_labels_mapped_asphalt = fine_labels_mapped[asphalt_mask]
    fine_labels_mapped_concrete = fine_labels_mapped[concrete_mask]
    fine_labels_mapped_paving_stones = fine_labels_mapped[paving_stones_mask]
    fine_labels_mapped_sett = fine_labels_mapped[sett_mask]
    fine_labels_mapped_unpaved = fine_labels_mapped[unpaved_mask]

    three_mask_sett = (fine_labels_mapped_sett != 3)
    fine_labels_mapped_sett = fine_labels_mapped_sett[three_mask_sett]
    
    three_mask_unpaved = (fine_labels_mapped_unpaved != 3)
    fine_labels_mapped_unpaved = fine_labels_mapped_unpaved[three_mask_unpaved]
    
    # Compute the loss for each surface type
    if head == 'clm':
        fine_loss_asphalt = nn.NLLLoss()(torch.log(fine_output_asphalt[asphalt_mask] + 1e-9), fine_labels_mapped_asphalt)
        fine_loss_concrete = nn.NLLLoss()(torch.log(fine_output_concrete[concrete_mask] + 1e-9), fine_labels_mapped_concrete)
        fine_loss_paving_stones = nn.NLLLoss()(torch.log(fine_output_paving_stones[paving_stones_mask] + 1e-9), fine_labels_mapped_paving_stones)
        fine_loss_sett = nn.NLLLoss()(torch.log(fine_output_sett[sett_mask][three_mask_sett] + 1e-9), fine_labels_mapped_sett)
        fine_loss_unpaved = nn.NLLLoss()(torch.log(fine_output_unpaved[unpaved_mask][three_mask_unpaved] + 1e-9), fine_labels_mapped_unpaved)
    elif head == 'corn':
        fine_loss_asphalt = model.fine_criterion(fine_output_asphalt[asphalt_mask], fine_labels_mapped_asphalt, 4)
        fine_loss_concrete = model.fine_criterion(fine_output_concrete[concrete_mask], fine_labels_mapped_concrete, 4)
        fine_loss_paving_stones = model.fine_criterion(fine_output_paving_stones[paving_stones_mask], fine_labels_mapped_paving_stones, 4)
        fine_loss_sett = model.fine_criterion(fine_output_sett[sett_mask][three_mask_sett], fine_labels_mapped_sett, 3)
        fine_loss_unpaved = model.fine_criterion(fine_output_unpaved[unpaved_mask][three_mask_unpaved], fine_labels_mapped_unpaved, 3)
    elif head == 'regression':
        fine_loss_asphalt = nn.MSELoss()(fine_output_asphalt[asphalt_mask].flatten(), fine_labels_mapped_asphalt.float())
        fine_loss_concrete = nn.MSELoss()(fine_output_concrete[concrete_mask].flatten(), fine_labels_mapped_concrete.float())
        fine_loss_paving_stones = nn.MSELoss()(fine_output_paving_stones[paving_stones_mask].flatten(), fine_labels_mapped_paving_stones.float())
        fine_loss_sett = nn.MSELoss()(fine_output_sett[sett_mask][three_mask_sett].flatten(), fine_labels_mapped_sett.float())
        fine_loss_unpaved = nn.MSELoss()(fine_output_unpaved[unpaved_mask][three_mask_unpaved].flatten(), fine_labels_mapped_unpaved.float())
             
    fine_loss_asphalt = torch.nan_to_num(fine_loss_asphalt, nan=0.0)
    fine_loss_concrete = torch.nan_to_num(fine_loss_concrete, nan=0.0)
    fine_loss_paving_stones = torch.nan_to_num(fine_loss_paving_stones, nan=0.0)
    fine_loss_sett = torch.nan_to_num(fine_loss_sett, nan=0.0)
    fine_loss_unpaved = torch.nan_to_num(fine_loss_unpaved, nan=0.0)

    
    # Combine the losses
    fine_loss += fine_loss_asphalt
    fine_loss += fine_loss_concrete
    fine_loss += fine_loss_paving_stones
    fine_loss += fine_loss_sett
    fine_loss += fine_loss_unpaved
    
    return fine_loss

def compute_fine_accuracy(coarse_probs, fine_output, fine_labels, masks, head):
    # Separate the fine outputs
    if head == 'regression':
        fine_output_asphalt = fine_output[:, 0:1].float()
        fine_output_concrete = fine_output[:, 1:2].float()
        fine_output_paving_stones = fine_output[:, 2:3].float()
        fine_output_sett = fine_output[:, 3:4].float()
        fine_output_unpaved = fine_output[:, 4:5].float()
    
    elif head == 'corn':
        fine_output_asphalt = fine_output[:, 0:3]
        fine_output_concrete = fine_output[:, 3:6]
        fine_output_paving_stones = fine_output[:, 6:9]
        fine_output_sett = fine_output[:, 9:11]
        fine_output_unpaved = fine_output[:, 11:13]
        
    else:
        fine_output_asphalt = fine_output[:, 0:4]
        fine_output_concrete = fine_output[:, 4:8]
        fine_output_paving_stones = fine_output[:, 8:12]
        fine_output_sett = fine_output[:, 12:15]
        fine_output_unpaved = fine_output[:, 15:18]
    
    # Extract the masks
    asphalt_mask, concrete_mask, paving_stones_mask, sett_mask, unpaved_mask = masks
    
 # Initialize prediction tensor
    predictions = torch.zeros_like(fine_labels)

    if asphalt_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            asphalt_preds = torch.argmax(fine_output_asphalt[asphalt_mask], dim=1)
        elif head == 'regression':
            asphalt_preds = fine_output_asphalt[asphalt_mask].round().long()
            print(asphalt_preds)
        elif head == 'corn':
            asphalt_preds = corn_label_from_logits(fine_output_asphalt[asphalt_mask]).long()
        predictions[asphalt_mask] = map_predictions_to_quality(asphalt_preds, "asphalt")

    if concrete_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            concrete_preds = torch.argmax(fine_output_concrete[concrete_mask], dim=1)
        elif head == 'regression':
            concrete_preds = fine_output_concrete[concrete_mask].round().long()
        elif head == 'corn':
            concrete_preds = corn_label_from_logits(fine_output_concrete[concrete_mask]).long()
        predictions[concrete_mask] = map_predictions_to_quality(concrete_preds, "concrete")

    if paving_stones_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            paving_stones_preds = torch.argmax(fine_output_paving_stones[paving_stones_mask], dim=1)
        elif head == 'regression':
            paving_stones_preds = fine_output_paving_stones[paving_stones_mask].round().long()
        elif head == 'corn':
            paving_stones_preds = corn_label_from_logits(fine_output_paving_stones[paving_stones_mask]).long()
        predictions[paving_stones_mask] = map_predictions_to_quality(paving_stones_preds, "paving_stones")

    if sett_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            sett_preds = torch.argmax(fine_output_sett[sett_mask], dim=1)
        elif head == 'regression':
            sett_preds = fine_output_sett[sett_mask].round().long()
        elif head == 'corn':
            sett_preds = corn_label_from_logits(fine_output_sett[sett_mask]).long()
        predictions[sett_mask] = map_predictions_to_quality(sett_preds, "sett")

    if unpaved_mask.sum().item() > 0:
        if head == 'clm' or head == 'classification':
            unpaved_preds = torch.argmax(fine_output_unpaved[unpaved_mask], dim=1)
        elif head == 'regression':
            unpaved_preds = fine_output_unpaved[unpaved_mask].round().long()
        elif head == 'corn':
            unpaved_preds = corn_label_from_logits(fine_output_unpaved[unpaved_mask]).long()
        predictions[unpaved_mask] = map_predictions_to_quality(unpaved_preds, "unpaved")


    # Calculate accuracy
    correct = (predictions == fine_labels).sum().item()
    
    correct_1_off = ((predictions == fine_labels) | 
               (predictions == fine_labels + 1) | 
               (predictions == fine_labels - 1)).sum().item()
    
    return correct, correct_1_off
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

#cam_extractor = SmoothGradCAMpp(model)

# Define the data loaders and transformations

num_fine_classes = 18
    #counting the coarse classes
    
num_classes = len(train_data.selected_classes)



#create one-hot encoded tensors with the fine class labels
y_train = helper.to_one_hot_tensor(train_data.targets, num_fine_classes)
y_valid = helper.to_one_hot_tensor(valid_data.targets, num_fine_classes)


#here we define the label tree, left is the fine class (e.g. asphalt-excellent) and right the coarse (e.g.asphalt)
parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])


y_c_train = torch.zeros(y_train.size(0), num_classes, dtype=torch.float32)
y_c_valid = torch.zeros(y_train.size(0), num_classes, dtype=torch.float32)


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
    scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
    
if lw_modifier:
    alpha = torch.tensor(0.98)
    beta = torch.tensor(0.02)
    loss_weights_modifier = helper.LossWeightsModifier(alpha, beta)

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
    if head == 'corn':
        pass
    else:
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
    fine_correct_one_off = 0
    
    fine_correct_asphalt = 0
    fine_correct_concrete = 0
    fine_correct_sett = 0
    fine_correct_paving_stones = 0
    fine_correct_unpaved = 0
    
    
    for batch_index, (inputs, fine_labels) in enumerate(trainloader):
              
                
        # if batch_index == 0:
        #     print("Batch Images:")
        #     images_grid = vutils.make_grid(inputs, nrow=8, padding=2, normalize=True)
        #     np_img = images_grid.numpy()
        #     np_img = np.transpose(np_img, (1, 2, 0))
            
        #     # Plot the images
        #     plt.figure(figsize=(16, 16))
        #     plt.imshow(np_img)
        #     plt.axis('off')  # Turn off axis
        #     plt.show()

        inputs, labels = inputs.to(device), fine_labels.to(device)
        
        optimizer.zero_grad()
        
        coarse_labels = parent[fine_labels]
        coarse_one_hot = helper.to_one_hot_tensor(coarse_labels, num_classes).to(device)
        #coarse_one_hot = coarse_one_hot.type(torch.LongTensor)
        #, dtype=torch.float32

        #we give the coarse true labels for the conditional prob weights matrix as input to the model
        model_inputs = (inputs, coarse_one_hot, config.get('hierarchy_method'))
        
        fine_labels_mapped = torch.tensor([helper.map_quality_to_continuous(label) for label in fine_labels], dtype=torch.long).to(device)
        
        masks = [
        (coarse_labels == 0),  # asphalt_mask
        (coarse_labels == 1),  # concrete_mask
        (coarse_labels == 2),  # paving_stones_mask
        (coarse_labels == 3),  # sett_mask
        (coarse_labels == 4)   # unpaved_mask
        ]

        
        if config.get('hierarchy_method') == 'use_ground_truth':
                        
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
            
            if head == 'clm':
                fine_loss_asphalt = fine_criterion(torch.log(fine_output_asphalt + epsilon), fine_labels_mapped_aspahlt)
                fine_loss_concrete = fine_criterion(torch.log(fine_output_concrete + epsilon), fine_labels_mapped_concrete)
                fine_loss_paving_stones = fine_criterion(torch.log(fine_output_paving_stones + epsilon), fine_labels_mapped_paving_stones)
                fine_loss_sett = fine_criterion(torch.log(fine_output_sett + epsilon), fine_labels_mapped_sett)
                fine_loss_unpaved = fine_criterion(torch.log(fine_output_unpaved + epsilon), fine_labels_mapped_unpaved)
            
                
            elif head == 'regression':
                
                fine_output_asphalt = fine_output_asphalt.flatten().float()
                fine_output_concrete = fine_output_concrete.flatten().float()
                fine_output_paving_stones = fine_output_paving_stones.flatten().float()
                fine_output_sett = fine_output_sett.flatten().float()
                fine_output_unpaved = fine_output_unpaved.flatten().float()
                
                fine_labels_mapped_aspahlt = fine_labels_mapped_aspahlt.float()
                fine_labels_mapped_concrete = fine_labels_mapped_concrete.float()
                fine_labels_mapped_paving_stones = fine_labels_mapped_paving_stones.float()
                fine_labels_mapped_sett = fine_labels_mapped_sett.float()
                fine_labels_mapped_unpaved = fine_labels_mapped_unpaved.float()
                
                fine_loss_asphalt = fine_criterion(fine_output_asphalt, fine_labels_mapped_aspahlt)
                fine_loss_concrete = fine_criterion(fine_output_concrete, fine_labels_mapped_concrete)
                fine_loss_paving_stones = fine_criterion(fine_output_paving_stones, fine_labels_mapped_paving_stones)
                fine_loss_sett = fine_criterion(fine_output_sett, fine_labels_mapped_sett)
                fine_loss_unpaved = fine_criterion(fine_output_unpaved, fine_labels_mapped_unpaved)
    
        
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
            
            # if head == 'clm':
            #     fine_predictions = torch.argmax(fine_output, dim=1)
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
            
        elif config.get('hierarchy_method') == 'use_condition_layer' or config.get('hierarchy_method') == 'top_coarse_prob':
            
            # if head == 'corn':
            #     coarse_output, fine_output_asphalt, fine_output_concrete, fine_output_paving_stones, fine_output_sett, fine_output_unpaved= model.forward(model_inputs)

            coarse_output, fine_output = model.forward(model_inputs)
            
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            
            if head == 'clm':
                fine_loss = fine_criterion(torch.log(fine_output + epsilon), fine_labels_mapped)
                            
            elif head == 'regression' or head == 'single':
                fine_output = fine_output.flatten().float()
                fine_labels_mapped = fine_labels_mapped.float()
                fine_loss = fine_criterion(fine_output, fine_labels_mapped)
                
            elif head == 'corn':
                fine_loss = model.fine_criterion(fine_output, fine_labels_mapped, 4)
                #fine_loss_asphalt = model.fine_criterion(fine_output_asphalt, fine_labels_mapped, 4)
                # fine_loss_concrete = model.fine_criterion(fine_output_concrete, fine_labels_mapped, 4)
                # fine_loss_paving_stones = model.fine_criterion(fine_output_paving_stones, fine_labels_mapped, 4)
                # fine_loss_sett = model.fine_criterion(fine_output_sett, fine_labels_mapped, 3)
                # fine_loss_unpaved = model.fine_criterion(fine_output_unpaved, fine_labels_mapped, 3)
                # fine_loss = fine_loss_asphalt + fine_loss_concrete + fine_loss_paving_stones + fine_loss_sett + fine_loss_unpaved

            loss = coarse_loss + fine_loss  #weighted loss functions for different levels
            
            loss.backward()
            
            # print("Gradients for each layer:")
            # for name, param in model.classifier.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name} gradients: {param.grad}")
            
            # print("Gradients for each layer:")
            # for name, param in model.classifier_asphalt.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name} gradients: {param.grad}")
                    
            # for name, param in model.classifier_concrete.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name} gradients: {param.grad}")
            
            optimizer.step()
            
            if config.get('hierarchy_method') == 'use_condition_layer':
            #plot_grad_flow(model.named_parameters())
                #print(f'Fine output tensor: {fine_output}')
                #print("Gradients:", model.coarse_condition.weight.grad)
                if config == train_config.C_CNN_CLM:
                    model.coarse_condition.weight.data = model.constraint(model.coarse_condition.weight.data)
                #print(f'CPWM after optimizer step: {model.coarse_condition.weight.data}')
            
            running_loss += loss.item()
            
            coarse_loss_total += coarse_loss.item()
            fine_loss_total += fine_loss.item()
        
            coarse_output = model.get_class_probabilies(coarse_output)
            coarse_predictions = torch.argmax(coarse_output, dim=1)
            coarse_correct += (coarse_predictions == coarse_labels).sum().item()
            
            if head == 'clm':
                fine_predictions = torch.argmax(fine_output, dim=1)
            elif head == 'regression' or head == 'single':
                fine_predictions = fine_output.round()
            elif head == 'corn':
                fine_predictions_asphalt = corn_label_from_logits(fine_output_asphalt).float()
            else:
                probs = model.get_class_probabilies(fine_output)
                predictions = torch.argmax(probs, dim=1)
            fine_correct += (fine_predictions == fine_labels_mapped).sum().item()
            
            
            # if batch_index == 0:
            #     break
            
            
        elif config.get('hierarchy_method') == 'b_cnn':
            
            coarse_output, fine_output = model.forward(model_inputs)
            
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            
            if head == 'clm':
                fine_loss = fine_criterion(torch.log(fine_output + epsilon), fine_labels_mapped)
                
            elif head == 'regression' or head == 'single':
                fine_output = fine_output.flatten().float()
                fine_labels_mapped = fine_labels_mapped.float()
                fine_loss = fine_criterion(fine_output, fine_labels_mapped)
                
            elif head == 'corn':
                corn_loss = fine_criterion(fine_output, fine_labels_mapped, num_fine_classes)
                
            if lw_modifier:
                loss = alpha * coarse_loss + beta * fine_loss
            else:
                loss = coarse_loss + fine_loss
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() 
            
            coarse_loss_total += coarse_loss.item()
            fine_loss_total += fine_loss.item()
            
            coarse_probs = model.get_class_probabilies(coarse_output)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            coarse_correct += (coarse_predictions == coarse_labels).sum().item()
            
            if head == 'clm':
                fine_predictions = torch.argmax(fine_output, dim=1)
                fine_correct += (fine_predictions == fine_labels).sum().item()
            elif head == 'regression':
                fine_predictions = fine_output.round()
                fine_correct += (fine_predictions == fine_labels).sum().item()
            elif head == 'single':
                fine_predictions = fine_output.round()
                fine_correct += (fine_predictions == fine_labels_mapped).sum().item()
            elif head == 'corn':
                fine_predictions = corn_label_from_logits(fine_output).float() #TODO:mapped or not?
                fine_correct += (fine_predictions == fine_labels_mapped).sum().item()
            else:
                probs = model.get_class_probabilies(fine_output)
                predictions = torch.argmax(probs, dim=1)
                fine_correct += (fine_predictions == fine_labels_mapped).sum().item() #TODO:mapped or not?

            


            # if batch_index == 0:
            #     break
         
        else:
            
            # if head == 'corn':
            #     coarse_output, fine_output_asphalt, fine_output_concrete, fine_output_paving_stones, fine_output_sett, fine_output_unpaved= model.forward(model_inputs)

            # else:
            coarse_output, fine_output = model.forward(model_inputs)
            
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            
            if head == 'clm':
                fine_loss = compute_fine_losses(fine_output, fine_labels_mapped, masks, head)
                #fine_loss = fine_criterion(torch.log(fine_output + epsilon), fine_labels)
                
            elif head == 'corn':
                fine_loss = compute_fine_losses(fine_output, fine_labels_mapped, masks, head)
                            
            elif head == 'regression' or head == 'single':
                fine_loss = compute_fine_losses(fine_output, fine_labels_mapped, masks, head)
                # fine_output = fine_output.flatten().float()
                # fine_labels_mapped = fine_labels_mapped.float()
                # fine_loss = fine_criterion(fine_output, fine_labels_mapped)
                
            if lw_modifier:
                loss = alpha * coarse_loss + beta * fine_loss
            else:
                loss = coarse_loss + fine_loss  #weighted loss functions for different levels
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            coarse_loss_total += coarse_loss.item()
            fine_loss_total += fine_loss.item()
        
            coarse_probs = model.get_class_probabilies(coarse_output)
            coarse_predictions = torch.argmax(coarse_probs, dim=1)
            coarse_correct += (coarse_predictions == coarse_labels).sum().item()
            
            if head == 'clm':
                fine_correct_item, fine_correct_one_off_item = compute_fine_accuracy(coarse_probs, fine_output, fine_labels, masks. head)
                fine_correct += fine_correct_item
                fine_correct_one_off += fine_correct_one_off_item
                #fine_correct += (fine_accuracy * fine_labels.size(0))
                #fine_predictions = torch.argmax(fine_output, dim=1)
                #fine_correct += (fine_predictions == fine_labels).sum().item()
                #fine_correct_one_off += accuracy_off1(fine_predictions, fine_labels, num_classes=num_fine_classes) * inputs.size(0)
            elif head == 'regression' or head == 'single':
                fine_correct_item, fine_correct_one_off_item = compute_fine_accuracy(coarse_probs, fine_output, fine_labels, masks, head)
                fine_correct += fine_correct_item
                fine_correct_one_off += fine_correct_one_off_item
                #fine_predictions = fine_output.round()
                #fine_correct += (fine_predictions == fine_labels_mapped).sum().item()
            elif head == 'corn':
                fine_correct_item, fine_correct_one_off_item = compute_fine_accuracy(coarse_probs, fine_output, fine_labels, masks, head)
                fine_correct += fine_correct_item
                fine_correct_one_off += fine_correct_one_off_item
                #fine_predictions_asphalt = corn_label_from_logits(fine_output_asphalt).float()
            else:
                probs = model.get_class_probabilies(fine_output)
                fine_correct_item, fine_correct_one_off_item = compute_fine_accuracy(coarse_probs, probs, fine_labels, masks, head)
                fine_correct += fine_correct_item
                fine_correct_one_off += fine_correct_one_off_item
                #predictions = torch.argmax(probs, dim=1)
                
            if batch_index == 0:
                break
                
               
    # #learning rate step        
    # before_lr = optimizer.param_groups[0]["lr"]
    # scheduler.step()
    # after_lr = optimizer.param_groups[0]["lr"]
    
    # #loss weights step
    # alpha, beta = loss_weights_modifier.on_epoch_end(epoch)
    
    # epoch_loss = running_loss /  len(inputs) * (batch_index + 1) 
    # epoch_coarse_accuracy = 100 * coarse_correct / (len(inputs) * (batch_index + 1))
    # epoch_fine_accuracy = 100 * fine_correct / (len(inputs) * (batch_index + 1))
    
    epoch_loss = running_loss /  len(trainloader.sampler)
    epoch_coarse_accuracy = 100 * coarse_correct / len(trainloader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(trainloader.sampler)
    
    epoch_fine_accuracy_one_off = 100 * fine_correct_one_off / len(trainloader.sampler)
    
    coarse_epoch_loss = coarse_loss_total / len(trainloader.sampler)
    fine_epoch_loss = fine_loss_total / len(trainloader.sampler)
    
    print(coarse_epoch_loss)
    print(fine_epoch_loss)


    # Validation
    model.eval()
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    if head == 'corn':
        fine_criterion = model.fine_criterion
    else:
        fine_criterion = model.fine_criterion(reduction="sum")
    
    
    val_running_loss = 0.0
    val_coarse_loss_total = 0.0
    val_fine_loss_total = 0.0
    
    val_coarse_correct = 0
    val_fine_correct = 0
    val_fine_correct_one_off = 0
    
    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(validloader):
            
            inputs, fine_labels = inputs.to(device), fine_labels.to(device)
            coarse_labels = parent[fine_labels]
            coarse_one_hot = helper.to_one_hot_tensor(coarse_labels, num_classes).to(device)
            
            #model_inputs = inputs
            model_inputs = (inputs, coarse_one_hot, config.get('hierarchy_method'))   
            
            fine_labels_mapped = torch.tensor([helper.map_quality_to_continuous(label) for label in fine_labels], dtype=torch.long).to(device)
            coarse_output, fine_output = model.forward(model_inputs)
            
            # cams = cam_extractor(fine_output.argmax(dim=1).cpu().numpy(), fine_output)
            
            # for i in range(inputs.size(0)):
            #     # Convert tensor to PIL image
            #     input_image = to_pil_image(inputs[i].cpu())
            #     cam_image = to_pil_image(cams[i].cpu())

            #     # Overlay CAM on the input image
            #     overlay_image = vutils.make_grid([inputs[i].cpu(), cams[i].cpu()], nrow=1)

            #     # Save the image
            #     plt.imshow(overlay_image.permute(1, 2, 0).numpy())
            #     plt.axis('off')
            #     plt.savefig(f"cam_{epoch}_{batch_index}_{i}.png")
            #     plt.close()
            
            coarse_loss = coarse_criterion(coarse_output, coarse_labels)
            val_coarse_output = model.get_class_probabilies(coarse_output)
            val_coarse_predictions = torch.argmax(val_coarse_output, dim=1)
            val_coarse_correct += (val_coarse_predictions == coarse_labels).sum().item()
            
            val_masks = [
            (val_coarse_predictions == 0),  # asphalt_mask
            (val_coarse_predictions == 1),  # concrete_mask
            (val_coarse_predictions == 2),  # paving_stones_mask
            (val_coarse_predictions == 3),  # sett_mask
            (val_coarse_predictions == 4)   # unpaved_mask
            ]
            
            if head == 'clm':
                fine_loss = compute_fine_losses(fine_output, fine_labels_mapped, val_masks, head)
                #fine_loss = fine_criterion(torch.log(fine_output + epsilon), fine_labels)
            elif head == 'regression' or head == 'single':
                fine_loss = compute_fine_losses(fine_output, fine_labels_mapped, val_masks, head)
            elif head == 'corn':
                fine_loss = compute_fine_losses(fine_output, fine_labels_mapped, val_masks, head)
                        
            if lw_modifier:
                loss = alpha * coarse_loss + beta * fine_loss
            else:
                loss = coarse_loss + fine_loss
        
            val_running_loss += loss.item()
            
            val_coarse_loss_total += coarse_loss.item()
            val_fine_loss_total += fine_loss.item()
            

            if head == 'classification':
                fine_output = model.get_class_probabilies(fine_output)
    
            val_fine_correct_item, val_fine_correct_one_off_item = compute_fine_accuracy(coarse_probs, fine_output, fine_labels, val_masks, head)
            val_fine_correct += val_fine_correct_item
            val_fine_correct_one_off += val_fine_correct_one_off_item

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
    
    val_epoch_loss = val_running_loss /  len(validloader.sampler)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(validloader.sampler)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(validloader.sampler)
    
    val_epoch_fine_accuracy_one_off = 100 * val_fine_correct_one_off / len(validloader.sampler)
    
    
    val_coarse_epoch_loss = val_coarse_loss_total / len(validloader.sampler)
    val_fine_epoch_loss = val_fine_loss_total / len(validloader.sampler)

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
                "train/coarse/loss": coarse_epoch_loss,
                "train/fine/loss": fine_epoch_loss,
                "train/accuracy/coarse": epoch_coarse_accuracy,
                "train/accuracy/fine": epoch_fine_accuracy, 
                "train/accuracy/fine_1_off": epoch_fine_accuracy_one_off,
                "eval/loss": val_epoch_loss,
                "eval/coarse/loss": val_coarse_epoch_loss,
                "eval/fine/loss": val_fine_epoch_loss,
                "eval/accuracy/coarse": val_epoch_coarse_accuracy,
                "eval/accuracy/fine": val_epoch_fine_accuracy,
                "eval/accuracy/fine_1_off": val_epoch_fine_accuracy_one_off,
                "trainable_params": trainable_params,
                "learning_rate": scheduler.get_last_lr()[0],
                "hierarchy_method": config.get("hierarchy_method"),
                "head": config.get("head"),
            }
        )
    
    print(f"""
        Epoch: {epoch+1}:,
        
        Train loss: {epoch_loss:.3f},
        
        Coarse train loss: {coarse_epoch_loss:.3f},
        Fine train loss: {fine_epoch_loss:.3f}, 
        
        Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
        Train fine accuracy: {epoch_fine_accuracy:.3f}%,
        Train fine 1-off accuracy: {epoch_fine_accuracy_one_off:.3f}%,
        
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
        Validation fine accuracy: {val_epoch_fine_accuracy:.3f}%, 
        Validation fine 1-off accuracy: {val_epoch_fine_accuracy_one_off:.3f}%
        
        Learning_rate: {scheduler.get_last_lr()[0]}
                
        """)

    
    if lw_modifier:
        alpha, beta = loss_weights_modifier.on_epoch_end(epoch)
    
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
        
    
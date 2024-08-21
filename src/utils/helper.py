import sys


sys.path.append('.')

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import optim
from src import constants as const
from src.architecture import efficientnet, vgg16, vgg16_C_CNN_pretrained, vgg16_B_CNN_pretrained, vgg16_GH_CNN_pretrained, vgg16_HierarchyNet_pretrained
import json
import argparse
from matplotlib.lines import Line2D
from torch.utils.data import Dataset
import os
import tensorflow as tf
from coral_pytorch.dataset import corn_label_from_logits
import torch.nn as nn
import torch.nn.functional as F
import wandb


def string_to_object(string):

    string_dict = {
        const.VGG16: vgg16.CustomVGG16,
        const.VGG16REGRESSION: vgg16.CustomVGG16,
        const.EFFICIENTNET: efficientnet.CustomEfficientNetV2SLogsoftmax,
        const.EFFNET_LINEAR: efficientnet.CustomEfficientNetV2SLinear,
        const.OPTI_ADAM: optim.Adam,
        const.CCNNCLMPRE: vgg16_C_CNN_pretrained.Condition_CNN_CLM_PRE,
        const.BCNN_PRE: vgg16_B_CNN_pretrained.VGG16_B_CNN_PRE,
        const.HNET_PRE: vgg16_HierarchyNet_pretrained.HierarchyNet_Pre,
        const.GHCNN_PRE: vgg16_GH_CNN_pretrained.GH_CNN_PRE,

    }

    return string_dict.get(string)

def format_sweep_config(config):
    p = {
        key: value
        for key, value in config.items()
        if key in ["name", "method", "metric"]
    }

    sweep_params = {
        **{
            key: {"value": value}
            for key, value in config.items()
            if key
            not in [
                "transform",
                "augment",
                "search_params",
                "name",
                "method",
                "metric",
                "wandb_mode",
                "project",
                "sweep_counts",
                "wandb_on"
            ]
        },
        "transform": {
            "parameters": {
                key: {"value": value} for key, value in config.get("transform").items()
            }
        },
        "augment": {
            "parameters": {
                key: {"value": value} for key, value in config.get("augment").items()
            }
        },
        **config.get("search_params"),
    }

    return {
        **p,
        "parameters": sweep_params,
    }

def format_config(config):
    return {
                key: value
                for key, value in config.items()
                if key not in ["wandb_mode", "wandb_on", "project", "name"]
            }

def dict_type(arg):
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("The argument is no valid dict type.")


# auxiliary visualization function

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def multi_imshow(images, labels):

    fig, axes = plt.subplots(figsize=(20,4), ncols=8)

    for ii in range(8):
        ax = axes[ii]
        label = labels[ii]
        ax.set_title(f'Label: {label}')
        imshow(images[ii], ax=ax, normalize=True)
        
def make_hook(key, feature_maps):
    def hook(model, input, output):
        feature_maps[key] = output.detach()
    return hook


def to_one_hot_tensor(labels, num_classes):
    labels = torch.tensor(labels)
    one_hot = torch.zeros(labels.size(0), num_classes, dtype=torch.float32)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


class NonNegUnitNorm:
    '''Enforces all weight elements to be non-negative and each column/row to be unit norm'''
    def __init__(self, axis=1):
        self.axis = axis
    
    def __call__(self, w):
        w = w * (w >= 0).float()  # Set negative weights to zero
        norm = torch.sqrt(torch.sum(w ** 2, dim=self.axis, keepdim=True))
        w = w / (norm + 1e-8)  # Normalize each column/row to unit norm
        return w

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
        if epoch >= 3:
            self.alpha = torch.tensor(0.6)
            self.beta = torch.tensor(0.4)
        if epoch >= 6:
            self.alpha = torch.tensor(0.2)
            self.beta = torch.tensor(0.8)
        if epoch >= 9:
            self.alpha = torch.tensor(0.0)
            self.beta = torch.tensor(1.0)
            
        return self.alpha, self.beta
    
#this helps us adopt a regression on the second level for multi-label models  
def map_flatten_to_ordinal(quality_label):
    quality_mapping = {
        0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 0.0, 5: 1.0,
        6: 2.0, 7: 3.0, 8: 0.0, 9: 1.0, 10: 2.0, 11: 3.0,
        12: 0.0, 13: 1.0, 14: 2.0, 15: 0.0, 16: 1.0, 17: 2.0
    }
    return quality_mapping[quality_label.item()]

def map_ordinal_to_flatten(label, type):
    if type == 'asphalt':
        return label  
    elif type == 'concrete':
        return label + 4
    elif type == 'paving_stones':
        return label + 8
    elif type == 'sett':
        return label + 12
    elif type == 'unpaved':
        return label + 15
    else:
        raise ValueError("Unknown type")


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    
class Custom_Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = [dataset.targets[i] for i in indices]
    
    def __getitem__(self, idx):
        image, _ = self.dataset[self.indices[idx]]
        target = self.targets[idx]
        return image, target
    
    def __len__(self):
        return len(self.indices)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#for both tensorflow and pytorch   
def fix_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    
def get_parameters_by_layer(model, layer_name):
    """
    Get the parameters of a specific layer by name.
    """
    params = []
    for name, param in model.named_parameters():
        if layer_name in name:
            params.append(param)
    return params


def save_gradient_plots(epoch, gradients, first_moments, second_moments, save_dir="multi_label\gradients"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(gradients, label="Gradients")
    plt.title("Gradients of Last CLM Layer")
    plt.xlabel("Batch")
    plt.ylabel("Gradient Norm")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(first_moments, label="First Moment (m_t)")
    plt.title("First Moment (m_t) of Last CLM Layer")
    plt.xlabel("Batch")
    plt.ylabel("First Moment Norm")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(second_moments, label="Second Moment (v_t)")
    plt.title("Second Moment (v_t) of Last CLM Layer")
    plt.xlabel("Batch")
    plt.ylabel("Second Moment Norm")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}.png"))
    plt.close()
    
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def load_images_and_labels(base_path, img_shape, custom_label_order):
    images = []
    labels = []
    for label_folder in os.listdir(base_path):
        label_folder_path = os.path.join(base_path, label_folder)
        if os.path.isdir(label_folder_path):
            for img_file in os.listdir(label_folder_path):
                img_path = os.path.join(label_folder_path, img_file)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_shape)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label_folder)  # Use the folder name as the label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    # Create a mapping based on custom_label_order
    label_to_index = {label: idx for idx, label in enumerate(custom_label_order)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    # Map labels to the custom order indices
    y = np.array([label_to_index[label] for label in labels])
    
    return np.array(images), y, label_to_index, index_to_label

class CustomDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target
    
def map_predictions_to_quality(predictions, surface_type):
    quality_mapping = {
        "asphalt": [0, 1, 2, 3, 4, 5, 6, 7],  # Modify as needed
        "concrete": [4, 5, 6, 7, 8, 9],
        "paving_stones": [8, 9, 10, 11, 12, 13, 14, 15],
        "sett": [12, 13, 14, 15, 16],
        "unpaved": [15, 16, 17, 18,]
    }
    return torch.tensor([quality_mapping[surface_type][pred] for pred in predictions], dtype=torch.long)


def compute_fine_losses(model, fine_criterion, fine_output, fine_labels, device, coarse_filter, hierarchy_method, head):
    fine_loss = 0.0
    
    if hierarchy_method == 'use_ground_truth':
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
        
        fine_labels_mapped = torch.tensor([map_flatten_to_ordinal(label) for label in fine_labels], dtype=torch.long).to(device)
        
        masks = [
        (coarse_filter == 0),
        (coarse_filter == 1), 
        (coarse_filter == 2),  
        (coarse_filter == 3), 
        (coarse_filter == 4)  
        ]
        
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
            fine_loss_asphalt = fine_criterion(torch.log(fine_output_asphalt[asphalt_mask] + 1e-9), fine_labels_mapped_asphalt) #TODO: check if that works 
            fine_loss_concrete = fine_criterion(torch.log(fine_output_concrete[concrete_mask] + 1e-9), fine_labels_mapped_concrete)
            fine_loss_paving_stones = fine_criterion(torch.log(fine_output_paving_stones[paving_stones_mask] + 1e-9), fine_labels_mapped_paving_stones)
            fine_loss_sett = fine_criterion(torch.log(fine_output_sett[sett_mask][three_mask_sett] + 1e-9), fine_labels_mapped_sett)
            fine_loss_unpaved = fine_criterion(torch.log(fine_output_unpaved[unpaved_mask][three_mask_unpaved] + 1e-9), fine_labels_mapped_unpaved)
        elif head == 'corn':
            fine_loss_asphalt = fine_criterion(fine_output_asphalt[asphalt_mask], fine_labels_mapped_asphalt, 4)
            fine_loss_concrete = fine_criterion(fine_output_concrete[concrete_mask], fine_labels_mapped_concrete, 4)
            fine_loss_paving_stones = fine_criterion(fine_output_paving_stones[paving_stones_mask], fine_labels_mapped_paving_stones, 4) #TODO: hard coding num_classes vermeiden
            fine_loss_sett = fine_criterion(fine_output_sett[sett_mask][three_mask_sett], fine_labels_mapped_sett, 3)
            fine_loss_unpaved = fine_criterion(fine_output_unpaved[unpaved_mask][three_mask_unpaved], fine_labels_mapped_unpaved, 3)
        elif head == 'regression':
            fine_loss_asphalt = fine_criterion(fine_output_asphalt[asphalt_mask].flatten(), fine_labels_mapped_asphalt.float())
            fine_loss_concrete = fine_criterion(fine_output_concrete[concrete_mask].flatten(), fine_labels_mapped_concrete.float())
            fine_loss_paving_stones = fine_criterion(fine_output_paving_stones[paving_stones_mask].flatten(), fine_labels_mapped_paving_stones.float())
            fine_loss_sett = fine_criterion(fine_output_sett[sett_mask][three_mask_sett].flatten(), fine_labels_mapped_sett.float())
            fine_loss_unpaved = fine_criterion(fine_output_unpaved[unpaved_mask][three_mask_unpaved].flatten(), fine_labels_mapped_unpaved.float())
                
        fine_loss_asphalt = torch.nan_to_num(fine_loss_asphalt, nan=0.0)
        fine_loss_concrete = torch.nan_to_num(fine_loss_concrete, nan=0.0)
        fine_loss_paving_stones = torch.nan_to_num(fine_loss_paving_stones, nan=0.0)
        fine_loss_sett = torch.nan_to_num(fine_loss_sett, nan=0.0)
        fine_loss_unpaved = torch.nan_to_num(fine_loss_unpaved, nan=0.0)
        
        fine_loss += fine_loss_asphalt
        fine_loss += fine_loss_concrete
        fine_loss += fine_loss_paving_stones
        fine_loss += fine_loss_sett
        fine_loss += fine_loss_unpaved

    elif hierarchy_method == 'use_model_structure':
        
        if head == 'classification':
            fine_loss = fine_criterion(fine_output, fine_labels)
        
        elif head == 'clm':
            fine_loss = fine_criterion(torch.log(fine_output + 1e-9), fine_labels_mapped) #TODO wie kann das berechnet werden?
                            
        # elif head == 'regression' or head == 'single':
        #     fine_output = fine_output.flatten().float()
        #     fine_labels_mapped = fine_labels_mapped.float()
        #     fine_loss = fine_criterion(fine_output, fine_labels_mapped)
            
        # elif head == 'corn':
        #     fine_loss = model.fine_criterion(fine_output, fine_labels_mapped, 4)
    
    return fine_loss



def compute_fine_metrics(fine_output, fine_labels, coarse_filter, hierarchy_method, head):
    
    if hierarchy_method == 'use_ground_truth':
        masks = [
            (coarse_filter == 0),
            (coarse_filter == 1), 
            (coarse_filter == 2),  
            (coarse_filter == 3), 
            (coarse_filter == 4)  
            ]
        
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
        
        # Initialize prediction tensor and metrics
        predictions = torch.zeros_like(fine_labels)
        total_mse = 0
        total_mae = 0

        def compute_metrics(output, labels, mask, category):
            nonlocal total_mse, total_mae
            
            if mask.sum().item() > 0:
                if head == 'clm' or head == 'classification':
                    preds = torch.argmax(output[mask], dim=1)
                elif head == 'regression':
                    preds = output[mask].round().long()
                elif head == 'corn':
                    preds = corn_label_from_logits(output[mask]).long()
                    
                predictions[mask] = map_predictions_to_quality(preds, category)
                
                # Calculate MSE and MAE
                if head == 'regression':
                    mse = F.mse_loss(output[mask], labels[mask].float(), reduction='sum').item()
                    mae = F.l1_loss(output[mask], labels[mask].float(), reduction='sum').item()
                else:
                    # Convert predicted classes to their corresponding numerical values
                    mse = F.mse_loss(preds.float(), labels[mask].float(), reduction='sum').item()
                    mae = F.l1_loss(preds.float(), labels[mask].float(), reduction='sum').item()

                total_mse += mse
                total_mae += mae

        compute_metrics(fine_output_asphalt, fine_labels, asphalt_mask, "asphalt")
        compute_metrics(fine_output_concrete, fine_labels, concrete_mask, "concrete")
        compute_metrics(fine_output_paving_stones, fine_labels, paving_stones_mask, "paving_stones")
        compute_metrics(fine_output_sett, fine_labels, sett_mask, "sett")
        compute_metrics(fine_output_unpaved, fine_labels, unpaved_mask, "unpaved")
        
    else:
        if head == 'clm' or head == 'classification':
            predictions = torch.argmax(fine_output, dim=1)
            total_mse = F.mse_loss(predictions.float(), fine_labels.float(), reduction='sum').item()
            total_mae = F.l1_loss(predictions.float(), fine_labels.float(), reduction='sum').item()
        elif head == 'regression': #TODO: macht das überhaupt Sinn?
            predictions = fine_output.round().long()
            total_mse = F.mse_loss(fine_output, fine_labels.float(), reduction='sum').item()
            total_mae = F.l1_loss(fine_output, fine_labels.float(), reduction='sum').item()

    # Calculate accuracy
    correct = (predictions == fine_labels).sum().item()
    
    correct_1_off = ((predictions == fine_labels) | 
               (predictions == fine_labels + 1) | 
               (predictions == fine_labels - 1)).sum().item()

    # Return the sum of MSE and MAE
    return correct, correct_1_off, total_mse, total_mae


def compute_all_metrics(outputs, labels, head, model):
    
    if head == 'regression': 
        predictions = outputs.round()
    elif head == 'clm':
        predictions = torch.argmax(outputs, dim=1)
    elif head == 'corn':
        predictions = corn_label_from_logits(outputs).long()
    else:  #classification
        probs = model.get_class_probabilities(outputs)
        predictions = torch.argmax(probs, dim=1)

    # Calculate accuracy
    correct = (predictions == labels).sum().item()

    # Calculate 1-off accuracy
    correct_1_off = ((predictions == labels) |
                     (predictions == labels + 1) |
                     (predictions == labels - 1)).sum().item()

    # Calculate MSE and MAE
    if head == 'regression':
        total_mse = F.mse_loss(outputs, labels.float(), reduction='sum').item()
        total_mae = F.l1_loss(outputs, labels.float(), reduction='sum').item()
    else:
        # For classification and other head types, compare with predicted classes
        total_mse = F.mse_loss(predictions.float(), labels.float(), reduction='sum').item()
        total_mae = F.l1_loss(predictions.float(), labels.float(), reduction='sum').item()

    return correct, correct_1_off, total_mse, total_mae


def compute_and_log_CC_metrics(df, trainloaders, validloaders, wandb_on):
    
    def calculate_accuracy(correct_sum, total_samples):
        return 100 * correct_sum / total_samples
        
    epochs = df['epoch'].unique()
    
    for epoch in epochs:
        epoch_df = df[df['epoch'] == epoch]
        level = epoch_df['level'].iloc[0]  # Use `.iloc[0]` to get the first value

        average_metrics = epoch_df.drop(columns=['epoch', 'level']).mean()

        if level == 'surface':
            coarse_epoch_loss = average_metrics['train_loss'] / sum(len(loader.sampler) for loader in trainloaders)
            val_coarse_epoch_loss = average_metrics['val_loss'] / sum(len(loader.sampler) for loader in validloaders)
            
            epoch_coarse_accuracy = 100 * average_metrics['train_correct'] / sum(len(loader.sampler) for loader in trainloaders)
            val_epoch_coarse_accuracy = 100 * average_metrics['val_correct'] / sum(len(loader.sampler) for loader in validloaders)
            
            if wandb_on: 
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/coarse/loss": coarse_epoch_loss,
                        "train/accuracy/coarse": epoch_coarse_accuracy,
                        "eval/coarse/loss": val_coarse_epoch_loss,
                        "eval/accuracy/coarse": val_epoch_coarse_accuracy,
                    }
                )
            
        else:
            fine_epoch_loss = average_metrics['train_loss'] / sum(len(loader.sampler) for loader in trainloaders)
            val_fine_epoch_loss = average_metrics['val_loss'] / sum(len(loader.sampler) for loader in validloaders)
            
            surface_types = ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']

            # Accumulate correct predictions and total sample counts
            total_train_correct = 0
            total_val_correct = 0
            total_train_samples = sum(len(loader.sampler) for loader in trainloaders)
            total_val_samples = sum(len(loader.sampler) for loader in validloaders)

            for surface in surface_types:
                train_correct_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'train_correct'].sum()
                val_correct_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'val_correct'].sum()
                
                train_correct_one_off_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'train_correct_one_off'].sum()
                val_correct_one_off_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'val_correct_one_off'].sum()
                
                total_train_correct += train_correct_sum
                total_val_correct += val_correct_sum

            # Calculate overall accuracy
            epoch_fine_accuracy = calculate_accuracy(total_train_correct, total_train_samples)
            epoch_fine_accuracy_one_off = calculate_accuracy(train_correct_one_off_sum, total_train_samples)
            val_epoch_fine_accuracy = calculate_accuracy(total_val_correct, total_val_samples)
            val_epoch_fine_accuracy_one_off = calculate_accuracy(val_correct_one_off_sum, total_val_samples)

            # Logging the results
            if wandb_on: 
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/fine/loss": fine_epoch_loss,
                        "train/accuracy/fine": epoch_fine_accuracy, 
                        "train/accuracy/fine_1_off": epoch_fine_accuracy_one_off,
                        "eval/fine/loss": val_fine_epoch_loss,
                        "eval/accuracy/fine": val_epoch_fine_accuracy,
                        "eval/accuracy/fine_1_off": val_epoch_fine_accuracy_one_off,
                    }
                )



# def compute_all_metrics_CC(outputs, labels, head, model, type):
    
#     if head == 'regression': 
#         predictions = outputs.round()
#     elif head == 'clm':
#         predictions = torch.argmax(outputs, dim=1)
#     elif head == 'corn':
#         predictions = corn_label_from_logits(outputs).long()
#     else:  #classification
#         probs = model.get_class_probabilities(outputs)
#         predictions = torch.argmax(probs, dim=1)
        
#     predictions_mapped = map_predictions_to_quality(predictions, type)
#     labels_mapped = map_ordinal_to_flatten(labels, type)

#     # Calculate accuracy
#     correct = (predictions_mapped == labels_mapped).sum().item()

#     # Calculate 1-off accuracy
#     correct_1_off = ((predictions == labels) |
#                      (predictions == labels + 1) |
#                      (predictions == labels - 1)).sum().item()

#     # Calculate MSE and MAE
#     if head == 'regression':
#         total_mse = F.mse_loss(outputs, labels.float(), reduction='sum').item()
#         total_mae = F.l1_loss(outputs, labels.float(), reduction='sum').item()
#     else:
#         # For classification and other head types, compare with predicted classes
#         total_mse = F.mse_loss(predictions.float(), labels.float(), reduction='sum').item()
#         total_mae = F.l1_loss(predictions.float(), labels.float(), reduction='sum').item()

#     return correct, correct_1_off, total_mse, total_mae



# def compute_fine_metrics(coarse_probs, fine_output, fine_labels, masks, head):
#     # Separate the fine outputs
#     if head == 'regression':
#         fine_output_asphalt = fine_output[:, 0:1].float()
#         fine_output_concrete = fine_output[:, 1:2].float()
#         fine_output_paving_stones = fine_output[:, 2:3].float()
#         fine_output_sett = fine_output[:, 3:4].float()
#         fine_output_unpaved = fine_output[:, 4:5].float()
    
#     elif head == 'corn':
#         fine_output_asphalt = fine_output[:, 0:3]
#         fine_output_concrete = fine_output[:, 3:6]
#         fine_output_paving_stones = fine_output[:, 6:9]
#         fine_output_sett = fine_output[:, 9:11]
#         fine_output_unpaved = fine_output[:, 11:13]
        
#     else:
#         fine_output_asphalt = fine_output[:, 0:4]
#         fine_output_concrete = fine_output[:, 4:8]
#         fine_output_paving_stones = fine_output[:, 8:12]
#         fine_output_sett = fine_output[:, 12:15]
#         fine_output_unpaved = fine_output[:, 15:18]
    
#     # Extract the masks
#     asphalt_mask, concrete_mask, paving_stones_mask, sett_mask, unpaved_mask = masks
    
#  # Initialize prediction tensor
#     predictions = torch.zeros_like(fine_labels)

#     if asphalt_mask.sum().item() > 0:
#         if head == 'clm' or head == 'classification':
#             asphalt_preds = torch.argmax(fine_output_asphalt[asphalt_mask], dim=1)
#         elif head == 'regression':
#             asphalt_preds = fine_output_asphalt[asphalt_mask].round().long()
#             print(asphalt_preds)
#         elif head == 'corn':
#             asphalt_preds = corn_label_from_logits(fine_output_asphalt[asphalt_mask]).long()
#         predictions[asphalt_mask] = map_predictions_to_quality(asphalt_preds, "asphalt")

#     if concrete_mask.sum().item() > 0:
#         if head == 'clm' or head == 'classification':
#             concrete_preds = torch.argmax(fine_output_concrete[concrete_mask], dim=1)
#         elif head == 'regression':
#             concrete_preds = fine_output_concrete[concrete_mask].round().long()
#         elif head == 'corn':
#             concrete_preds = corn_label_from_logits(fine_output_concrete[concrete_mask]).long()
#         predictions[concrete_mask] = map_predictions_to_quality(concrete_preds, "concrete")

#     if paving_stones_mask.sum().item() > 0:
#         if head == 'clm' or head == 'classification':
#             paving_stones_preds = torch.argmax(fine_output_paving_stones[paving_stones_mask], dim=1)
#         elif head == 'regression':
#             paving_stones_preds = fine_output_paving_stones[paving_stones_mask].round().long()
#         elif head == 'corn':
#             paving_stones_preds = corn_label_from_logits(fine_output_paving_stones[paving_stones_mask]).long()
#         predictions[paving_stones_mask] = map_predictions_to_quality(paving_stones_preds, "paving_stones")

#     if sett_mask.sum().item() > 0:
#         if head == 'clm' or head == 'classification':
#             sett_preds = torch.argmax(fine_output_sett[sett_mask], dim=1)
#         elif head == 'regression':
#             sett_preds = fine_output_sett[sett_mask].round().long()
#         elif head == 'corn':
#             sett_preds = corn_label_from_logits(fine_output_sett[sett_mask]).long()
#         predictions[sett_mask] = map_predictions_to_quality(sett_preds, "sett")

#     if unpaved_mask.sum().item() > 0:
#         if head == 'clm' or head == 'classification':
#             unpaved_preds = torch.argmax(fine_output_unpaved[unpaved_mask], dim=1)
#         elif head == 'regression':
#             unpaved_preds = fine_output_unpaved[unpaved_mask].round().long()
#         elif head == 'corn':
#             unpaved_preds = corn_label_from_logits(fine_output_unpaved[unpaved_mask]).long()
#         predictions[unpaved_mask] = map_predictions_to_quality(unpaved_preds, "unpaved")


#     # Calculate accuracy
#     correct = (predictions == fine_labels).sum().item()
    
#     correct_1_off = ((predictions == fine_labels) | 
#                (predictions == fine_labels + 1) | 
#                (predictions == fine_labels - 1)).sum().item()
    
#     return correct, correct_1_off


parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])

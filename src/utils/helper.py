import sys


sys.path.append('.')

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import optim
from src import constants as const
from src.architecture import efficientnet, vgg16, vgg16_C_CNN_pretrained, vgg16_B_CNN_pretrained, vgg16_GH_CNN_pretrained, vgg16_HierarchyNet_pretrained, vgg16_test
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
from sklearn.metrics import cohen_kappa_score


def string_to_object(string):

    string_dict = {
        const.VGG16: vgg16.CustomVGG16,
        const.VGG16REGRESSION: vgg16.CustomVGG16,
        const.EFFICIENTNET: efficientnet.CustomEfficientNetV2SLogsoftmax,
        const.EFFNET_LINEAR: efficientnet.CustomEfficientNetV2SLinear,
        const.OPTI_ADAM: optim.Adam,
        const.CCNN: vgg16_C_CNN_pretrained.C_CNN,
        const.BCNN: vgg16_B_CNN_pretrained.B_CNN,
        const.HNET: vgg16_HierarchyNet_pretrained.H_NET,
        const.GHCNN: vgg16_GH_CNN_pretrained.GH_CNN,
        const.VGG16Test: vgg16_test.VGG_16_test,

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
    
class LossWeightsModifier_GH(): #TODO: wenn Zeit epochs nicht hard coden
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch):
        if 0.15 * 12 <= epoch < 0.25 * 12:
            self.alpha = torch.tensor(0.5)
            self.beta = torch.tensor(0.5)
        elif epoch >= 0.25 * 12:
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
        
        fine_loss += (1/5 * fine_loss_asphalt + 1/5 * fine_loss_concrete + 1/5 * fine_loss_paving_stones + 1/5 * fine_loss_sett + 1/5 * fine_loss_unpaved)
        
        # fine_loss += fine_loss_asphalt
        # fine_loss += fine_loss_concrete
        # fine_loss += fine_loss_paving_stones
        # fine_loss += fine_loss_sett
        # fine_loss += fine_loss_unpaved

    elif hierarchy_method == 'use_model_structure':
        
        if head == const.CLASSIFICATION or head == const.CLASSIFICATION_QWK:
            fine_loss = fine_criterion(fine_output, fine_labels)
        
        elif head == 'clm':
            fine_loss = fine_criterion(torch.log(fine_output + 1e-9), fine_labels) #TODO wie kann das berechnet werden?
                            
        # elif head == 'regression' or head == 'single':
        #     fine_output = fine_output.flatten().float()
        #     fine_labels_mapped = fine_labels_mapped.float()
        #     fine_loss = fine_criterion(fine_output, fine_labels_mapped)
            
        # elif head == 'corn':
        #     fine_loss = model.fine_criterion(fine_output, fine_labels_mapped, 4)
    
    return fine_loss



def compute_fine_metrics_hierarchical(fine_output, fine_labels, coarse_filter, hierarchy_method, head):
    
    # Initialize overall prediction tensor and metrics
    predictions = torch.zeros_like(fine_labels)
    total_mse = 0
    total_mae = 0
    all_predictions = []
    all_labels = []

    if hierarchy_method == 'use_ground_truth':
        
        masks = [
            (coarse_filter == 0),
            (coarse_filter == 1), 
            (coarse_filter == 2),  
            (coarse_filter == 3), 
            (coarse_filter == 4)  
            ]
        
        # Separate the fine outputs based on the head
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

        def compute_metrics(output, labels, mask, category):
            nonlocal total_mse, total_mae
            
            if mask.sum().item() > 0:
                if head == 'clm' or head == const.CLASSIFICATION or head == const.CLASSIFICATION_QWK:
                    preds = torch.argmax(output[mask], dim=1)
                elif head == 'regression':
                    preds = output[mask].round().long()
                elif head == 'corn':
                    preds = corn_label_from_logits(output[mask]).long()
                    
                predictions[mask] = map_predictions_to_quality(preds, category)
                
                # Collect predictions and labels for QWK
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())

                # Calculate MSE and MAE
                if head == 'regression':
                    mse = F.mse_loss(output[mask], labels[mask].float(), reduction='sum').item()
                    mae = F.l1_loss(output[mask], labels[mask].float(), reduction='sum').item()
                else:
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
        if head == 'clm' or head == const.CLASSIFICATION or head == const.CLASSIFICATION_QWK:
            predictions = torch.argmax(fine_output, dim=1)
            total_mse = F.mse_loss(predictions.float(), fine_labels.float(), reduction='sum').item()
            total_mae = F.l1_loss(predictions.float(), fine_labels.float(), reduction='sum').item()
        elif head == 'regression':
            predictions = fine_output.round().long()
            total_mse = F.mse_loss(fine_output, fine_labels.float(), reduction='sum').item()
            total_mae = F.l1_loss(fine_output, fine_labels.float(), reduction='sum').item()
            
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(fine_labels.cpu().numpy())

    # Calculate accuracy
    correct = (predictions == fine_labels).sum().item()
    
    correct_1_off = compute_one_off_accuracy_within_groups(predictions, fine_labels, parent)
    # Calculate QWK across all predictions and labels
    qwk = cohen_kappa_score(all_labels, all_predictions, weights='quadratic')
    
    hierarchy_violations = sum(is_hierarchy_violation(fine_labels, predictions, parent))
    
    # Return the sum of MSE, MAE, and QWK
    return correct, correct_1_off, total_mse, total_mae, qwk, hierarchy_violations


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
    correct_1_off = ((predictions == labels) |  # Exact match
                        (predictions == labels + 1) |  # +1 within same group
                        (predictions == labels - 1)   # -1 within same group
                    ).sum().item()
    #correct_1_off = compute_one_off_accuracy_within_groups(predictions, labels, parent)

    # Calculate MSE and MAE
    if head == 'regression':
        total_mse = F.mse_loss(outputs, labels.float(), reduction='sum').item()
        total_mae = F.l1_loss(outputs, labels.float(), reduction='sum').item()
    else:
        # For classification and other head types, compare with predicted classes
        total_mse = F.mse_loss(predictions.float(), labels.float(), reduction='sum').item()
        total_mae = F.l1_loss(predictions.float(), labels.float(), reduction='sum').item()
        
    qwk = cohen_kappa_score(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(), weights='quadratic')

    return correct, correct_1_off, total_mse, total_mae, qwk


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
            total_train_correct_one_off = 0
            total_val_correct_one_off = 0
            total_train_samples = sum(len(loader.sampler) for loader in trainloaders)
            total_val_samples = sum(len(loader.sampler) for loader in validloaders)

            for surface in surface_types:
                train_correct_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'train_correct'].sum()
                val_correct_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'val_correct'].sum()
                
                train_correct_one_off_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'train_correct_one_off'].sum()
                val_correct_one_off_sum = epoch_df.loc[epoch_df['level'] == f'smoothness/{surface}', 'val_correct_one_off'].sum()
                
                total_train_correct += train_correct_sum
                total_val_correct += val_correct_sum
                
                total_train_correct_one_off += train_correct_one_off_sum 
                total_val_correct_one_off += val_correct_one_off_sum    

            # Calculate overall accuracy
            epoch_fine_accuracy = calculate_accuracy(total_train_correct, total_train_samples)
            epoch_fine_accuracy_one_off = calculate_accuracy(total_train_correct_one_off, total_train_samples)
            val_epoch_fine_accuracy = calculate_accuracy(total_val_correct, total_val_samples)
            val_epoch_fine_accuracy_one_off = calculate_accuracy(total_val_correct_one_off, total_val_samples)

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



def is_hierarchy_violation(true_label, predicted_label, parent):
    """
    Check if the predicted label violates the hierarchy given the true label.

    Parameters:
    - true_label: The ground truth label (scalar value).
    - predicted_label: The model's prediction (scalar value).
    - parent: A tensor where each index represents a class and its value represents the parent class.

    Returns:
    - Boolean indicating whether the prediction violates the hierarchy.
    """
    # Check if the parent classes of the true and predicted labels are the same
    return parent[true_label] != parent[predicted_label]

parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])

def compute_one_off_accuracy_within_groups(predictions, labels, parent):
# Get the parent group of the predictions and labels
    #predictions = predictions - 1
    pred_parents = parent[predictions]
    label_parents = parent[labels]

    # Check if predictions are within ±1 and in the same parent group
    correct_1_off = ((predictions == labels) |  # Exact match
                        ((predictions == labels + 1) & (pred_parents == label_parents)) |  # +1 within same group
                        ((predictions == labels - 1) & (pred_parents == label_parents))    # -1 within same group
                    ).sum().item()

    return correct_1_off

class ActivationHook:
    def __init__(self, module):
        self.module = module
        self.hook = None
        self.activation = None

    def __enter__(self):
        self.hook = self.module.register_forward_hook(self.hook_func)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def hook_func(self, module, input, output):
        self.activation = output.detach()

    def close(self):
        if self.hook is not None:
            self.hook.remove()
            
            
def generate_cam(activation, classifier_weights):
    # Calculate CAM based on activation map and classifier weights
    cam = torch.einsum("ck,kij->cij", classifier_weights, activation)
    return cam

def compute_fine_prediction_hierarchical(fine_output, coarse_filter, hierarchy_method, head, fine_classes):


    if hierarchy_method == 'use_ground_truth':
    
        # Separate the fine outputs based on the head
        if head == const.REGRESSION:
            fine_output_asphalt = fine_output[:, 0:1].float()
            fine_output_concrete = fine_output[:, 1:2].float()
            fine_output_paving_stones = fine_output[:, 2:3].float()
            fine_output_sett = fine_output[:, 3:4].float()
            fine_output_unpaved = fine_output[:, 4:5].float()
        
        elif head == const.CORN:
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
        

        def compute_prediction(output, category):            
            if head == 'clm' or head == const.CLASSIFICATION or head == const.CLASSIFICATION_QWK:
                preds = torch.argmax(output, dim=1)
            elif head == 'regression':
                preds = output.round().long()
            elif head == 'corn':
                preds = corn_label_from_logits(output).long()
                    
                fine_idx = map_predictions_to_quality(preds, category)
                fine_pred_class = fine_classes[fine_idx]
                
                return fine_idx, fine_pred_class

        if coarse_filter == 0:
            fine_idx, fine_pred_class = compute_prediction(fine_output_asphalt, "asphalt")
        elif coarse_filter == 1:
            fine_idx, fine_pred_class = compute_prediction(fine_output_concrete, "concrete")
        elif coarse_filter == 2:
            fine_idx, fine_pred_class = compute_prediction(fine_output_paving_stones, "paving_stones")
        elif coarse_filter == 3:
            fine_idx, fine_pred_class = compute_prediction(fine_output_sett, "sett")
        elif coarse_filter == 4:
            fine_idx, fine_pred_class = compute_prediction(fine_output_unpaved, "unpaved")
        
        
    else:
        if head == const.CLASSIFICATION or head == const.CLM:
            fine_idx = torch.argmax(fine_output, dim=0).item()
            fine_pred_class = fine_classes[fine_idx]
 
        elif head == const.REGRESSION:
            fine_idx = fine_output.round().long()
            fine_pred_class = fine_classes[fine_idx]
            
        elif head == const.CORN:
            fine_idx = corn_label_from_logits(fine_output).long()
            fine_pred_class = fine_classes[fine_idx]
            
            
    return fine_idx, fine_pred_class



def get_reduced_out_weights(model, head):
    """
    Generates reduced weights for each classifier head in the model based on the head type.
    
    Parameters:
    - model: The model containing classifier heads for each surface type.
    - head: Type of the head (e.g., `const.CORN`, `const.REGRESSION`, `const.CLM`).
    
    Returns:
    - Dictionary of reduced weights for each classifier head.
    """
    # Define the number of classes for each head type
    head_config = {
        const.CORN: {'asphalt': 3, 'concrete': 3, 'paving_stones': 3, 'sett': 2, 'unpaved': 2},
        const.REGRESSION: {'asphalt': 1, 'concrete': 1, 'paving_stones': 1, 'sett': 1, 'unpaved': 1},
        const.CLM: {'asphalt': 4, 'concrete': 4, 'paving_stones': 4, 'sett': 3, 'unpaved': 3}
    }

    # Initialize a dictionary to hold the reduced weights
    out_weights_fine = {}

    # Loop over each surface type to compute reduced weights
    for surface_type, num_classes in head_config[head].items():
        classifier_attr = f"classifier_{surface_type}"  # Dynamically get the attribute name
        classifier_weight = getattr(model, classifier_attr)[-1].weight  # Access the weight
        # Reshape and reduce the weights as specified
        reduced_weight = classifier_weight.view(num_classes, 512, 2).sum(dim=2)
        out_weights_fine[surface_type] = reduced_weight
    
    return out_weights_fine

def reduce_weights(weight, num_classes, channels=512, reduction_dim=2):
    """
    Reshapes and reduces the weights to match CAM requirements.
    """
    return weight.view(num_classes, channels, reduction_dim).sum(dim=reduction_dim)

def get_fine_weights(model, level, head):
    """
    Returns reduced weights for fine classifiers based on the level and head.
    """
    if level == const.HIERARCHICAL:
        if head == const.CLASSIFICATION:
            out_weights_fine = model.fine_classifier[-1].weight
            return reduce_weights(out_weights_fine, num_classes=18)
        else:
            return get_reduced_out_weights(model, head)
    
    elif level == const.FLATTEN:
        out_weights = model.classifier[-1].weight
        num_classes = {
            const.CORN: 17,
            const.REGRESSION: 1
        }.get(head, 18)  # Default to 18 if head is not CORN or REGRESSION
        return reduce_weights(out_weights, num_classes=num_classes)
    
    elif level == const.SMOOTHNESS:
        out_weights = model.classifier[-1].weight
        surface_classes = {
            (const.ASPHALT, const.CONCRETE, const.PAVING_STONES): {
                const.CORN: 3,
                const.REGRESSION: 1
            },
            (const.SETT, const.UNPAVED): {
                const.CORN: 2,
                const.REGRESSION: 1
            }
        }
        
        # Determine the number of classes for the surface type and head type
        for surface_types, head_config in surface_classes.items():
            if model.surface_type in surface_types:
                num_classes = head_config.get(head, 4 if model.surface_type in (const.ASPHALT, const.CONCRETE, const.PAVING_STONES) else 3)
                return reduce_weights(out_weights, num_classes=num_classes)
        
    return None  # Return None if level doesn't match expected configurations


def compute_fine_prediction_hierarchical_GT(fine_output, coarse_filter, hierarchy_method, head, fine_classes):
    """
    Computes fine-grained predictions for each image based on coarse filter and head type.
    
    Parameters:
    - fine_output: Tensor containing fine output predictions for each image.
    - coarse_filter: List or tensor of coarse class predictions for each image.
    - hierarchy_method: Method for hierarchical prediction.
    - head: Model head type (e.g., 'regression', 'corn', 'classification').
    - fine_classes: List of fine classes.

    Returns:
    - pred_fine_classes: List of fine predictions for each image.
    """
    # Initialize list to store predictions for each image
    pred_fine_classes = []

    # Define the fine outputs structure based on the head type
    if head == 'regression':
        fine_slices = {
            "asphalt": fine_output[:, 0:1].float(),
            "concrete": fine_output[:, 1:2].float(),
            "paving_stones": fine_output[:, 2:3].float(),
            "sett": fine_output[:, 3:4].float(),
            "unpaved": fine_output[:, 4:5].float()
        }
    elif head == 'corn':
        fine_slices = {
            "asphalt": fine_output[:, 0:3],
            "concrete": fine_output[:, 3:6],
            "paving_stones": fine_output[:, 6:9],
            "sett": fine_output[:, 9:11],
            "unpaved": fine_output[:, 11:13]
        }
    else:  # For classification or other types
        fine_slices = {
            "asphalt": fine_output[:, 0:4],
            "concrete": fine_output[:, 4:8],
            "paving_stones": fine_output[:, 8:12],
            "sett": fine_output[:, 12:15],
            "unpaved": fine_output[:, 15:18]
        }

    # Iterate over each image and compute the fine predictions
    for i in range(fine_output.size(0)):
        # Get the coarse prediction for this image
        coarse_class = coarse_filter[i]
        
        # Get the corresponding fine output slice for this coarse class
        output_slice = fine_slices[coarse_class]

        # Compute predictions based on the head type
        if head in ['clm', const.CLASSIFICATION, const.CLASSIFICATION_QWK]:
            preds = torch.argmax(output_slice[i], dim=0, keepdim=True)
        elif head == 'regression':
            preds = output_slice[i].round().long()
        elif head == 'corn':
            preds = corn_label_from_logits(output_slice[i].unsqueeze(0)).long()

        # Map predictions to quality and append to results
        quality_pred = map_predictions_to_quality(preds, coarse_class)
        pred_fine_classes.append(quality_pred.item())

    return pred_fine_classes


flattened_mapping_true = {
    ("asphalt", "excellent"): 0,
    ("asphalt", "good"): 1,
    ("asphalt", "intermediate"): 2,
    ("asphalt", "bad"): 3,
    ("concrete", "excellent"): 4,
    ("concrete", "good"): 5,
    ("concrete", "intermediate"): 6,
    ("concrete", "bad"): 7,
    ("paving_stones", "excellent"): 8,
    ("paving_stones", "good"): 9,
    ("paving_stones", "intermediate"): 10,
    ("paving_stones", "bad"): 11,
    ("sett", "good"): 12,
    ("sett", "intermediate"): 13,
    ("sett", "bad"): 14,
    ("unpaved", "intermediate"): 15,
    ("unpaved", "bad"): 16,
    ("unpaved", "very_bad"): 17
}


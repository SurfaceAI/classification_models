import sys
sys.path.append('.')

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import optim
from src import constants as const
from src.architecture import Rateke_CNN, efficientnet, vgg16, vgg16_B_CNN, vgg16_CLM
import json
import argparse
from matplotlib.lines import Line2D
from torch.utils.data import Dataset
import os


def string_to_object(string):

    string_dict = {
        const.RATEKE: Rateke_CNN.ConvNet,
        const.VGG16: vgg16.CustomVGG16,
        const.VGG16REGRESSION: vgg16.CustomVGG16,
        const.EFFICIENTNET: efficientnet.CustomEfficientNetV2SLogsoftmax,
        const.EFFNET_LINEAR: efficientnet.CustomEfficientNetV2SLinear,
        const.OPTI_ADAM: optim.Adam,
        const.BCNN: vgg16_B_CNN.B_CNN,
        const.VGG16_CLM: vgg16_CLM.CustomVGG16_CLM
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
        if epoch >= 10:
            self.alpha = torch.tensor(0.6)
            self.beta = torch.tensor(0.4)
        if epoch >= 20:
            self.alpha = torch.tensor(0.2)
            self.beta = torch.tensor(0.8)
        if epoch >= 30:
            self.alpha = torch.tensor(0.0)
            self.beta = torch.tensor(1.0)
            
        return self.alpha, self.beta
    
#this helps us adopt a regression on the second level for multi-label models  
def map_quality_to_continuous(quality_label):
    quality_mapping = {
        0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 0.0, 5: 1.0,
        6: 2.0, 7: 3.0, 8: 0.0, 9: 1.0, 10: 2.0, 11: 3.0,
        12: 0.0, 13: 1.0, 14: 2.0, 15: 0.0, 16: 1.0, 17: 2.0
    }
    return quality_mapping[quality_label.item()]

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

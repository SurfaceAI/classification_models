import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from src import constants as const
from src.architecture import Rateke_CNN, efficientnet, vgg16, vgg16_B_CNN
import json
import argparse

def string_to_object(string):

    string_dict = {
        const.RATEKE: Rateke_CNN.ConvNet,
        const.VGG16: vgg16.CustomVGG16,
        const.VGG16REGRESSION: vgg16.CustomVGG16,
        const.EFFICIENTNET: efficientnet.CustomEfficientNetV2SLogsoftmax,
        const.EFFNET_LINEAR: efficientnet.CustomEfficientNetV2SLinear,
        const.OPTI_ADAM: optim.Adam,
        const.BCNN: vgg16_B_CNN.B_CNN,
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
        0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 1.0, 5: 2.0,
        6: 3.0, 7: 4.0, 8: 1.0, 9: 2.0, 10: 3.0, 11: 4.0,
        12: 2.0, 13: 3.0, 14: 4.0, 15: 3.0, 16: 4.0, 17: 5.0
    }
    return quality_mapping[quality_label.item()]

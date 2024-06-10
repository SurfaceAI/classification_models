import sys
sys.path.append('.')

import os
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from src import constants as const
from src.architecture import Rateke_CNN, efficientnet, vgg16
import json
import argparse

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

def string_to_object(string):

    string_dict = {
        const.RATEKE: Rateke_CNN.ConvNet,
        const.VGG16: vgg16.CustomVGG16,
        const.VGG16REGRESSION: vgg16.CustomVGG16,
        const.EFFICIENTNET: efficientnet.CustomEfficientNetV2SLogsoftmax,
        const.EFFNET_LINEAR: efficientnet.CustomEfficientNetV2SLinear,
        const.OPTI_ADAM: optim.Adam,
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

def extract_type_and_quality_and_id_from_img_path(img_path):
    #"no_street", "not_recognizable","revise"
    valid_types = ["asphalt", "concrete", "paving_stones", "sett", "unpaved"]
    path_split = img_path.split('/') #todo. change back to ('\\')'/'
    extracted_type = path_split[-3]
    if extracted_type in valid_types:
        extracted_quality = path_split[-2]
    else:
        extracted_type = path_split[-2]
        extracted_quality = None
    extracted_id = os.path.splitext(path_split[-1])[0]
    return extracted_type, extracted_quality, extracted_id


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
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

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
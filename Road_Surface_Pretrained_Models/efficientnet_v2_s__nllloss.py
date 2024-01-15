import sys

sys.path.append("./")

import os
import random
from collections import OrderedDict

import helper
import numpy as np
import preprocessing
import torch
from torch import nn, optim
from torchvision import models

import src.training as training
import wandb

# config
batch_size = 48
valid_batch_size = 48
epochs = 2
# learning_rate = 0.003
# learning_rate = 0.001
learning_rate = 0.0003
seed = 42

validation_size = 0.2

image_height = 256
image_width = 256
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
horizontal_flip = True
random_rotation = 10


# # # unfreeze all parameters after x epochs
# # unfreeze = False
# epochs_freeze = 5

data_version = "V0"

# image data path
data_path = (
    "/Users/edith/HTW Cloud/SHARED/SurfaceAI/data/mapillary_images/training_data"
)


# W&B initialisation
wandb.login()
run = wandb.init(
    # set project and tags
    project="road-surface-classification-type",
    name="efficient net",
    # track hyperparameters and run metadata
    config={
        "architecture": "Efficient Net v2 s",
        "dataset": data_version,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "augmented": "Yes",
    },
)

# preprocessing

general_transform = {
    "resize": (image_height, image_width),
    "normalize": (norm_mean, norm_std),
}

train_augmentation = {
    "random_horizontal_flip": horizontal_flip,
    "random_rotation": random_rotation,
}


train_transform = preprocessing.transform(**general_transform, **train_augmentation)
valid_transform = preprocessing.transform(**general_transform)


torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

# dataset
data_root = os.path.join(data_path, data_version)

train_data, valid_data = preprocessing.train_validation_spilt_datasets(
    data_root, validation_size, train_transform, valid_transform, random_state=seed
)
# len(torch.tensor(train_data.dataset.targets)[train_data.indices])

# # Define transforms for the training data and testing data
# train_transforms = transforms.Compose([transforms.RandomRotation(10),
#                                        transforms.Resize((image_height, image_width)),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.485, 0.456, 0.406],
#                                                             [0.229, 0.224, 0.225])])

# # Create dataset
# train_data_old = datasets.ImageFolder(data_root, transform=train_transforms)
# trainloader_old = torch.utils.data.DataLoader(train_data_old, batch_size=batch_size, shuffle=True)

num_classes = len(train_data.classes)

trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size)


# model
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

# Unfreeze parameters
for param in model.parameters():
    param.requires_grad = True

# adapt output layer
fc = nn.Sequential(
    OrderedDict(
        [("fc1", nn.Linear(1280, num_classes)), ("output", nn.LogSoftmax(dim=1))]
    )
)

model.classifier[1] = fc

# optimizer
# Start with training the output layer parameters, feature parameters are frozen
# TODO: unfreeze parameters
optimizer = optim.Adam(model.classifier[1].parameters(), lr=learning_rate)

# loss, reduction='sum'
criterion = nn.NLLLoss()

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# train the model

train_loss_list = []
valid_loss_list = []
accuracy_list = []

for epoch in range(epochs):

    train_loss = training.train_epoch(model, trainloader, criterion, optimizer, device)

    val_loss, val_accuracy = training.validate_epoch(
        model, validloader, criterion, device
    )

    train_loss_list.append(train_loss)
    valid_loss_list.append(val_loss)
    accuracy_list.append(val_accuracy)

    wandb.log(
        {
            "epoch": epoch + 1,
            "train loss": train_loss_list[-1],
            "validation loss": valid_loss_list[-1],
            "validation accuracy": accuracy_list[-1],
        }
    )

    print(
        f"Epoch {epoch+1}/{epochs}.. ",
        f"Train loss: {train_loss_list[-1]:.3f}.. ",
        f"Test loss: {valid_loss_list[-1]:.3f}.. ",
        f"Test accuracy: {accuracy_list[-1]:.3f}",
    )

torch.save(model, "efficientnet_v2_s__nllloss.pt")
wandb.save("efficientnet_v2_s__nllloss.pt")

wandb.unwatch()

print("Done.")

import sys
sys.path.append('./')

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from pathlib import Path
import os
import preprocessing

# config
batch_size = 48
test_batch_size = 48
epochs = 10
# lr = 0.003
# lr = 0.001
lr = 0.0003
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

data_version = 'V0'

# image data path
data_path = '/Users/edith/HTW Cloud/SHARED/SurfaceAI/data/mapillary_images/training_data'

data_root= os.path.join(data_path, data_version)

# preprocessing

general_transform = {
    'resize': (image_height, image_width),
    'normalize': (norm_mean, norm_std),
}

train_augmentation = {
    'random_horizontal_flip': horizontal_flip,
    'random_rotation': 10,
}

train_transform = preprocessing.transform(**general_transform, **train_augmentation)
valid_transform = preprocessing.transform(**general_transform)

# dataset
train_data, valid_data = preprocessing.train_validation_spilt_datasets(data_root, validation_size, train_transform, valid_transform, random_state=seed)
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

num_classes = len(train_data.dataset.classes)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch_size)


# model
model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

# Unfreeze parameters
for param in model.parameters():
    param.requires_grad = True

# adapt output layer
from collections import OrderedDict
fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1280, num_classes)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

model.classifier[1] = fc

# optimizer
# Start with training the output layer parameters, feature parameters are frozen
# TODO: unfreeze parameters
optimizer = optim.Adam(model.classifier[1].parameters(), lr=lr)

# loss, reduction='sum'
criterion = nn.NLLLoss(reduction='sum')

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

torch.manual_seed(seed)

# config wandb
# TODO

# train the model

train_loss_list = []
valid_loss_list = []
accuracy_list = []

for epoch in range(epochs):

    # train with all batches for one epoch
    train_loss = 0
    model.train()
    for inputs, labels in trainloader:
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        # # unfreeze and optimize all parameters after x epochs
        # if unfreeze and epoch >= epochs_freeze:
        #     optimizer = optim.Adam(model.parameters(), lr=lr)

        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # for i, l in enumerate(logps):
        #   print(f"Train: True Label: {labels[i]}, prediction: {l}")

    # validate after one epoch
    with torch.no_grad():
        valid_loss = 0
        correct_predictions = 0
        model.eval()
        for inputs, labels in validloader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()

            # for i, l in enumerate(logps):
            #   print(f"Test:  True Label: {labels[i]}, prediction: {l}")


    train_loss_list.append(train_loss/len(trainloader.sampler))
    valid_loss_list.append(valid_loss/len(validloader.sampler))
    accuracy_list.append(correct_predictions/len(validloader.sampler))


    print(f"Epoch {epoch+1}/{epochs}.. ",
          f"Train loss: {train_loss_list[-1]:.3f}.. ",
          f"Test loss: {valid_loss_list[-1]:.3f}.. ",
          f"Test accuracy: {accuracy_list[-1]:.3f}",)

print("Done.")
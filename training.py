import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import os
import preprocessing
import helper
import random

import wandb

import config as general_config

# complete training routine
def config_and_train_model(config, load_model, optimizer_class, criterion, augment=None):

    torch.manual_seed(config.get('seed'))

    _ = init_wandb(config, augment)

    # dataset
    data_path = create_data_path()
    data_root= os.path.join(data_path, config.get('dataset'))

    train_transform, valid_transform = create_transform(config, augment)

    train_data, valid_data = preprocessing.train_validation_spilt_datasets(data_root, config.get('validation_size'), train_transform, valid_transform, random_state=config.get('seed'))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.get('batch_size'), shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=config.get('valid_batch_size'))

    # load model
    num_classes = len(train_data.classes)

    #TODO: instanciate model!
    result = load_model(num_classes)
    if isinstance(result, tuple):
            model, optimizer_layers = result
    else:
        model = result
        optimizer_layers = None

    # Unfreeze parameters
    for param in model.parameters():
        param.requires_grad = True

    # setup optimizer
    if optimizer_layers is None:
        optimizer_params = model.parameters()
    else:
        optimizer_params = []
        for layer in optimizer_layers:
            optimizer_params += [p for p in layer.parameters()]

    # set parameters to optimize
    optimizer = optimizer_class(optimizer_params, lr=config.get('learning_rate'))

    # Use GPU if it's available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{general_config.gpu_kernel}" if torch.cuda.is_available() else "cpu")
    print(device)


    trained_model = train(model, config.get('save_name'), trainloader, validloader, criterion, optimizer, device, config.get('epochs'))



# create images data path
# TODO: generalize for all users
def create_data_path():
    data_path = config.training_data_path
    #data_path = '/Users/edith/HTW Cloud/SHARED/SurfaceAI/data/mapillary_images/training_data'
    return data_path


# W&B initialisation
def init_wandb(config_input, augment=None):

    #set augmentation
    if augment is not None:
        augmented = "Yes"
    else:
        augmented = "No"
    wandb.login()
    wandb.init(
        #set project and tags 
        project = config_input.get('project'), 
        name = config_input.get('name'),
        
        #track hyperparameters and run metadata
        # TODO: config=config???
        config={
        "architecture": config_input.get('architecture'),
        "dataset": config_input.get('dataset'),
        "learning_rate": config_input.get('learning_rate'),
        "batch_size": config_input.get('batch_size'),
        "seed": config_input.get('seed'),
        "augmented": augmented
        }
    ) 

# preprocessing
def create_transform(config, augment=None):

    # TODO: check if image_size/normalize in config
    general_transform = {
        'resize': config.get('image_size_h_w'),
        'normalize': (config.get('norm_mean'), config.get('norm_std')),
    }

    train_augmentation = augment


    train_transform = preprocessing.transform(**general_transform, **train_augmentation)
    valid_transform = preprocessing.transform(**general_transform)

    return train_transform, valid_transform




# train the model
def train(model, model_name, trainloader, validloader, criterion, optimizer, device, epochs):

    model.to(device)

    for epoch in range(epochs):

        train_loss = train_epoch(model, trainloader, criterion, optimizer, device)

        val_loss, val_accuracy = validate_epoch(model, validloader, criterion, device)

        wandb.log({'epoch': epoch+1, 'train loss': train_loss, 'validation loss': val_loss, 'validation accuracy': val_accuracy})

        print(f"Epoch {epoch+1}/{epochs}.. ",
            f"Train loss: {train_loss:.3f}.. ",
            f"Test loss: {val_loss:.3f}.. ",
            f"Test accuracy: {val_accuracy:.3f}",)

    save_model(model, model_name)
    wandb.save(model_name)
    wandb.unwatch()

    print("Done.")

    return model


# train a single epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.reduction = 'sum'
    running_loss = 0.0

    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader.sampler)

# validate a single epoch
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    criterion.reduction = 'sum'
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()

    return running_loss / len(dataloader.sampler), correct_predictions / len(dataloader.sampler)

# save model locally
def save_model(model, model_name):
    
    path = "Road_Surface_Pretrained_Models"
    folder = "models"

    folder_path = os.path.join(path, folder)

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    model_path = os.path.join(folder_path, model_name)
    torch.save(model, model_path)
import sys

sys.path.append(".")
# sys.path.append('..')

import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from collections import OrderedDict
import os
from utils import preprocessing
from utils import constants
from utils import helper
from utils import general_config
from model_config import model_config
import random

import wandb

from utils import general_config

def run_fixed(project, fixed_config, name=None):

    os.environ["WANDB_MODE"] = general_config.wandb_mode

    run_training(project=project, name=name, config=fixed_config)

    # TODO: save model

def run_sweep(project, sweep_config, sweep_counts):

    os.environ["WANDB_MODE"] = general_config.wandb_mode
    
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    wandb.agent(sweep_id=sweep_id, function=run_training, count=sweep_counts)

    # TODO: save best model

    print('Sweep done.')

# main for sweep and single training
def run_training(project=None, name=None, config=None): 

    run = wandb.init(project=project, name=name, config=config)
    config = wandb.config

    model_cfg = model_config.config(config.get('model'))
    
    general_transform = config.get("transform")

    augment = None
    if config.get("augment") == constants.AUGMENT_TRUE:
        augment = general_config.augmentation

    model_cls = model_cfg.get('model_cls')
    optimizer_cls = model_config.optimizer_config(config.get('optimizer_cls'))
    dataset = config.get('dataset')
    label_type = config.get('label_type')
    selected_classes = config.get('selected_classes')
    validation_size = config.get('validation_size')
    batch_size = config.get('batch_size')
    valid_batch_size = general_config.valid_batch_size
    learning_rate = config.get('learning_rate')
    random_seed = config.get('seed')
    criterion = model_cfg.get('criterion_cls')() 
    epochs = config.get('epochs')
    # model_saving_name = # from model_config?

    
    device = torch.device(
        f"cuda:{general_config.gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    trainloader, validloader, model, optimizer = prepare_train(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        transform=general_transform,
        augment=augment,
        dataset=dataset,
        label_type=label_type,
        selected_classes=selected_classes,
        validation_size=validation_size,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed
    )

    trained_model = train(
        model=model,
        model_saving_name=None,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )


def prepare_train(
    model_cls,
    optimizer_cls,
    transform,
    augment,
    dataset,
    label_type,
    selected_classes,
    validation_size,
    batch_size,
    valid_batch_size,
    learning_rate,
    random_seed,
):
    train_data, valid_data = preprocessing.create_train_validation_datasets(
        dataset=dataset,
        label_type=label_type,
        selected_classes=selected_classes,
        validation_size=validation_size,
        general_transform=transform,
        augmentation=augment,
        random_state=random_seed,
    )

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=valid_batch_size
    )

    # load model
    num_classes = len(train_data.classes)

    # TODO: instanciate model!
    model = model_cls(num_classes)

    # Unfreeze parameters
    for param in model.parameters():
        param.requires_grad = True

    optimizer_layers = None
    if hasattr(model, "get_optimizer_layers") and callable(model.get_optimizer_layers):
        optimizer_layers = model.get_optimizer_layers()

    # setup optimizer
    if optimizer_layers is None:
        optimizer_params = model.parameters()
    else:
        optimizer_params = []
        for layer in optimizer_layers:
            optimizer_params += [p for p in layer.parameters()]

    # set parameters to optimize
    optimizer = optimizer_cls(optimizer_params, lr=learning_rate)

    return trainloader, validloader, model, optimizer

# complete training routine
def config_and_train_model(
    config, model_cls, optimizer_cls, criterion, augmentation=None
):
    torch.manual_seed(config.get("seed"))  # main

    _ = init_wandb(config, augmentation) # main

    general_transform = {   # nested params
        "resize": config.get("image_size_h_w"),
        "crop": config.get("crop"),
        "normalize": config.get("normalization"),
    }

    train_data, valid_data = preprocessing.create_train_validation_datasets(
        config.get("dataset"),
        config.get("label_type"),
        config.get("selected_classes"),
        config.get("validation_size"),
        general_transform,
        augmentation,
        random_state=config.get("seed"),
    )

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=config.get("batch_size"), shuffle=True
    )
    validloader = torch.utils.data.DataLoader(
        valid_data, batch_size=config.get("valid_batch_size")
    )

    # load model
    num_classes = len(train_data.classes)

    # TODO: instanciate model!
    model = model_cls(num_classes)

    optimizer_layers = None
    if hasattr(model, "get_optimizer_layers") and callable(model.get_optimizer_layers):
        optimizer_layers = model.get_optimizer_layers()

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
    optimizer = optimizer_cls(optimizer_params, lr=config.get("learning_rate"))

    # Use GPU if it's available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
        f"cuda:{general_config.gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )
    print(device)  # main

    trained_model = train(
        model,
        config.get("save_name"),
        trainloader,
        validloader,
        criterion,
        optimizer,
        device,
        config.get("epochs"),
    )


# W&B initialisation
def init_wandb(config_input, augment=None):
    # set augmentation
    if augment is not None:
        augmented = "Yes"
    else:
        augmented = "No"
    wandb.login()
    wandb.init(
        # set project and tags
        project=config_input.get("project"),
        name=config_input.get("name"),
        # track hyperparameters and run metadata
        # TODO: config=config???
        config={
            "architecture": config_input.get("architecture"),
            "dataset": config_input.get("dataset"),
            "learning_rate": config_input.get("learning_rate"),
            "batch_size": config_input.get("batch_size"),
            "seed": config_input.get("seed"),
            "augmented": augmented,
            "crop": config_input.get("crop"),
            "selected_classes": config_input.get("selected_classes"),
        },
    )


# train the model
def train(
    model,
    model_saving_name,
    trainloader,
    validloader,
    criterion,
    optimizer,
    device,
    epochs,
):
    model.to(device)

    for epoch in range(epochs):
        train_loss = train_epoch_test(model, trainloader, criterion, optimizer, device)

        val_loss, val_accuracy = validate_epoch_test(
            model, validloader, criterion, device
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "eval/loss": val_loss,
                "eval/acc": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch+1}/{epochs}.. ",
            f"Train loss: {train_loss:.3f}.. ",
            f"Test loss: {val_loss:.3f}.. ",
            f"Test accuracy: {val_accuracy:.3f}",
        )

    # save_model(model, model_saving_name)
    # wandb.save(model_saving_name)
    wandb.unwatch()

    print("Done.")

    return model


# train a single epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.reduction = "sum"
    running_loss = 0.0

    for inputs, labels in dataloader:
        helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader.sampler)


# train a single epoch
def train_epoch_test(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.reduction = "sum"
    running_loss = 0.0

    for inputs, labels in dataloader:
        # helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        break

    return running_loss / len(dataloader.sampler)


# validate a single epoch
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    criterion.reduction = "sum"
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

    return running_loss / len(dataloader.sampler), correct_predictions / len(
        dataloader.sampler
    )


# validate a single epoch
def validate_epoch_test(model, dataloader, criterion, device):
    model.eval()
    criterion.reduction = "sum"
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

            break

    return running_loss / len(dataloader.sampler), correct_predictions / len(
        dataloader.sampler
    )


# save model locally
def save_model(model, model_name):
    save_path = general_config.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_path = os.path.join(save_path, model_name)
    torch.save(model, model_path)


# load model from wandb
def load_wandb_model(model_name, run_path):
    best_model = wandb.restore(model_name, run_path=run_path)

    model = torch.load(best_model.name)

    return model

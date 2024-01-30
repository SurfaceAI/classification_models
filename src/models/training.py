import sys

sys.path.append(".")
# sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn, optim
from torchvision import models
from collections import OrderedDict, Counter
import time
import os
from src.utils import preprocessing
from src import constants
from src.utils import helper
from src.config import general_config
from src.utils import str_conv
from src.utils import param_config

import random

import wandb

from src.config import general_config

def run_fixed_training(individual_params, model, project=None, name=None):

    os.environ["WANDB_MODE"] = general_config.wandb_mode

    if general_config.wandb_mode == constants.WANDB_MODE_OFF:
        project = 'OFF_' + project

    config = param_config.fixed_config(individual_params=individual_params, model=model)

    wandb_training(project=project, name=name, config=config)

    # TODO: save model

def run_sweep_training(individual_params, models, method, metric=None, project=None, name=None, sweep_counts=constants.WANDB_DEFAULT_SWEEP_COUNTS):

    os.environ["WANDB_MODE"] = general_config.wandb_mode
    
    if general_config.wandb_mode == constants.WANDB_MODE_OFF:
        project = 'OFF_' + project

    sweep_config = param_config.sweep_config(individual_params=individual_params, models=models, method=method, metric=metric, name=name)

    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    wandb.agent(sweep_id=sweep_id, function=wandb_training, count=sweep_counts)

    # TODO: save best model

    print('Sweep done.')

# main for sweep and single training
def wandb_training(project=None, name=None, config=None):

    run = wandb.init(project=project, name=name, config=config)
    config = wandb.config

    model_name = config.get('model')
    model_cfg = str_conv.model_name_to_config(model_name)
    model_cls = model_cfg.get('model_cls')
    criterion = model_cfg.get('criterion')
    
    optimizer_cls = str_conv.optim_name_to_class(config.get('optimizer_cls'))
    dataset = config.get('dataset')
    label_type = config.get('label_type')
    selected_classes = config.get('selected_classes')
    validation_size = config.get('validation_size')
    batch_size = config.get('batch_size')
    valid_batch_size = general_config.valid_batch_size
    learning_rate = config.get('learning_rate')
    epochs = config.get('epochs')
    seed = config.get('seed')
    general_transform = config.get("transform")

    augment = None
    if config.get('augment') == constants.AUGMENT_TRUE:
        augment = general_config.augmentation

    trained_model, model_path = run_training(model_name=model_name, model_cls=model_cls, optimizer_cls=optimizer_cls, criterion=criterion, dataset=dataset, label_type=label_type, validation_size=validation_size, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, valid_batch_size=valid_batch_size, general_transform=general_transform, augment=augment, selected_classes=selected_classes, seed=seed)

    wandb.save(model_path)

def run_training(model_name, model_cls, optimizer_cls, criterion, dataset, label_type, validation_size, learning_rate, epochs, batch_size, valid_batch_size, general_transform, augment=None, selected_classes=None, seed=42):
    torch.manual_seed(seed)
    
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
        random_seed=seed
    )

    trained_model = train(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    saving_name = save_model(trained_model, model_name)
    print(f'Model saved locally: {saving_name}')

    return trained_model, saving_name

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

    # TODO: loader in preprocessing?
    class_counts = Counter(train_data.targets)
    sample_weights = [1/class_counts[i] for i in train_data.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))

    trainloader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler
    )    # shuffle=True only if no sampler defined
    validloader = DataLoader(
        valid_data, batch_size=valid_batch_size
    )

    # load model
    num_classes = len(train_data.classes)

    # instanciate model with number of classes
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

# TODO: old!
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

    trainloader = DataLoader(
        train_data, batch_size=config.get("batch_size"), shuffle=True
    )
    validloader = DataLoader(
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

    # save_model(model, model_saving_name)
    # wandb.save(model_saving_name)
    wandb.unwatch()  # not necessary?

    


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
    trainloader,
    validloader,
    criterion,
    optimizer,
    device,
    epochs,
):
    model.to(device)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch_test(model, trainloader, criterion, optimizer, device)

        val_loss, val_accuracy = validate_epoch_test(
            model, validloader, criterion, device
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_accuracy,
                "eval/loss": val_loss,
                "eval/acc": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch+1}/{epochs}.. ",
            f"Train loss: {train_loss:.3f}.. ",
            f"Test loss: {val_loss:.3f}.. ",
            f"Train accuracy: {train_accuracy:.3f}.. ",
            f"Test accuracy: {val_accuracy:.3f}",
        )

    print("Done.")

    return model


# train a single epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.reduction = "sum"
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in dataloader:
        # helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()

    return running_loss / len(dataloader.sampler), correct_predictions / len(
        dataloader.sampler
    )


# train a single epoch
def train_epoch_test(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.reduction = "sum"
    running_loss = 0.0
    correct_predictions = 0
    
    targets = []
    for _, labels in dataloader:
        targets.extend(labels.numpy())

    for inputs, labels in dataloader:
        # helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        break

    return running_loss / len(dataloader.sampler), correct_predictions / len(
        dataloader.sampler
    )


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

    ts = round(time.time())
    model_name += '_' + str(ts) + '.pt'

    model_path = os.path.join(save_path, model_name)
    torch.save(model.state_dict(), model_path)

    return model_path


# load model from wandb
def load_wandb_model(model_name, run_path):
    best_model = wandb.restore(model_name, run_path=run_path)

    model = torch.load(best_model.name)

    return model

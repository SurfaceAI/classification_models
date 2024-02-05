import sys

sys.path.append(".")
# sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn, optim
from torchvision import models
from collections import OrderedDict, Counter
from datetime import datetime
import os
from src.utils import preprocessing
from src import constants
from src.utils import helper
from src.config import general_config
from src.utils import str_conv
from src.utils import wandb_conv
from src.utils import checkpointing

import random

import wandb

from src.config import general_config

def run_fixed_training(individual_params, model, project=None, name=None, level=None):

    os.environ["WANDB_MODE"] = general_config.wandb_mode

    if general_config.wandb_mode == constants.WANDB_MODE_OFF:
        project = 'OFF_' + project

    config = wandb_conv.fixed_config(individual_params=individual_params, model=model, level=level)

    wandb_training(project=project, name=name, config=config)

    # TODO: save model

def run_sweep_training(individual_params, models, method, metric=None, project=None, name=None, level=None, sweep_counts=constants.WANDB_DEFAULT_SWEEP_COUNTS):

    os.environ["WANDB_MODE"] = general_config.wandb_mode
    
    if general_config.wandb_mode == constants.WANDB_MODE_OFF:
        project = 'OFF_' + project

    sweep_config = wandb_conv.sweep_config(individual_params=individual_params, models=models, method=method, metric=metric, name=name, level=level)

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

    level = config.get('level').split('/',1)
    type_class = None
    if len(level) == 2:
        type_class = level[-1]

    augment = None
    if config.get('augment') == constants.AUGMENT_TRUE:
        augment = general_config.augmentation

    start_time = datetime.fromtimestamp(run.start_time).strftime("%Y%m%d_%H%M%S")
    id = run.id
    separator = '-'
    saving_name = separator.join([separator.join(level), model_name, start_time, id])
    saving_name = saving_name + '.pt'

    trained_model, model_path = run_training(saving_name=saving_name, model_cls=model_cls, optimizer_cls=optimizer_cls, criterion=criterion, dataset=dataset, label_type=label_type, validation_size=validation_size, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, valid_batch_size=valid_batch_size, general_transform=general_transform, augment=augment, selected_classes=selected_classes, level=level[0], type_class=type_class, seed=seed)
    
    # wandb.save(model_path)

def run_training(saving_name, model_cls, optimizer_cls, criterion, dataset, label_type, validation_size, learning_rate, epochs, batch_size, valid_batch_size, general_transform, augment=None, selected_classes=None, level=None, type_class=None, seed=42):
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
        level=level,
        type_class=type_class,
        selected_classes=selected_classes,
        validation_size=validation_size,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=learning_rate,
        random_seed=seed
    )

    trained_model = train(
        model=model,
        saving_name=saving_name,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    # TODO: save best instead of last model
    model_path = save_model(trained_model, saving_name)
    print(f'Model saved locally: {model_path}')

    return trained_model, model_path

def prepare_train(
    model_cls,
    optimizer_cls,
    transform,
    augment,
    dataset,
    label_type,
    level,
    type_class,
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
        level=level,
        type_class=type_class,
    )

    # torch.save(valid_data, os.path.join(general_config.save_path, "valid_data.pt"))
    print(f'classes: {train_data.class_to_idx}')

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
    config, model_cls, optimizer_cls, criterion, type_class=None, augmentation=None
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
        type_class=type_class,
    )

    torch.save(valid_data, os.path.join(general_config.save_path, "valid_data.pt"))

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
    

# TODO: OLD!
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
    saving_name,
    trainloader,
    validloader,
    criterion,
    optimizer,
    device,
    epochs,
):
    model.to(device)

    checkpointer = checkpointing.CheckpointSaver(dirpath=general_config.save_path, saving_name=saving_name, decreasing=False, top_n=general_config.checkpoint_top_n, early_stop_thresh=general_config.early_stop_thresh)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch_test(model, trainloader, criterion, optimizer, device)

        val_loss, val_accuracy = validate_epoch_test(
            model, validloader, criterion, device
        )

        # checkpoint saving with early stopping
        early_stop = checkpointer(model=model, epoch=epoch, metric_val=val_accuracy, optimizer=optimizer)

        # TODO: with try? for usage w/o wandb
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

        if early_stop:
            print(f"Early stopped training at epoch {epoch}")
            break

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

        # TODO: metric as function, metric_name as input argument
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
def save_model(model, saving_name):
    save_path = general_config.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_path = os.path.join(save_path, saving_name)
    torch.save(model.state_dict(), model_path)

    # TODO: return value saving success
    return model_path


# load model from wandb
def load_wandb_model(model_name, run_path):
    best_model = wandb.restore(model_name, run_path=run_path)

    model = torch.load(best_model.name)

    return model


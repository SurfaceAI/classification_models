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
import time
import os
from src.utils import preprocessing
from src import constants
from src.utils import helper
from src.utils import parser
from src.utils import checkpointing

import random

import wandb

# TODO: what is the difference to run_training?
def run_fixed_training(config, project=None, name=None, level=None, wandb_mode=constants.WANDB_MODE_OFF, wandb_on=True):
    # TODO: doc config

    os.environ["WANDB_MODE"] = wandb_mode

    if wandb_mode == constants.WANDB_MODE_OFF:
        project = 'OFF_' + project

    selected_classes = config.get['selected_classes']
    level_list = extract_levels(level=level, selected_classes=selected_classes)
    
    for level in level_list:
        config = {
            **config,
            **level,
        }
        run_training(project=project, name=name, config=config, wandb_on=wandb_on)
        print(f'Level {level} trained.')

    # TODO: save model
    print('Done.')

def run_sweep_training(config_params, method, metric=None, project=None, name=None, level=None, sweep_counts=constants.WANDB_DEFAULT_SWEEP_COUNTS, wandb_mode=constants.WANDB_MODE_OFF):
    # TODO: doc config_params

    os.environ["WANDB_MODE"] = wandb_mode

    if wandb_mode == constants.WANDB_MODE_OFF:
        project = 'OFF_' + project

    selected_classes = config_params.get['selected_classes']['value']
    level_list = extract_levels(level=level, selected_classes=selected_classes)

    for level in level_list:
        for key, value in level.items():
            level[key] = {'value': value}

        sweep_params = {
            **config_params,
            **level,
        }

        sweep_config = {
            'name': name,
            **method,
            **metric,
            'parameters': sweep_params,
        }
        
        sweep_id = wandb.sweep(sweep=sweep_config, project=project)

        wandb.agent(sweep_id=sweep_id, function=run_training, count=sweep_counts)
        # TODO: save/print best model
        print(f'Level {level} trained.')

    

    print('Done.')

# main for sweep and single training
def run_training(project=None, name=None, config=None, wandb_on=True):

    # TODO: config sweep ...
    if wandb_on:
        run = wandb.init(project=project, name=name, config=config)
        config = wandb.config
        # best instead of last value for metric
        wandb.define_metric("eval/acc", summary="max")

    model_name = config.get('model')
    model_cfg = parser.model_name_to_config(model_name)
    model_cls = model_cfg.get('model_cls')
    criterion = model_cfg.get('criterion')
    
    optimizer_cls = parser.optim_name_to_class(config.get('optimizer_cls'))
    dataset = config.get('dataset')
    selected_classes = config.get('selected_classes')
    validation_size = config.get('validation_size')
    batch_size = config.get('batch_size')
    valid_batch_size = config.get('valid_batch_size')
    learning_rate = config.get('learning_rate')
    epochs = config.get('epochs')
    seed = config.get('seed')
    general_transform = config.get("transform")
    augment = config.get("augment")
    gpu_kernel = config.get("gpu_kernel")
    
    checkpoint_top_n = config.get("checkpoint_top_n", default=constants.CHECKPOINT_DEFAULT_TOP_N)
    early_stop_thresh = config.get("early_stop_thresh", default=constants.EARLY_STOPPING_DEFAULT)
    save_state = config.get("save_state", default=True)

    level = config.get('level').split('/',1)
    type_class = None
    if len(level) == 2:
        type_class = level[-1]

    data_root = config.get('root/data')
    model_root = config.get('root/model')

    start_time = datetime.fromtimestamp(time.time() if not wandb_on else run.start_time).strftime("%Y%m%d_%H%M%S")
    id = '' if not wandb_on else '-' + run.id
    saving_name = '-'.join(level) + '-' + model_name + '-' + start_time + id + '.pt'

    torch.manual_seed(seed)
    
    # TODO: testing gpu_kernel = None
    device = torch.device(
        f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    trainloader, validloader, model, optimizer = prepare_train(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        transform=general_transform,
        augment=augment,
        dataset=dataset,
        data_root=data_root,
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
        model_saving_path=model_root,
        model_saving_name=saving_name,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        wandb_on=wandb_on,
        checkpoint_top_n=checkpoint_top_n,
        early_stop_thresh=early_stop_thresh,
        save_state=save_state,
        config=config,

    )

    # TODO: save best instead of last model (if checkpoint used)
    # TODO: save dict incl. config .. + model param (compare checkpoints)
    # model_path = save_model(trained_model, saving_name)
    # print(f'Model saved locally: {model_path}')

    return trained_model #, model_path

    # wandb.save(model_path)


def prepare_train(
    model_cls,
    optimizer_cls,
    transform,
    augment,
    dataset,
    data_root,
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
        data_root=data_root,
        dataset=dataset,
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
    # TODO: weighted sampling on/off?
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

# train the model
def train(
    model,
    model_saving_path,
    model_saving_name,
    trainloader,
    validloader,
    criterion,
    optimizer,
    device,
    epochs,
    wandb_on,
    checkpoint_top_n=constants.CHECKPOINT_DEFAULT_TOP_N,
    early_stop_thresh=constants.EARLY_STOPPING_DEFAULT,
    save_state=True,
    config=None,
):
    model.to(device)

    # TODO: decresing depending on metric
    checkpointer = checkpointing.CheckpointSaver(dirpath=model_saving_path, saving_name=model_saving_name, decreasing=False, config=config, top_n=checkpoint_top_n, early_stop_thresh=early_stop_thresh, save_state=save_state)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch_test(model, trainloader, criterion, optimizer, device)

        val_loss, val_accuracy = validate_epoch_test(
            model, validloader, criterion, device
        )

        # checkpoint saving with early stopping
        early_stop = checkpointer(model=model, epoch=epoch, metric_val=val_accuracy, optimizer=optimizer)

        if wandb_on:
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
def save_model(model,saving_path, saving_name):
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    model_path = os.path.join(saving_path, saving_name)
    torch.save(model.state_dict(), model_path)

    # TODO: return value saving success
    return model_path


# load model from wandb
def load_wandb_model(model_name, run_path):
    best_model = wandb.restore(model_name, run_path=run_path)

    model = torch.load(best_model.name)

    return model

def extract_levels(level, selected_classes):
    # TODO: selected_classes must not be None (for surface/smoothness), but None should be possible (=all classes)
    level_list = []
    if level == constants.FLATTEN:
        level_list.append({'level': level, 'selected_classes': selected_classes})
    elif level == constants.SURFACE:
        level_list.append({'level': level, 'selected_classes': list(selected_classes.keys())})     
    elif level == constants.SMOOTHNESS:
        for type_class in selected_classes.keys():
            level_list.append({'level': level + '/' + type_class, 'selected_classes': selected_classes[type_class]}) 
    else:
        level_list.append({'level': level, 'selected_classes': selected_classes})
    
    return level_list
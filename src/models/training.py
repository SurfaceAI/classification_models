import sys

sys.path.append(".")

import os
import time
from collections import Counter
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import wandb
from src import constants as const
from src.utils import checkpointing, helper, preprocessing
import argparse



def run_training(
    config,
    is_sweep=False,
):
    os.environ["WANDB_MODE"] = config.get("wandb_mode", const.WANDB_MODE_OFF)

    project = config.get("project")
    if os.environ["WANDB_MODE"] == const.WANDB_MODE_OFF:
        project = "OFF_" + project
    
    # extract all levels that need training, only multi elements if smoothness is trained
    # each surface has to be trained seperately
    to_train_list = extract_levels(
        level=config.get("level"), selected_classes=config.get("selected_classes")
    )

    for t in to_train_list:
        config = {
            **config,
            **t,
        }

        if is_sweep:
            sweep_id = wandb.sweep(
                sweep=helper.format_sweep_config(config), project=config.get("project")
            )

            wandb.agent(
                sweep_id=sweep_id, function=_run_training, count=config.get("sweep_counts")
            )
        else:
            _run_training(
            project=project,
            name=config.get("name"),
            config=helper.format_config(config),
            wandb_on=config.get("wandb_on"),
        )
            
        print(f"Level {t} trained.")

    print("Done.")


# main for sweep and single training
def _run_training(project=None, name=None, config=None, wandb_on=True):
    # TODO: config sweep ...
    if wandb_on:
        run = wandb.init(project=project, name=name, config=config)
        config = wandb.config
        # best instead of last value for metric
        wandb.define_metric("eval/acc", summary="max")

    model_cls = helper.string_to_object(config.get("model"))
    optimizer_cls = helper.string_to_object(config.get("optimizer"))

    level = config.get("level").split("/", 1)
    type_class = None
    if len(level) == 2:
        type_class = level[-1]

    start_time = datetime.fromtimestamp(
        time.time() if not wandb_on else run.start_time
    ).strftime("%Y%m%d_%H%M%S")
    id = "" if not wandb_on else "-" + run.id
    saving_name = (
        "-".join(level) + "-" + config.get("model") + "-" + start_time + id + ".pt"
    )

    torch.manual_seed(config.get("seed"))

    # TODO: testing gpu_kernel = None
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    trainloader, validloader, model, optimizer = prepare_train(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        transform=config.get("transform"),
        augment=config.get("augment"),
        dataset=config.get("dataset"),
        data_root=config.get("root_data"),
        level=level[0],
        type_class=type_class,
        selected_classes=config.get("selected_classes"),
        validation_size=config.get("validation_size"),
        batch_size=config.get("batch_size"),
        valid_batch_size=config.get("valid_batch_size"),
        learning_rate=config.get("learning_rate"),
        random_seed=config.get("seed"),
        is_regression=config.get("is_regression"),
    )

    trained_model = train(
        model=model,
        model_saving_path=config.get("root_model"),
        model_saving_name=saving_name,
        trainloader=trainloader,
        validloader=validloader,
        optimizer=optimizer,
        eval_metric=config.get("eval_metric"),
        device=device,
        epochs=config.get("epochs"),
        wandb_on=wandb_on,
        checkpoint_top_n=config.get("checkpoint_top_n", const.CHECKPOINT_DEFAULT_TOP_N),
        early_stop_thresh=config.get("early_stop_thresh", const.EARLY_STOPPING_DEFAULT),
        save_state=config.get("save_state", True),
        config=config,
    )

    # TODO: save best instead of last model (if checkpoint used)
    # TODO: save dict incl. config .. + model param (compare checkpoints)
    # model_path = save_model(trained_model, saving_name)
    # print(f'Model saved locally: {model_path}')

    if wandb_on:
        wandb.finish()

    return trained_model  # , model_path

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
    is_regression,
):
    train_data, valid_data = preprocessing.create_train_validation_datasets(
        data_root=data_root,
        dataset=dataset,
        selected_classes=selected_classes,
        validation_size=validation_size,
        general_transform=transform,
        augmentation=augment,
        random_state=random_seed,
        is_regression=is_regression,
        level=level,
        type_class=type_class,
    )

    # torch.save(valid_data, os.path.join(general_config.save_path, "valid_data.pt"))
    print(f"classes: {train_data.class_to_idx}")

    # TODO: loader in preprocessing?
    # TODO: weighted sampling on/off?
    class_counts = Counter(train_data.targets)
    sample_weights = [1 / class_counts[i] for i in train_data.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))

    trainloader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler
    )  # shuffle=True only if no sampler defined
    validloader = DataLoader(valid_data, batch_size=valid_batch_size)

    # load model
    if is_regression:
        num_classes = 1
    else:
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
    optimizer,
    eval_metric,
    device,
    epochs,
    wandb_on,
    checkpoint_top_n=const.CHECKPOINT_DEFAULT_TOP_N,
    early_stop_thresh=const.EARLY_STOPPING_DEFAULT,
    save_state=True,
    config=None,
):
    model.to(device)

    # TODO: decresing depending on metric
    checkpointer = checkpointing.CheckpointSaver(
        dirpath=model_saving_path,
        saving_name=model_saving_name,
        decreasing=False,
        config=config,
        dataset=validloader.dataset,
        top_n=checkpoint_top_n,
        early_stop_thresh=early_stop_thresh,
        save_state=save_state,
    )

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model,
            trainloader,
            optimizer,
            device,
            eval_metric=eval_metric,
        )

        val_loss, val_accuracy = validate_epoch(
            model,
            validloader,
            device,
            eval_metric,
        )

        # checkpoint saving with early stopping
        early_stop = checkpointer(
            model=model, epoch=epoch, metric_val=val_accuracy, optimizer=optimizer
        )

        if wandb_on:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_accuracy,  # TODO: metric not necessarily accuracy
                    "eval/loss": val_loss,
                    "eval/acc": val_accuracy,  # TODO: metric not necessarily accuracy
                }
            )

        print(
            f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}.. ",
            f"Train loss: {train_loss:.3f}.. ",
            f"Test loss: {val_loss:.3f}.. ",
            f"Train {eval_metric}: {train_accuracy:.3f}.. ",
            f"Test {eval_metric}: {val_accuracy:.3f}",
        )

        if early_stop:
            print(f"Early stopped training at epoch {epoch}")
            break

    print("Done.")

    return model


# train a single epoch
def train_epoch(model, dataloader, optimizer, device, eval_metric):
    model.train()
    criterion = model.criterion(reduction="sum")
    running_loss = 0.0
    eval_metric_value = 0

    for inputs, labels in dataloader:
        # helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        if isinstance(criterion, nn.MSELoss):
            outputs = outputs.flatten()
            labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # TODO: metric as function, metric_name as input argument

        if eval_metric == const.EVAL_METRIC_ACCURACY:
            if isinstance(criterion, nn.MSELoss):
                predictions = outputs.round()
            else:
                probs = model.get_class_probabilies(outputs)
                predictions = torch.argmax(probs, dim=1)
            eval_metric_value += (predictions == labels).sum().item()

        elif eval_metric == const.EVAL_METRIC_MSE:
            if not isinstance(criterion, nn.MSELoss):
                raise ValueError(
                    f"Criterion must be nn.MSELoss for eval_metric {eval_metric}"
                )
            eval_metric_value = running_loss
        else:
            raise ValueError(f"Unknown eval_metric: {eval_metric}")

    return running_loss / len(dataloader.sampler), eval_metric_value / len(
        dataloader.sampler
    )


# validate a single epoch
def validate_epoch(model, dataloader, device, eval_metric):
    model.eval()
    criterion = model.criterion(reduction="sum")
    running_loss = 0.0
    eval_metric_value = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)

            if isinstance(criterion, nn.MSELoss):
                outputs = outputs.flatten()
                labels = labels.float()
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            if eval_metric == const.EVAL_METRIC_ACCURACY:
                if isinstance(criterion, nn.MSELoss):
                    predictions = outputs.round()
                else:
                    probs = model.get_class_probabilies(outputs)
                    predictions = torch.argmax(probs, dim=1)
                eval_metric_value += (predictions == labels).sum().item()

            elif eval_metric == const.EVAL_METRIC_MSE:
                if not isinstance(criterion, nn.MSELoss):
                    raise ValueError(
                        f"Criterion must be nn.MSELoss for eval_metric {eval_metric}"
                    )
                eval_metric_value = running_loss
            else:
                raise ValueError(f"Unknown eval_metric: {eval_metric}")

    return running_loss / len(dataloader.sampler), eval_metric_value / len(
        dataloader.sampler
    )


# save model locally
def save_model(model, saving_path, saving_name):
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
    to_train_list = []
    if level == const.FLATTEN:
        to_train_list.append({"level": level, "selected_classes": selected_classes})
    elif level == const.SURFACE:
        to_train_list.append(
            {"level": level, "selected_classes": list(selected_classes.keys())}
        )
    elif level == const.SMOOTHNESS:
        for type_class in selected_classes.keys():
            to_train_list.append(
                {
                    "level": level + "/" + type_class,
                    "selected_classes": selected_classes[type_class],
                }
            )
    else:
        to_train_list.append({"level": level, "selected_classes": selected_classes})

    return to_train_list


def main():
    '''train image classifier
    
    command line args:
    - config: with
        - project
        - name
        - wandb_mode
        - config
        ...
    - sweep: (Optional) False (dafault) or True
    '''
    arg_parser = argparse.ArgumentParser(description='Model Training')
    arg_parser.add_argument('config', type=helper.dict_type, help='Required: configuration for training')
    arg_parser.add_argument('--sweep', type=bool, default=False, help='Optinal: Running a sweep or no sweep (default=False)')
    
    args = arg_parser.parse_args()

    run_training(args.config, args.sweep)

if __name__ == "__main__":
    main()

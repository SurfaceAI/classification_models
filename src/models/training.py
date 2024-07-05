import sys

sys.path.append(".")

import os
import time
from collections import Counter
from datetime import datetime

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import random

import wandb
from src import constants as const
from src.utils import checkpointing, helper, preprocessing
from multi_label import QWK
import argparse
import matplotlib.pyplot as plt



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
        # TODO: wandb best instead of last value for metric
        summary = "max" if config.get("eval_metric")==const.EVAL_METRIC_ACCURACY else "min"
        wandb.define_metric(f'eval/{config.get("eval_metric")}', summary=summary)
        # wandb.define_metric("eval/acc", summary="max")
        # wandb.define_metric("eval/mse", summary="min")
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

    helper.set_seed(config.get("seed"))

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
        clm=config.get("clm"),
        two_optimizers=config.get("two_optimizers"),
        max_class_size=config.get("max_class_size"),
    )

    trained_model = train(
        model=model,
        model_saving_path=config.get("root_model"),
        model_saving_name=saving_name,
        trainloader=trainloader,
        validloader=validloader,
        optimizer=optimizer,
        eval_metric=config.get("eval_metric"),
        clm = config.get("clm"), 
        two_optimizers=config.get("two_optimizers"),
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
    clm,
    two_optimizers, 
    max_class_size,
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
    # print(f"classes: {train_data.class_to_idx}")

    # load model
    if is_regression or clm:
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
    if two_optimizers:
        cls_params = helper.get_parameters_by_layer(model, 'CLM')
        rest_params = [param for name, param in model.named_parameters() if 'CLM' not in name]
        
        optimizer_clm = optimizer_cls(cls_params, lr=0.01)
        optimizer_model = optimizer_cls(rest_params, lr=learning_rate)
        optimizer = optimizer_clm, optimizer_model
    
    else:
        optimizer = optimizer_cls(optimizer_params, lr=learning_rate)

    # limit max class size
    if max_class_size is not None:
        # define indices with max number of class size
        indices = []
        class_counts = {}
        # TODO: randomize sample picking?
        for i, label in enumerate(train_data.targets):
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < max_class_size:
                indices.append(i)
                class_counts[label] += 1
            # stop if all classes are filled
            if all(count >= max_class_size for count in class_counts.values()):
                break

        # create a) (Subset with indices + WeightedRandomSampler) or b) (SubsetRandomSampler) (no weighting, if max class size larger than smallest class size!)
        # b) SubsetRandomSampler ? 
        #    Samples elements randomly from a given list of indices, without replacement.
        # a):
        train_data = Subset(train_data, indices)

        sample_weights = [1.0 / class_counts[label] for _, label in train_data]
    else:
        # TODO: loader in preprocessing?
        # TODO: weighted sampling on/off?
        class_counts = Counter(train_data.targets)
        sample_weights = [1.0 / class_counts[label] for label in train_data.targets]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))        

    trainloader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler
    )  # shuffle=True only if no sampler defined
    validloader = DataLoader(valid_data, batch_size=valid_batch_size)

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
    clm,
    two_optimizers,
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
        decreasing=True,
        config=config,
        dataset=validloader.dataset,
        top_n=checkpoint_top_n,
        early_stop_thresh=early_stop_thresh,
        save_state=save_state,
    )

    for epoch in range(epochs):
        train_loss, train_metric_value = train_epoch(
            model,
            trainloader,
            optimizer,
            device,
            eval_metric=eval_metric,
            clm=clm,
            two_optimizers=two_optimizers,
        )

        val_loss, val_metric_value = validate_epoch(
            model,
            validloader,
            device,
            eval_metric,
            clm=clm,
        )

        # checkpoint saving with early stopping
        early_stop = checkpointer(
            model=model, epoch=epoch, metric_val=val_loss, optimizer=optimizer
        )

        if wandb_on:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    f"train/{eval_metric}": train_metric_value,
                    "eval/loss": val_loss,
                    f"eval/{eval_metric}": val_metric_value,
                }
            )

        print(
            f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}.. ",
            f"Train loss: {train_loss:.3f}.. ",
            f"Test loss: {val_loss:.3f}.. ",
            f"Train {eval_metric}: {train_metric_value:.3f}.. ",
            f"Test {eval_metric}: {val_metric_value:.3f}",
        )

        if early_stop:
            print(f"Early stopped training at epoch {epoch}")
            break

    print("Done.")

    return model


# train a single epoch
def train_epoch(model, dataloader, optimizer, device, eval_metric, clm, two_optimizers):
    model.train()
    if clm:
        cost_matrix = QWK.make_cost_matrix(4)
        criterion = model.criterion(cost_matrix)
        #criterion = model.criterion(reduction="sum")
    else:
        criterion = model.criterion(reduction="sum")
        
    wandb.watch(model, log="all", log_freq=10)
    running_loss = 0.0
    eval_metric_value = 0

    gradients = []
    first_moments = []
    second_moments = []

    for inputs, labels in dataloader:
        # helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)
        
        if two_optimizers:
            optimizer_clm, optimizer_model = optimizer
            optimizer_clm.zero_grad()
            optimizer_model.zero_grad()
        
        else:
            optimizer.zero_grad()

        outputs = model.forward(inputs)
        if isinstance(criterion, nn.MSELoss):
            outputs = outputs.flatten()
            labels = labels.float()
        elif clm:
            loss = criterion(helper.to_one_hot_tensor(labels, 4), outputs)
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()
        
        # Print gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} gradient: {param.grad.norm()}")
                #print(f"{name} gradient values: {param.grad}")

        
        print(f"Thresholds before optimizer step: b: {model.CLM.thresholds_b.data}, a: {model.CLM.thresholds_a.data}")
        
        
        if two_optimizers:
            optimizer_clm.step()
            optimizer_model.step()
        else:
            optimizer.step()
        
        # Capture gradients and moments for the CLM layer
        # for name, param in model.named_parameters():
        #     if 'CLM' in name and param.grad is not None:
        #         gradients.append(param.grad.norm().item())
        #         if param in optimizer_clm.state:
        #             param_state = optimizer_clm.state[param]
        #             if 'exp_avg' in param_state and 'exp_avg_sq' in param_state:
        #                 first_moment = param_state['exp_avg'].norm().item()
        #                 second_moment = param_state['exp_avg_sq'].norm().item()
        #                 first_moments.append(first_moment)
        #                 second_moments.append(second_moment)
                
        #         # Print optimizer state for the parameter
        #         if param in optimizer_clm.state:
        #             param_state = optimizer_clm.state[param]
        #             print(f"{name} optimizer state: {param_state}")
        
        print(f"Thresholds after optimizer step: b: {model.CLM.thresholds_b.data}, a: {model.CLM.thresholds_a.data}")

        running_loss += loss.item()

        # TODO: metric as function, metric_name as input argument

        if eval_metric == const.EVAL_METRIC_ACCURACY:
            if isinstance(criterion, nn.MSELoss): # compare with is_regression for generalization?
                predictions = outputs.round()
            elif clm:
                predictions = torch.argmax(outputs, dim=1)
            else:
                probs = model.get_class_probabilies(outputs)
                predictions = torch.argmax(probs, dim=1)
            eval_metric_value += (predictions == labels).sum().item()

        elif eval_metric == const.EVAL_METRIC_MSE:
            if not isinstance(criterion, nn.MSELoss): # compare with is_regression for generalization?
                raise ValueError(
                    f"Criterion must be nn.MSELoss for eval_metric {eval_metric}"
                )
            eval_metric_value = running_loss
        else:
            raise ValueError(f"Unknown eval_metric: {eval_metric}")
        
        wandb.log(
        {
            "batch_loss": running_loss,
        }
        )
        #break
    
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 3, 1)
    # plt.plot(gradients, label="Gradients")
    # plt.title("Gradients of Last CLM Layer")
    # plt.xlabel("Batch")
    # plt.ylabel("Gradient Norm")
    # plt.legend()

    # plt.subplot(1, 3, 2)
    # plt.plot(first_moments, label="First Moment (m_t)")
    # plt.title("First Moment (m_t) of Last CLM Layer")
    # plt.xlabel("Batch")
    # plt.ylabel("First Moment Norm")
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.plot(second_moments, label="Second Moment (v_t)")
    # plt.title("Second Moment (v_t) of Last CLM Layer")
    # plt.xlabel("Batch")
    # plt.ylabel("Second Moment Norm")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return running_loss / len(dataloader.sampler), eval_metric_value / len(
        dataloader.sampler
    )


# validate a single epoch
def validate_epoch(model, dataloader, device, eval_metric, clm):
    model.eval()
    if clm:
        cost_matrix = QWK.make_cost_matrix(4)
        criterion = model.criterion(cost_matrix)
        # criterion = model.criterion(reduction="sum")
    else:
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
            
            if clm:
                loss = criterion(helper.to_one_hot_tensor(labels, 4), outputs)
            else:
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            if eval_metric == const.EVAL_METRIC_ACCURACY:
                if isinstance(criterion, nn.MSELoss):
                    predictions = outputs.round()
                elif clm:
                    predictions = torch.argmax(outputs, dim=1)
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
            
            #break

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

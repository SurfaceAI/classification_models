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

from sklearn.model_selection import StratifiedShuffleSplit

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from coral_pytorch.dataset import corn_label_from_logits

import pandas as pd
from sklearn.metrics import cohen_kappa_score




def run_training(config, is_sweep=False):
    os.environ["WANDB_MODE"] = config.get("wandb_mode", const.WANDB_MODE_OFF)

    project = config.get("project")
    if os.environ["WANDB_MODE"] == const.WANDB_MODE_OFF:
        project = "OFF_" + project
    
    to_train_list = extract_levels(
        level=config.get("level"), selected_classes=config.get("selected_classes")
    )
    
    all_cc_metrics = []  
    all_trainloaders = []
    all_validloaders = []
    
    for t in to_train_list:
        inner_config = {
            **config,
            **t,
        }
        
        if is_sweep:
            print("Sweep config:", inner_config)
            sweep_id = wandb.sweep(
                sweep=helper.format_sweep_config(inner_config), project=inner_config.get("project")
            )
    
            wandb.agent(
                sweep_id=sweep_id, function=_run_training, count=inner_config.get("sweep_counts")
            )
            
        else:
            if inner_config.get("hierarchy_method") == const.CC:
                _, all_epoch_metrics_df, trainloader, validloader = _run_training(
                    project=project,
                    name=inner_config.get("name"),
                    config=helper.format_config(inner_config),
                    wandb_on=inner_config.get("wandb_on"),
                )
                
                all_epoch_metrics_df['level'] = t['level']
                all_cc_metrics.append(all_epoch_metrics_df)
                
                all_trainloaders.append(trainloader)
                all_validloaders.append(validloader)

                        
            else:
                _run_training(
                project=project,
                name=inner_config.get("name"),
                config=helper.format_config(inner_config),
                wandb_on=inner_config.get("wandb_on"),
            )               
            
            print(f"Level {t} trained.")
    
    if config.get("hierarchy_method") == const.CC:
        # Combine all metrics from different levels into a single DataFrame
        combined_cc_metrics_df = pd.concat(all_cc_metrics, ignore_index=True)
        
        helper.compute_and_log_CC_metrics(combined_cc_metrics_df, all_trainloaders, all_validloaders, config.get("wandb_on"))
        
        if config.get("wandb_on"):
            wandb.finish()
    
    print("Done.")


# main for sweep and single training
def _run_training(project=None, name=None, config=None, wandb_on=True):
    # TODO: config sweep ...
    if wandb_on:
        run = wandb.init(project=project, name=name, config=config)
        config = wandb.config #TODO: why do we need this and why does it overwrite my loop of t loop
        # TODO: wandb best instead of last value for metric
        summary = "max" if config.get("eval_metric")==const.EVAL_METRIC_ACCURACY else "min"
        wandb.define_metric(f'eval/{config.get("eval_metric")}', summary=summary)
        # wandb.define_metric("val/acc", summary="max")
        # wandb.define_metric("val/mse", summary="min")
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
        head=config.get("head"),
        hierarchy_method=config.get("hierarchy_method"),
        max_class_size=config.get("max_class_size"),
        freeze_convs=config.get("freeze_convs"),
    )


    if config.get('level') == 'hierarchical':
        trained_model = train_hierarchical(
            model=model,
            model_saving_path=config.get("root_model"),
            model_saving_name=saving_name,
            trainloader=trainloader,
            validloader=validloader,
            optimizer=optimizer,
            head=config.get("head"),
            hierarchy_method=config.get("hierarchy_method"),
            gamma=config.get("gamma"),
            device=device,
            epochs=config.get("epochs"),
            wandb_on=wandb_on,
            checkpoint_top_n=config.get("checkpoint_top_n", const.CHECKPOINT_DEFAULT_TOP_N),
            early_stop_thresh=config.get("early_stop_thresh", const.EARLY_STOPPING_DEFAULT),
            save_state=config.get("save_state", True),
            lr_scheduler=config.get("lr_scheduler"),
            lw_modifier=config.get("lw_modifier"),
            config=config,
        )
        
        if wandb_on:
                wandb.finish()
            
        return trained_model
        
    else:
        if config.get("hierarchy_method") == const.CC:
            trained_model, all_epoch_metrics_df = train(
                model=model,
                model_saving_path=config.get("root_model"),
                model_saving_name=saving_name,
                trainloader=trainloader,
                validloader=validloader,
                optimizer=optimizer,
                eval_metric=config.get("eval_metric"),
                head = config.get("head"), 
                hierarchy_method=config.get("hierarchy_method"),
                gamma=config.get("gamma"),
                device=device,
                epochs=config.get("epochs"),
                wandb_on=wandb_on,
                checkpoint_top_n=config.get("checkpoint_top_n", const.CHECKPOINT_DEFAULT_TOP_N),
                early_stop_thresh=config.get("early_stop_thresh", const.EARLY_STOPPING_DEFAULT),
                save_state=config.get("save_state", True),
                config=config,
                lr_scheduler=config.get("lr_scheduler"),
            )
        
            
            return trained_model, all_epoch_metrics_df, trainloader, validloader
            
        else:
            trained_model = train(
                model=model,
                model_saving_path=config.get("root_model"),
                model_saving_name=saving_name,
                trainloader=trainloader,
                validloader=validloader,
                optimizer=optimizer,
                eval_metric=config.get("eval_metric"),
                head = config.get("head"), 
                hierarchy_method=config.get("hierarchy_method"),
                gamma = config.get("gamma"),
                device=device,
                epochs=config.get("epochs"),
                wandb_on=wandb_on,
                checkpoint_top_n=config.get("checkpoint_top_n", const.CHECKPOINT_DEFAULT_TOP_N),
                early_stop_thresh=config.get("early_stop_thresh", const.EARLY_STOPPING_DEFAULT),
                save_state=config.get("save_state", True),
                config=config,
                lr_scheduler=config.get("lr_scheduler"),
            )
            
            if wandb_on:
                wandb.finish()
            
            return trained_model
    

    # TODO: save best instead of last model (if checkpoint used)
    # TODO: save dict incl. config .. + model param (compare checkpoints)
    # model_path = save_model(trained_model, saving_name)
    # print(f'Model saved locally: {model_path}')

     # , model_path

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
    head,
    hierarchy_method,
    max_class_size,
    freeze_convs,
):
    
    train_data, valid_data = preprocessing.create_train_validation_datasets(
        data_root=data_root,
        dataset=dataset,
        selected_classes=selected_classes,
        validation_size=validation_size,
        general_transform=transform,
        augmentation=augment,
        random_state=random_seed,
        head=head,
        level=level,
        type_class=type_class,
    )

    # torch.save(valid_data, os.path.join(general_config.save_path, "valid_data.pt"))
    # print(f"classes: {train_data.class_to_idx}")
    
    helper.fix_seeds(random_seed)

    # Load images and labels
    # X, y, label_to_index, index_to_label = helper.load_images_and_labels(r'c:\Users\esthe\Documents\GitHub\classification_models\data\training\V12/annotated/asphalt', 
    #                                                                      (256,256), 
    #                                                                      ['excellent', 'good', 'intermediate', 'bad'])


    # sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_seed)
    # sss_splits = list(sss.split(X=X, y=y))
    # train_idx, test_idx = sss_splits[0]

    # X_trainval, X_test = X[train_idx], X[test_idx]
    # y_trainval, y_test = y[train_idx], y[test_idx]

    # sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)  
    # sss_splits_val = list(sss_val.split(X=X_trainval, y=y_trainval))
    # train_idx, val_idx = sss_splits_val[0]

    # X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    # y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # train_data = helper.CustomDataset(X_train, y_train, transform=transform)
    # valid_data = helper.CustomDataset(X_val, y_val, transform=transform)
    # test_data = helper.CustomDataset(X_test, y_test, transform=transform)

    # sampler = None
    # if max_class_size is not None:
    #     class_counts = Counter(train_data.labels)
    #     indices = []
    #     for i, label in enumerate(train_data.labels):
    #         if class_counts[label] > max_class_size:
    #             continue
    #         indices.append(i)
    #         class_counts[label] -= 1
    #     train_data = Subset(train_data, indices)
    #     sample_weights = [1.0 / class_counts[label] for _, label in train_data]
    #     sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))

    # trainloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    # validloader = DataLoader(valid_data, batch_size=valid_batch_size)

    # load model
    if level == const.HIERARCHICAL: #TODO: adapt num_coarse_classes automatically
        num_coarse_classes = 5

    #head for fine classes hierarchical models or classifier chain quality part   
    if head == const.REGRESSION:
        num_classes = 1
        
    if level == const.ASPHALT:
        num_classes = 4
        
    else:
        num_classes = 18
        #num_classes = sum(len(selected_class) for selected_class in selected_classes.values())
        

    # instanciate model with number of classes
    if level == const.HIERARCHICAL:
        model = model_cls(num_coarse_classes, num_classes, head, hierarchy_method)
    else:
        model = model_cls(num_classes, head)

    # Unfreeze parameters
    if freeze_convs:
        for param in model.features.parameters():
            param.requires_grad = False
            
    else:
        for param in model.features.parameters():
            param.requires_grad = True
        
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    optimizer_layers = None
    if hasattr(model, "get_optimizer_layers") and callable(model.get_optimizer_layers):
        optimizer_layers = model.get_optimizer_layers()

    # setup optimizer
    if optimizer_layers is None:
        #optimizer_params = model.parameters()
        optimizer_params = model.parameters()
    else:
        optimizer_params = []
        for layer in optimizer_layers:
            optimizer_params += [p for p in layer.parameters()]

    #print(f"{len(optimizer_params)} optimizer params")

    for name, param in model.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")
        
    # Count parameters and print
    total_params, trainable_params, non_trainable_params = helper.count_parameters(model)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Non-trainable params: {non_trainable_params}")


    # set parameters to optimize
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
            
        indices_valid = []
        class_counts = {}
        for i, label in enumerate(valid_data.targets):
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < max_class_size:
                indices_valid.append(i)
                class_counts[label] += 1
            # stop if all classes are filled
            if all(count >= max_class_size for count in class_counts.values()):
                break

        # create a) (Subset with indices + WeightedRandomSampler) or b) (SubsetRandomSampler) (no weighting, if max class size larger than smallest class size!)
        # b) SubsetRandomSampler ? 
        #    Samples elements randomly from a given list of indices, without replacement.
        # a):
        train_data = Subset(train_data, indices)
        train_data.dataset.targets = [train_data.dataset.targets[i] for i in indices]
        
        valid_data = Subset(valid_data, indices_valid)
        valid_data.dataset.targets = [valid_data.dataset.targets[i] for i in indices_valid]

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
    head,
    hierarchy_method,
    gamma,
    device,
    epochs,
    wandb_on,
    checkpoint_top_n=const.CHECKPOINT_DEFAULT_TOP_N,
    early_stop_thresh=const.EARLY_STOPPING_DEFAULT,
    save_state=True,
    lr_scheduler=None,
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
    
    # if wandb_on:
    #     wandb.watch(model, log_freq=27)
    if lr_scheduler:
        scheduler = StepLR(optimizer, step_size=4, gamma=gamma) 
        #scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)


    if hierarchy_method == const.CC:
        all_epoch_metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_correct', 'train_correct_one_off', 
                                                    'train_mse', 'train_mae', 'val_loss', 'val_correct', 
                                                    'val_correct_one_off', 'val_mse', 'val_mae'])

    for epoch in range(epochs):
        #This is the case where we train only one model, e.g. for asphalt or the surface model only
        if eval_metric == const.EVAL_METRIC_ALL and hierarchy_method == const.FLATTEN:
            train_loss, accuracy, accuracy_one_off, mse, mae, qwk  = train_epoch(
                model,
                trainloader,
                optimizer,
                device,
                eval_metric,
                head,
                hierarchy_method,
                wandb_on=wandb_on,
            )

            val_loss, val_accuracy, val_accuracy_one_off, val_mse, val_mae, val_qwk  = validate_epoch(
                model,
                validloader,
                device,
                eval_metric,
                head,
                hierarchy_method,
            )
        
        #This is the case for the Classifier Chain, or training all quality models sequentially     
        elif eval_metric == const.EVAL_METRIC_ALL and hierarchy_method == const.CC:
        
            epoch_metrics_df  = train_epoch(
                model,
                trainloader,
                optimizer,
                device,
                eval_metric,
                head,
                hierarchy_method,
                wandb_on=wandb_on,
            )

            val_epoch_metrics_df  = validate_epoch(
                model,
                validloader,
                device,
                eval_metric,
                head,
                hierarchy_method,
            )
            
            val_loss = epoch_metrics_df['loss'].item()
            
            #TODO: something wrong with epoch_metrics_df['loss'] more than one value 
            epoch_data = pd.DataFrame([{
                'epoch': epoch + 1,
                'train_loss': epoch_metrics_df['loss'].item(),  # Convert Series to scalar
                'train_correct': epoch_metrics_df['correct'].item(),
                'train_correct_one_off': epoch_metrics_df['correct_one_off'].item(),
                'train_mse': epoch_metrics_df['mse'].item(),
                'train_mae': epoch_metrics_df['mae'].item(),
                'val_loss': val_epoch_metrics_df['loss'].item(),
                'val_correct': val_epoch_metrics_df['correct'].item(),
                'val_correct_one_off': val_epoch_metrics_df['correct_one_off'].item(),
                'val_mse': val_epoch_metrics_df['mse'].item(),
                'val_mae': val_epoch_metrics_df['mae'].item()
            }])

            # Use pd.concat to append the new row to the existing DataFrame
            all_epoch_metrics_df = pd.concat([all_epoch_metrics_df, epoch_data], ignore_index=True)
                        
        else:
            train_loss, train_metric_value = train_epoch(
                model,
                trainloader,
                optimizer,
                device,
                eval_metric,
                head,
                hierarchy_method,
                wandb_on=wandb_on,
            )

            val_loss, eval_metric_value = validate_epoch(
                model,
                validloader,
                device,
                eval_metric,
                head,
                hierarchy_method,
            )
        
        if lr_scheduler:
            scheduler.step()
        
        #helper.save_gradient_plots(epoch, gradients, first_moments, second_moments)

        # checkpoint saving with early stopping #TODO: update for CC training
        early_stop = checkpointer( 
            model=model, epoch=epoch, metric_val=val_loss, optimizer=optimizer
        )

        if wandb_on:
            if eval_metric == const.EVAL_METRIC_ALL and hierarchy_method == const.FLATTEN:
                
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/fine/loss": train_loss,
                        "train/accuracy/fine": accuracy,
                        "train/accuracy/fine_1_off": accuracy_one_off,
                        "train/mse/fine": mse,
                        "train/mae/fine": mae, #
                        "train/qwk/fine": qwk,
                        "eval/fine/loss": val_loss,
                        "eval/accuracy/fine": val_accuracy,
                        "eval/accuracy/fine_1_off":val_accuracy_one_off,
                        "eval/mse/fine": val_mse,
                        "eval/mae/fine": val_mae,
                        "eval/qwk/fine": val_qwk,
                    }
                )
                
                print(
                    f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}.. ",
                    f"Train loss: {train_loss:.3f}.. ",
                    f"Test loss: {val_loss:.3f}.. ",
                    f"Train accuracy: {accuracy:.3f}.. ",
                    f"Test accuracy: {val_accuracy:.3f}",
                    f"Train accuracy_1_off: {accuracy_one_off:.3f}.. ",
                    f"Test accuracy_1_off: {val_accuracy_one_off:.3f}",
                    f"Train MSE: {mse:.3f}.. ",
                    f"Test MSE: {mae:.3f}",
                    f"Train MAE: {mae:.3f}.. ",
                    f"Test MAE: {mae:.3f}",
                    )
                
            elif hierarchy_method == const.CC:
                pass
            
            else:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        f"train/{eval_metric}": train_metric_value,
                        "eval/loss": val_loss,
                        f"eval/{eval_metric}": eval_metric_value,
                    }
                )
                
                print(
                    f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}.. ",
                    f"Train loss: {train_loss:.3f}.. ",
                    f"Test loss: {val_loss:.3f}.. ",
                    f"Train {eval_metric}: {train_metric_value:.3f}.. ",
                    f"Test {eval_metric}: {eval_metric_value:.3f}",
                    f"Learning Rate: {scheduler.get_last_lr()[0]}"
                    )

        # if early_stop:
        #     print(f"Early stopped training at epoch {epoch}")
        #     break

    print("Done.")

    return model, all_epoch_metrics_df

def train_hierarchical(
    model,
    model_saving_path,
    model_saving_name,
    trainloader,
    validloader,
    optimizer,
    head,
    hierarchy_method,
    gamma,
    device,
    epochs,
    wandb_on,
    checkpoint_top_n=const.CHECKPOINT_DEFAULT_TOP_N,
    early_stop_thresh=const.EARLY_STOPPING_DEFAULT,
    save_state=True,
    lr_scheduler=None,
    lw_modifier=None,
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
    
    # if wandb_on:
    #     wandb.watch(model, log_freq=27)
    if lr_scheduler:
        scheduler = StepLR(optimizer, step_size=4, gamma=gamma) 
    
    if lw_modifier:
        alpha = torch.tensor(0.98)
        beta = torch.tensor(0.02)
        if model == const.GHCNN:
            loss_weights_modifier = helper.LossWeightsModifier_GH(alpha, beta)
        else:
            loss_weights_modifier = helper.LossWeightsModifier(alpha, beta)
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(epochs):
        (epoch_loss, 
         epoch_coarse_accuracy, 
         epoch_fine_accuracy, 
         epoch_fine_accuracy_one_off,
         epoch_fine_mse,
         epoch_fine_mae,
         epoch_fine_qwk,
         epoch_fine_hv,
         coarse_epoch_loss, 
         fine_epoch_loss) = train_epoch_hierarchical(
            model, 
            trainloader, 
            optimizer, 
            device, 
            head, 
            hierarchy_method,
            lw_modifier,
            wandb_on,
            epoch,
            alpha=alpha, #TODO: save within model
            beta=beta,
        )

        (val_epoch_loss, 
         val_epoch_coarse_accuracy, 
         val_epoch_fine_accuracy, 
         val_epoch_fine_accuracy_one_off, 
         val_epoch_fine_mse,
         val_epoch_fine_mae,
         val_epoch_fine_qwk,
         val_epoch_fine_hv,
         val_coarse_epoch_loss, 
         val_fine_epoch_loss) = validate_epoch_hierarchical(
            model, 
            validloader, 
            device, 
            head, 
            hierarchy_method,
            lw_modifier,
            epoch,
            alpha=alpha, #TODO: save within model
            beta=beta,
            #wandb_on
        )
        
        if lr_scheduler:
            scheduler.step()
        
        #helper.save_gradient_plots(epoch, gradients, first_moments, second_moments)

        # checkpoint saving with early stopping
        early_stop = checkpointer(
            model=model, epoch=epoch, metric_val=val_epoch_loss, optimizer=optimizer
        )

        if wandb_on:
            if lr_scheduler:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": epoch_loss,
                        "train/coarse/loss": coarse_epoch_loss,
                        "train/fine/loss": fine_epoch_loss,
                        "train/accuracy/coarse": epoch_coarse_accuracy,
                        "train/accuracy/fine": epoch_fine_accuracy, 
                        "train/accuracy/fine_1_off": epoch_fine_accuracy_one_off,
                        "train/mse/fine":epoch_fine_mse,
                        "train/mae/fine":epoch_fine_mae,
                        "train/qwk/fine":epoch_fine_qwk,
                        "train/hv/fine":epoch_fine_hv,
                        "eval/loss": val_epoch_loss,
                        "eval/coarse/loss": val_coarse_epoch_loss,
                        "eval/fine/loss": val_fine_epoch_loss,
                        "eval/accuracy/coarse": val_epoch_coarse_accuracy,
                        "eval/accuracy/fine": val_epoch_fine_accuracy,
                        "eval/accuracy/fine_1_off": val_epoch_fine_accuracy_one_off,
                        "eval/mse/fine":val_epoch_fine_mse,
                        "eval/mae/fine":val_epoch_fine_mae,
                        "eval/qwk/fine":val_epoch_fine_qwk,
                        "eval/hv/fine":val_epoch_fine_hv,
                        "trainable_params": trainable_params,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
                
                print(f"""
                    Epoch: {epoch+1}:,
                    Train loss: {epoch_loss:.3f},
                    Coarse train loss: {coarse_epoch_loss:.3f},
                    Fine train loss: {fine_epoch_loss:.3f}, 
                    Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
                    Train fine accuracy: {epoch_fine_accuracy:.3f}%,
                    Train fine 1-off accuracy: {epoch_fine_accuracy_one_off:.3f}%,
                    Validation loss: {val_epoch_loss:.3f}, 
                    Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
                    Validation fine accuracy: {val_epoch_fine_accuracy:.3f}%, 
                    Validation fine 1-off accuracy: {val_epoch_fine_accuracy_one_off:.3f}%
                    Learning_rate: {scheduler.get_last_lr()[0]}
                            
                    """)
            else:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": epoch_loss,
                        "train/coarse/loss": coarse_epoch_loss,
                        "train/fine/loss": fine_epoch_loss,
                        "train/accuracy/coarse": epoch_coarse_accuracy,
                        "train/accuracy/fine": epoch_fine_accuracy, 
                        "train/accuracy/fine_1_off": epoch_fine_accuracy_one_off,
                        "train/mse/fine":epoch_fine_mse,
                        "train/mae/fine":epoch_fine_mae,
                        "train/qwk/fine":epoch_fine_qwk,
                        "train/hv/fine":epoch_fine_hv,
                        "eval/loss": val_epoch_loss,
                        "eval/coarse/loss": val_coarse_epoch_loss,
                        "eval/fine/loss": val_fine_epoch_loss,
                        "eval/accuracy/coarse": val_epoch_coarse_accuracy,
                        "eval/accuracy/fine": val_epoch_fine_accuracy,
                        "eval/accuracy/fine_1_off": val_epoch_fine_accuracy_one_off,
                        "eval/mse/fine":val_epoch_fine_mse,
                        "eval/mae/fine":val_epoch_fine_mae,
                        "eval/qwk/fine":val_epoch_fine_qwk,
                        "eval/hv/fine":val_epoch_fine_hv,
                        "trainable_params": trainable_params,
                    }
                )
                

                print(f"""
                Epoch: {epoch+1}:,
                Train loss: {epoch_loss:.3f},
                Coarse train loss: {coarse_epoch_loss:.3f},
                Fine train loss: {fine_epoch_loss:.3f}, 
                Train coarse accuracy: {epoch_coarse_accuracy:.3f}%, 
                Train fine accuracy: {epoch_fine_accuracy:.3f}%,
                Train fine 1-off accuracy: {epoch_fine_accuracy_one_off:.3f}%,
                Validation loss: {val_epoch_loss:.3f}, 
                Validation coarse accuracy: {val_epoch_coarse_accuracy:.3f}%, 
                Validation fine accuracy: {val_epoch_fine_accuracy:.3f}%, 
                Validation fine 1-off accuracy: {val_epoch_fine_accuracy_one_off:.3f}%,
                        
                """)

        if lw_modifier:
            alpha, beta = loss_weights_modifier.on_epoch_end(epoch)
        
        if early_stop:
            print(f"Early stopped training at epoch {epoch}")
            break

    print("Done.")

    return model


# train a single epoch
def train_epoch(model, dataloader, optimizer, device, eval_metric, head, hierarchy_method, wandb_on):
    model.train()
    
    if head == const.CORN:
        criterion = model.criterion
    #corn_loss has no reduction parameter
    elif head == const.CLASSIFICATION_QWK or head == const.CLM_QWK:
        criterion = model.criterion(num_classes=model.num_classes, mode='quadratic')
    else:
        criterion = model.criterion(reduction="sum")
        
    running_loss = 0.0
    
    # if eval_metric == const.EVAL_METRIC_ALL and hierarchy_method == const.CC:
    #     epoch_metrics_df = {
    #         'loss': 0.0,
    #         'correct': 0,
    #         'correct_one_off': 0,
    #         'mse': 0.0,
    #         'mae': 0.0
    #     }
        
    if eval_metric == const.EVAL_METRIC_ALL:
        correct = 0
        correct_one_off = 0
        mse = 0
        mae = 0
        qwk = 0
            
    else:
        eval_metric_value = 0
    # gradients = []
    # first_moments = []
    # second_moments = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # helper.multi_imshow(inputs, labels)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        
        if head == 'regression':
            outputs = outputs.flatten()
            labels = labels.float()
            loss = criterion(outputs, labels)
        #loss = criterion(helper.to_one_hot_tensor(labels, 4), outputs) Todo: for QWK
        elif head == 'clm':
            loss = criterion(torch.log(outputs + 1e-9), labels)
        elif head == 'corn':
            loss = criterion(outputs, labels, model.num_classes) #TODO: numclasses
        else:
            loss = criterion(outputs, labels)
            
        loss.backward()

        optimizer.step()
  
        running_loss += loss.item()
        
        #I put these together as this is what I need to compare the results with the hierarchical models
        if eval_metric == const.EVAL_METRIC_ALL:
            correct_item, correct_one_off_item, mse_item, mae_item, qwk_item = helper.compute_all_metrics(outputs, labels, head, model)
            correct += correct_item
            correct_one_off += correct_one_off_item
            mse += mse_item
            mae += mae_item
            qwk += qwk_item
        
            if batch_idx == 0:
                break

        # TODO: metric as function, metric_name as input argument
        else:
            if eval_metric == const.EVAL_METRIC_ACCURACY:
                if head == 'regression': # compare with is_regression for generalization?
                    predictions = outputs.round()
                elif head == 'clm':
                    predictions = torch.argmax(outputs, dim=1)
                elif head == 'corn':
                    predictions = corn_label_from_logits(outputs.long())    
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
            
            #break
            
            
    if eval_metric == const.EVAL_METRIC_ALL:
        if hierarchy_method == const.CC:           
            
            epoch_metrics_df = pd.DataFrame({
                'correct': [correct],
                'correct_one_off': [correct_one_off],
                'mse': [mse],  #
                'mae': [mae],  
                'loss': [running_loss], 
                'qwk': [qwk], #TODO: I added this but have to continue adding qwk in the other parts
            }) 
            return epoch_metrics_df
        
        else:
            epoch_accuracy = 100 * correct / len(dataloader.sampler)
            epoch_accuracy_one_off = 100 * correct_one_off / len(dataloader.sampler)
            epoch_mse = mse / len(dataloader.sampler)
            epoch_mae = mae / len(dataloader.sampler)
            epoch_loss = running_loss / len(dataloader.sampler)
            epoch_qwk = qwk / len(dataloader)

            return epoch_loss, epoch_accuracy, epoch_accuracy_one_off, epoch_mse, epoch_mae, epoch_qwk
    
    else:
        return running_loss / len(dataloader.sampler), eval_metric_value / len(
            dataloader.sampler), 
    #gradients, first_moments, second_moments


# validate a single epoch
def validate_epoch(model, dataloader, device, eval_metric, head, hierarchy_method):
    model.eval()
     
    if head == const.CORN:
        criterion = model.criterion
    elif head == const.CLASSIFICATION_QWK or head == const.CLM_QWK:
        criterion = model.criterion(num_classes=model.num_classes, mode='quadratic')
    #corn_loss has no reduction parameter
    else:
        criterion = model.criterion(reduction="sum")
    
    eval_running_loss = 0.0
    
    if eval_metric == const.EVAL_METRIC_ALL:
        eval_correct = 0
        eval_correct_one_off = 0
        eval_mse = 0
        eval_mae = 0
        eval_qwk = 0
            
    else:
        eval_metric_value = 0   
        

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            
            if head == 'regression':
                outputs = outputs.flatten()
                labels = labels.float()
                loss = criterion(outputs, labels)
            #loss = criterion(helper.to_one_hot_tensor(labels, 4), outputs) Todo: for QWK
            elif head == 'clm':
                loss = criterion(torch.log(outputs + 1e-9), labels)
            elif head == 'corn':
                loss = criterion(outputs, labels, model.num_classes) #TODO: numclasses
            else:
                loss = criterion(outputs, labels)

            eval_running_loss += loss.item()
            
            if eval_metric == const.EVAL_METRIC_ALL:
                eval_correct_item, eval_correct_one_off_item, eval_mse_item, eval_mae_item, eval_qwk_item = helper.compute_all_metrics(outputs, labels, head, model)
                eval_correct += eval_correct_item
                eval_correct_one_off += eval_correct_one_off_item
                eval_mse += eval_mse_item
                eval_mae += eval_mae_item
                eval_qwk += eval_qwk_item
                
                if batch_idx == 0:
                    break
                
                #break
                
            else:
                if eval_metric == const.EVAL_METRIC_ACCURACY:
                    if head == 'regression': # compare with is_regression for generalization?
                        predictions = outputs.round()
                    elif head == 'clm':
                        predictions = torch.argmax(outputs, dim=1)
                    elif head == 'corn':
                        predictions = corn_label_from_logits(outputs.long())    
                    else:
                        probs = model.get_class_probabilies(outputs)
                        predictions = torch.argmax(probs, dim=1)
                    eval_metric_value += (predictions == labels).sum().item()

                elif eval_metric == const.EVAL_METRIC_MSE:
                    if not isinstance(criterion, nn.MSELoss):
                        raise ValueError(
                            f"Criterion must be nn.MSELoss for eval_metric {eval_metric}"
                        )
                    eval_metric_value = eval_running_loss
                else:
                    raise ValueError(f"Unknown eval_metric: {eval_metric}")
            
            #break
        
        if eval_metric == const.EVAL_METRIC_ALL:
            
            if hierarchy_method == const.CC:
                
                epoch_metrics_df = pd.DataFrame({
                    'correct': [eval_correct],
                    'correct_one_off': [eval_correct_one_off],
                    'mse': [eval_mse],  
                    'mae': [eval_mae],  
                    'loss': [eval_running_loss],  
                    'qwk': [eval_qwk],
                }) 
             
                return epoch_metrics_df
                
            else:
                val_epoch_loss = eval_running_loss / len(dataloader.sampler)
                epoch_eval_accuracy = 100 * eval_correct / len(dataloader.sampler)
                epoch_eval_accuracy_one_off = 100 * eval_correct_one_off / len(dataloader.sampler)
                epoch_eval_mse = eval_mse / len(dataloader.sampler)
                epoch_eval_mae = eval_mae / len(dataloader.sampler)
                epoch_eval_qwk = eval_qwk / len(dataloader)
                return val_epoch_loss, epoch_eval_accuracy, epoch_eval_accuracy_one_off, epoch_eval_mse, epoch_eval_mae, epoch_eval_qwk,
            
        else:
            return eval_running_loss / len(dataloader.sampler), eval_metric_value / len(
                dataloader.sampler
)

def train_epoch_hierarchical(model, dataloader, optimizer, device, head, hierarchy_method, lw_modifier, wandb_on, epoch, alpha, beta,):
    model.train()
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    if head == 'corn':
        fine_criterion = model.fine_criterion
    elif head == const.CLASSIFICATION_QWK or head == const.CLM_QWK:
        fine_criterion = model.fine_criterion(num_classes=model.num_classes, mode='quadratic')
    else:
        fine_criterion = model.fine_criterion(reduction="sum")
        
    running_loss = 0.0
    coarse_loss_total = 0.0
    fine_loss_total = 0.0
    
    coarse_correct = 0
    fine_correct = 0
    fine_correct_one_off = 0
    
    fine_mse = 0
    fine_mae = 0
    fine_qwk = 0
    fine_hv = 0

    for batch_idx, (inputs, fine_labels) in enumerate(dataloader):
        # helper.multi_imshow(inputs, labels)

        inputs, fine_labels = inputs.to(device), fine_labels.to(device)

        optimizer.zero_grad()
        
        coarse_labels = helper.parent[fine_labels]
        coarse_one_hot = helper.to_one_hot_tensor(coarse_labels, model.num_c).to(device)
        
        model_inputs = (inputs, coarse_one_hot)
        
        if model == const.GHCNN:
            raw_coarse, raw_fine = model.forward(inputs)
            
            #3 different training phases
            if epoch < 0.15 * 15:
                coarse_output, fine_output = raw_coarse, raw_fine
                
            elif 0.15 * 15 <= epoch < 0.25 * 15: 
                coarse_output, fine_output = model.teacher_forcing(raw_coarse, raw_fine, coarse_one_hot)
                
            else:
                coarse_output, fine_output = model.bayesian_adjustment(raw_coarse, raw_fine)
        
        else:
            coarse_output, fine_output = model.forward(model_inputs)
            
        coarse_loss = coarse_criterion(coarse_output, coarse_labels)
        fine_loss = helper.compute_fine_losses(model, fine_criterion, fine_output, fine_labels, device, coarse_labels, hierarchy_method, head)                
       
        #GH-CNN training phases
        if model == const.GHCNN:
            if lw_modifier:
                loss_h = torch.sum(alpha * coarse_loss + beta * fine_loss)
            else:
                loss_h = coarse_loss + fine_loss
            
            #coarse only, weights should be (1,0)
            if epoch < 0.15 * 15:
                loss = loss_h   
            
            #teacher forcing 
            elif 0.15 * 15 <= epoch < 0.25 * 15:
                loss = loss_h 
                
            #added calculating the loss_v (greatest error on a prediction where coarse and subclass prediction dont match)
            else:
                try:
                    fine_probs = model.get_class_probabilies(fine_output)
                    fine_predictions = torch.argmax(fine_probs, dim=1)
                    mismatched_indices = (coarse_predictions != helper.parent[fine_predictions])
                    max_mismatched_coarse_loss = coarse_loss[mismatched_indices].max()
                    max_mismatched_fine_loss = fine_loss[mismatched_indices].max()
                    loss_v = max(max_mismatched_coarse_loss, max_mismatched_fine_loss)
                    loss = loss_h + loss_v
                except IndexError as e:
                    print(f"IndexError encountered: {e}. Skipping this iteration and continuing.")
                    loss = loss_h  # Optionally, handle the loss in some other way or set it to a default
                except Exception as e:
                    print(f"An error occurred: {e}. Skipping this iteration and continuing.")
                    loss = loss_h  # Optionally, handle the loss in some other way or set it to a default
          
        #All other hierarchical models phases      
        else:
            if lw_modifier:
                loss = alpha * coarse_loss + beta * fine_loss
            else:
                loss = coarse_loss + fine_loss  #weighted loss functions for different levels
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        coarse_loss_total += coarse_loss.item()
        if head == const.CLASSIFICATION_QWK or head == const.CLM_QWK:
            fine_loss_total += fine_loss
        else:
            fine_loss_total += fine_loss.item()
    
        coarse_probs = model.get_class_probabilies(coarse_output)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        coarse_correct += (coarse_predictions == coarse_labels).sum().item()

        # TODO: metric as function, metric_name as input argument

        if head == const.CLASSIFICATION or head == const.CLASSIFICATION_QWK:
            fine_output = model.get_class_probabilies(fine_output)
            
        (fine_correct_item, 
        fine_correct_one_off_item, 
        fine_mse_item, 
        fine_mae_item, 
        fine_qwk_item, 
        fine_hv_item, 
        ) = helper.compute_fine_metrics_hierarchical(fine_output, fine_labels, coarse_labels, hierarchy_method, head)
        
        fine_correct += fine_correct_item
        fine_correct_one_off += fine_correct_one_off_item
        fine_mse += fine_mse_item
        fine_mae += fine_mae_item
        fine_qwk += fine_qwk_item
        fine_hv += fine_hv_item
        
        #break
    
    epoch_loss = running_loss /  len(dataloader.sampler)
    epoch_coarse_accuracy = 100 * coarse_correct / len(dataloader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(dataloader.sampler)
    epoch_fine_accuracy_one_off = 100 * fine_correct_one_off / len(dataloader.sampler)
    epoch_mse = fine_mse / len(dataloader.sampler)
    epoch_mae = fine_mae / len(dataloader.sampler)
    epoch_qwk = fine_qwk / len(dataloader)
    epoch_hv = fine_hv / len(dataloader.sampler)
    
    coarse_epoch_loss = coarse_loss_total / len(dataloader.sampler)
    fine_epoch_loss = fine_loss_total / len(dataloader.sampler)
        
        
    return epoch_loss, epoch_coarse_accuracy, epoch_fine_accuracy, epoch_fine_accuracy_one_off, epoch_mse, epoch_mae, epoch_qwk, epoch_hv, coarse_epoch_loss, fine_epoch_loss

def validate_epoch_hierarchical(model, dataloader, device, head, hierarchy_method, lw_modifier, epoch, alpha, beta):
    model.eval()
    
    coarse_criterion = model.coarse_criterion(reduction="sum")
    if head == 'corn':
        fine_criterion = model.fine_criterion
    elif head == const.CLASSIFICATION_QWK or head == const.CLM_QWK:
        fine_criterion = model.fine_criterion(num_classes=model.num_classes, mode='quadratic')
    else:
        fine_criterion = model.fine_criterion(reduction="sum")
    
    val_running_loss = 0.0
    val_coarse_loss_total = 0.0
    val_fine_loss_total = 0.0
    
    val_coarse_correct = 0
    val_fine_correct = 0
    val_fine_correct_one_off = 0
    
    val_fine_mse = 0
    val_fine_mae = 0
    val_fine_qwk = 0
    val_fine_hv = 0

    for batch_idx, (inputs, fine_labels) in enumerate(dataloader):
        # helper.multi_imshow(inputs, labels)

        inputs, fine_labels = inputs.to(device), fine_labels.to(device)
        
        coarse_labels = helper.parent[fine_labels]
        coarse_one_hot = helper.to_one_hot_tensor(coarse_labels, model.num_c).to(device)
        
        model_inputs = (inputs, coarse_one_hot)
        coarse_output, fine_output = model.forward(model_inputs)
        
        coarse_loss = coarse_criterion(coarse_output, coarse_labels)
        val_coarse_probs = model.get_class_probabilies(coarse_output)
        val_coarse_predictions = torch.argmax(val_coarse_probs, dim=1)
        val_coarse_correct += (val_coarse_predictions == coarse_labels).sum().item()
              
        fine_loss = helper.compute_fine_losses(model, fine_criterion, fine_output, fine_labels, device, val_coarse_predictions, hierarchy_method, head)        
           
        if lw_modifier:
            loss = alpha * coarse_loss + beta * fine_loss
        else:
            loss = coarse_loss + fine_loss  #weighted loss functions for different levels
        
        val_running_loss += loss.item()
        val_coarse_loss_total += coarse_loss.item()
        if head == const.CLASSIFICATION_QWK or head == const.CLM_QWK:
            val_fine_loss_total += fine_loss
        else:
            val_fine_loss_total += fine_loss.item()

        if head == const.CLASSIFICATION or head == const.CLASSIFICATION_QWK:
            fine_output = model.get_class_probabilies(fine_output)
            
        (val_fine_correct_item, 
        val_fine_correct_one_off_item, 
        val_fine_mse_item, 
        val_fine_mae_item, 
        val_fine_qwk_item,
        val_fine_hv_item) = helper.compute_fine_metrics_hierarchical(fine_output, fine_labels, val_coarse_predictions, hierarchy_method, head)
        
        val_fine_correct += val_fine_correct_item
        val_fine_correct_one_off += val_fine_correct_one_off_item
        val_fine_mse += val_fine_mse_item
        val_fine_mae += val_fine_mae_item
        val_fine_qwk += val_fine_qwk_item
        val_fine_hv += val_fine_hv_item
        
        #break

    val_epoch_loss = val_running_loss /  len(dataloader.sampler)
    val_epoch_coarse_accuracy = 100 * val_coarse_correct / len(dataloader.sampler)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(dataloader.sampler)
    val_epoch_fine_accuracy_one_off = 100 * val_fine_correct_one_off / len(dataloader.sampler)
    val_epoch_fine_mse = val_fine_mse /  len(dataloader.sampler)
    val_epoch_fine_mae = val_fine_mae /  len(dataloader.sampler)
    val_epoch_fine_qwk = val_fine_qwk /  len(dataloader)
    val_epoch_fine_hv = val_fine_hv /  len(dataloader.sampler)
    
    val_coarse_epoch_loss = val_coarse_loss_total / len(dataloader.sampler)
    val_fine_epoch_loss = val_fine_loss_total / len(dataloader.sampler)

    return val_epoch_loss, val_epoch_coarse_accuracy, val_epoch_fine_accuracy, val_epoch_fine_accuracy_one_off, val_epoch_fine_mse, val_epoch_fine_mae, val_epoch_fine_qwk, val_epoch_fine_hv, val_coarse_epoch_loss, val_fine_epoch_loss

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

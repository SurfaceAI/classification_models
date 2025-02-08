import sys

sys.path.append(".")

import os
import time
from collections import Counter
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import random
from tqdm import tqdm

import wandb
from src import constants as const
from src.utils import checkpointing, helper, preprocessing
import argparse
from src.architecture import c_cnn


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
                sweep_id=sweep_id,
                function=_run_training,
                count=config.get("sweep_counts"),
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
        # summary = "max" if config.get("eval_metric")==const.EVAL_METRIC_ACCURACY else "min"
        # wandb.define_metric(f'eval/{config.get("eval_metric")}', summary=summary)
        wandb.define_metric("eval/comb/acc", summary="max")
        # wandb.define_metric("eval/acc", summary="max")
        # wandb.define_metric("eval/mse", summary="min")

    model_cls = helper.string_to_object(config.get("model"))  # base model
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

    (
        trainloader,
        validloader,
        model,
        optimizer,
        class_to_idx_local,
        idx_global_to_local_mapping,
    ) = prepare_train(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        # avg_pool=config.get("avg_pool", 1),
        transform=config.get("transform"),
        augment=config.get("augment"),
        dataset=config.get("dataset"),
        data_root=config.get("root_data"),
        metadata=config.get("metadata"),
        train_valid_split_list=config.get("train_valid_split_list"),
        level=level[0],
        type_class=type_class,
        selected_classes=config.get("selected_classes"),
        validation_size=config.get("validation_size"),
        batch_size=config.get("batch_size"),
        valid_batch_size=config.get("valid_batch_size"),
        learning_rate=config.get("learning_rate"),
        random_seed=config.get("seed"),
        # is_regression=config.get("is_regression"),
        head_fine=config.get("head_fine"),  # new
        num_last_blocks=config.get("num_last_blocks"),  # new
        max_class_size=config.get("max_class_size"),
    )

    trained_model = train(
        model=model,
        model_saving_path=config.get("root_model"),
        model_saving_name=saving_name,
        trainloader=trainloader,
        validloader=validloader,
        optimizer=optimizer,
        # eval_metric=config.get("eval_metric"),
        device=device,
        epochs=config.get("epochs"),
        wandb_on=wandb_on,
        class_to_idx_local=class_to_idx_local,
        idx_global_to_local_mapping=idx_global_to_local_mapping,
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
    # avg_pool,
    transform,
    augment,
    dataset,
    data_root,
    metadata,
    train_valid_split_list,
    level,
    type_class,
    selected_classes,
    validation_size,
    batch_size,
    valid_batch_size,
    learning_rate,
    random_seed,
    # is_regression,
    head_fine,
    num_last_blocks,
    max_class_size,
):
    train_data, valid_data = preprocessing.create_train_validation_datasets(
        data_root=data_root,
        dataset=dataset,
        metadata=metadata,
        train_valid_split_list=train_valid_split_list,
        selected_classes=selected_classes,
        validation_size=validation_size,
        general_transform=transform,
        augmentation=augment,
        random_state=random_seed,
        is_regression=None,
        level=level,
        type_class=type_class,
    )

    # torch.save(valid_data, os.path.join(general_config.save_path, "valid_data.pt"))
    # print(f"classes: {train_data.class_to_idx}")

    # # load model
    # if is_regression:
    #     num_classes = 1
    # else:
    #     num_classes = len(train_data.classes)

    class_to_idx_global = train_data.class_to_idx
    class_to_idx_local = defaultdict(dict)
    idx_global_to_local_mapping = defaultdict(dict)
    num_f = []
    for cls, class_index in sorted(class_to_idx_global.items()):
        t_q_split = cls.split("__", 1)
        type_class, quality_class = t_q_split[0], t_q_split[1]
        # quality_index = const.SMOOTHNESS_INT[quality_class]
        # class_to_idx_local[type_class]["quality"][quality_class]["global_index"] = class_index
        class_to_idx_local[type_class].setdefault("quality", {})[quality_class] = {
            "global_index": class_index
        }
    for i, type_class in enumerate(class_to_idx_local.keys()):
        class_to_idx_local[type_class]["local_index"] = i
        class_to_idx_local[type_class]["global_index"] = []
        for j, quality_class in enumerate(
            class_to_idx_local[type_class]["quality"].keys()
        ):
            if head_fine == const.HEAD_CLASSIFICATION:
                class_index = j
            elif head_fine == const.HEAD_REGRESSION:
                class_index = float(const.SMOOTHNESS_INT[quality_class])
            else:
                print("Fine head not applicable!")
                raise ValueError(f"Fine head {head_fine} not applicable!")
            class_to_idx_local[type_class]["quality"][quality_class][
                "local_index"
            ] = class_index
            global_index = class_to_idx_local[type_class]["quality"][quality_class][
                "global_index"
            ]
            class_to_idx_local[type_class]["global_index"].append(global_index)
            idx_global_to_local_mapping[global_index] = {
                "coarse": i,
                "fine": class_index,
            }
        num_f.append(len(class_to_idx_local[type_class]["global_index"]))

    # instanciate model
    # model = model_cls(num_classes, avg_pool)
    model = c_cnn.C_CNN(
        base_model=model_cls,
        number_of_last_blocks=num_last_blocks,
        head_fine=head_fine,
        num_f=num_f,
    )

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

    return (
        trainloader,
        validloader,
        model,
        optimizer,
        class_to_idx_local,
        idx_global_to_local_mapping,
    )


# train the model
def train(
    model,
    model_saving_path,
    model_saving_name,
    trainloader,
    validloader,
    optimizer,
    # eval_metric,
    device,
    epochs,
    wandb_on,
    class_to_idx_local,
    idx_global_to_local_mapping,
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
        if epoch == 3:
            optimizer.add_param_group(
                {
                    "params": model.common_blocks.parameters(),
                    "lr": config.get("learning_rate") * 0.1,
                }
            )
            optimizer.add_param_group(
                {
                    "params": model.coarse_blocks.parameters(),
                    "lr": config.get("learning_rate") * 0.2,
                }
            )
            optimizer.add_param_group(
                {
                    "params": model.fine_blocks.parameters(),
                    "lr": config.get("learning_rate") * 0.2,
                }
            )

        for i, group in enumerate(optimizer.param_groups):
            num_params = sum(p.numel() for p in group["params"])
            lr = group["lr"]
            print(f"Group {i}: {num_params} Parameter, Learning Rate: {lr}")

        (
            train_loss,
            train_coarse_loss,
            train_fine_loss,
            train_comb_acc,
            train_acc_coarse,
            train_acc_fine,
        ) = train_epoch(
            model,
            trainloader,
            optimizer,
            device,
            # eval_metric,
            class_to_idx_local,
            idx_global_to_local_mapping,
        )

        (
            val_loss,
            val_coarse_loss,
            val_fine_loss,
            val_comb_acc,
            val_acc_coarse,
            val_acc_fine,
        ) = validate_epoch(
            model,
            validloader,
            device,
            # eval_metric,
            class_to_idx_local,
            idx_global_to_local_mapping,
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
                    "train/comb/acc": train_comb_acc,
                    "eval/loss": val_loss,
                    "eval/comb/acc": val_comb_acc,
                    "train/loss/coarse": train_coarse_loss,
                    "train/acc/coarse": train_acc_coarse,
                    "eval/loss/coarse": val_coarse_loss,
                    "eval/acc/coarse": val_acc_coarse,
                    "train/loss/fine": train_fine_loss,
                    "train/acc/fine": train_acc_fine,
                    "eval/loss/fine": val_fine_loss,
                    "eval/acc/fine": val_acc_fine,
                }
            )

        print(
            f"Epoch {epoch+1:>{len(str(epochs))}}/{epochs}.. \n",
            f"Train loss: {train_loss:.3f}.. \n",
            f"Test loss: {val_loss:.3f}.. \n",
            f"Train acc: {train_comb_acc:.3f}.. \n",
            f"Test acc: {val_comb_acc:.3f}.. \n",
            f"Train loss coarse: {train_coarse_loss:.3f}.. \n",
            f"Test loss coarse: {val_coarse_loss:.3f}.. \n",
            f"Train acc coarse: {train_acc_coarse:.3f}.. \n",
            f"Test acc coarse: {val_acc_coarse:.3f}.. \n",
            f"Train loss fine: {train_fine_loss:.3f}.. \n",
            f"Test loss fine: {val_fine_loss:.3f}.. \n",
            f"Train acc fine: {train_acc_fine:.3f}.. \n",
            f"Test acc fine: {val_acc_fine:.3f}.. \n",
        )

        if early_stop:
            print(f"Early stopped training at epoch {epoch}")
            break

    print("Done.")

    return model


# train a single epoch
def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    # eval_metric,
    class_to_idx_local,
    idx_global_to_local_mapping,
):
    model.train()
    coarse_criterion = model.coarse_criterion(reduction="sum")
    fine_criterion = model.fine_criterion(reduction="sum")
    running_loss = 0.0
    running_loss_coarse = 0.0
    running_loss_fine = 0.0
    eval_acc_coarse = 0
    eval_acc_fine = 0
    alpha = 0.5
    beta = 0.5

    for inputs, labels in tqdm(dataloader, desc="train batches"):
        # helper.multi_imshow(inputs, labels, "test_blur")

        inputs, labels = inputs.to(device), labels.to(device)
        # gt_coarse = idx_global_to_local_mapping.values()["coarse"][torch.searchsorted(idx_global_to_local_mapping.keys(), labels)]
        # gt_fine = idx_global_to_local_mapping.values()["fine"][torch.searchsorted(idx_global_to_local_mapping.keys(), labels)]
        keys = torch.tensor(list(idx_global_to_local_mapping.keys()))
        coarse_values = torch.tensor(
            [v["coarse"] for v in idx_global_to_local_mapping.values()]
        )
        fine_values = torch.tensor(
            [v["fine"] for v in idx_global_to_local_mapping.values()]
        )

        gt_coarse = coarse_values[torch.searchsorted(keys, labels)]
        gt_fine = fine_values[torch.searchsorted(keys, labels)]

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        outputs_coarse, outputs_fine = model.forward(inputs, gt_coarse=gt_coarse)
        predictions_coarse, predictions_fine = model.get_prediction_indices(
            outputs_coarse, outputs_fine
        )
        # if isinstance(criterion, nn.MSELoss):
        #     # outputs = outputs.flatten()
        #     labels = labels.float()
        loss_coarse = coarse_criterion(outputs_coarse, gt_coarse)
        loss_fine = fine_criterion(outputs_fine, gt_fine)
        loss = alpha * loss_coarse + beta * loss_fine
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_coarse += loss_coarse.item()
        running_loss_fine += loss_fine.item()

        # TODO: acc first only

        eval_acc_coarse += (predictions_coarse == gt_coarse).sum().item()
        eval_acc_fine += (predictions_fine == gt_fine).sum().item()

        # if eval_metric == const.EVAL_METRIC_ACCURACY:
        #     if isinstance(criterion, nn.MSELoss): # compare with is_regression for generalization?
        #         predictions = outputs.round()
        #     else:
        #         probs = model.get_class_probabilies(outputs)
        #         predictions = torch.argmax(probs, dim=1)
        #     eval_metric_value += (predictions == labels).sum().item()

        # elif eval_metric == const.EVAL_METRIC_MSE:
        #     if not isinstance(criterion, nn.MSELoss): # compare with is_regression for generalization?
        #         raise ValueError(
        #             f"Criterion must be nn.MSELoss for eval_metric {eval_metric}"
        #         )
        #     eval_metric_value = running_loss
        # else:
        #     raise ValueError(f"Unknown eval_metric: {eval_metric}")
        # break

        # epoch_losss, epoch_loss_coarse, epoch_loss_fine
        # epoch_comb_accuracy, epoch_coarse_accuracy, epoch_fine_accuracy, epoch_fine_accuracy_one_off, epoch_fine_mse,

    eval_comb_acc = alpha * eval_acc_coarse + beta * eval_acc_fine
    sampler_len = len(dataloader.sampler)
    return (
        running_loss / sampler_len,
        running_loss_coarse / sampler_len,
        running_loss_fine / sampler_len,
        eval_comb_acc / sampler_len,
        eval_acc_coarse / sampler_len,
        eval_acc_fine / sampler_len,
    )


# validate a single epoch
def validate_epoch(
    model,
    dataloader,
    device,
    # eval_metric,
    class_to_idx_local,
    idx_global_to_local_mapping,
):
    model.eval()
    coarse_criterion = model.coarse_criterion(reduction="sum")
    fine_criterion = model.fine_criterion(reduction="sum")
    running_loss = 0.0
    running_loss_coarse = 0.0
    running_loss_fine = 0.0
    eval_acc_coarse = 0
    eval_acc_fine = 0
    alpha = 0.5
    beta = 0.5

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            keys = torch.tensor(list(idx_global_to_local_mapping.keys()))
            coarse_values = torch.tensor(
                [v["coarse"] for v in idx_global_to_local_mapping.values()]
            )
            fine_values = torch.tensor(
                [v["fine"] for v in idx_global_to_local_mapping.values()]
            )

            gt_coarse = coarse_values[torch.searchsorted(keys, labels)]
            gt_fine = fine_values[torch.searchsorted(keys, labels)]

            outputs_coarse, outputs_fine = model.forward(inputs, gt_coarse=gt_coarse)
            predictions_coarse, predictions_fine = model.get_prediction_indices(
                outputs_coarse, outputs_fine
            )
            # if isinstance(criterion, nn.MSELoss):
            #     # outputs = outputs.flatten()
            #     labels = labels.float()
            loss_coarse = coarse_criterion(outputs_coarse, gt_coarse)
            loss_fine = fine_criterion(outputs_fine, gt_fine)
            loss = alpha * loss_coarse + beta * loss_fine

            running_loss += loss.item()
            running_loss_coarse += loss_coarse.item()
            running_loss_fine += loss_fine.item()

            # TODO: acc first only

            eval_acc_coarse += (predictions_coarse == gt_coarse).sum().item()
            eval_acc_fine += (predictions_fine == gt_fine).sum().item()

            # break

    eval_comb_acc = alpha * eval_acc_coarse + beta * eval_acc_fine
    sampler_len = len(dataloader.sampler)
    return (
        running_loss / sampler_len,
        running_loss_coarse / sampler_len,
        running_loss_fine / sampler_len,
        eval_comb_acc / sampler_len,
        eval_acc_coarse / sampler_len,
        eval_acc_fine / sampler_len,
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
    if level == const.FLATTEN or level == const.HIERARCHICAL:
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
    """train image classifier

    command line args:
    - config: with
        - project
        - name
        - wandb_mode
        - config
        ...
    - sweep: (Optional) False (dafault) or True
    """
    arg_parser = argparse.ArgumentParser(description="Model Training")
    arg_parser.add_argument(
        "config", type=helper.dict_type, help="Required: configuration for training"
    )
    arg_parser.add_argument(
        "--sweep",
        type=bool,
        default=False,
        help="Optinal: Running a sweep or no sweep (default=False)",
    )

    args = arg_parser.parse_args()

    run_training(args.config, args.sweep)


if __name__ == "__main__":
    main()

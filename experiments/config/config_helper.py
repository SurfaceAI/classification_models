from experiments.config import general_config
from src import constants

def sweep_config(individual_params, models):

    model = {}
    if len(models) == 1:
        model = {'model': {'value': models[0]}}
    elif len(models) > 1:
        model = {'model': {'values': models}}

    transform = {}
    for key, value in general_config.general_transform.items():
        transform[key] = {'value': value}

    augment = {}
    for key, value in general_config.augmentation.items():
        augment[key] = {'value': value}

    general_params = {
        'selected_classes': {'value': general_config.selected_classes},
        'transform': {'parameters': transform},
        'augment': {'parameters': augment},
        'seed': {'value': general_config.seed},
        'validation_size': {'value': general_config.validation_size},
        'valid_batch_size': {'value': general_config.valid_batch_size},
        'gpu_kernel': {'value': general_config.gpu_kernel},
        'checkpoint_top_n': {'value': general_config.checkpoint_top_n},
        'early_stop_thresh': {'value': general_config.early_stop_thresh},
        'save_state': {'value': general_config.save_state},
        'root_data': {'value': str(general_config.data_training_path)},
        'dataset': {'value': general_config.dataset},
        'root_model': {'value': str(general_config.trained_model_path)},
    }

    sweep_params = {
        **model,
        **general_params,
        **individual_params,
    }

    return sweep_params

def fixed_config(individual_params, model):

    general_params = {
        'selected_classes': general_config.selected_classes,
        'transform': general_config.general_transform,
        'augment': general_config.augmentation,
        'seed': general_config.seed,
        'validation_size': general_config.validation_size,
        'valid_batch_size': general_config.valid_batch_size,
        'gpu_kernel': general_config.gpu_kernel,
        'checkpoint_top_n': general_config.checkpoint_top_n,
        'early_stop_thresh': general_config.early_stop_thresh,
        'save_state': general_config.save_state,
        'root_data': str(general_config.data_training_path),
        'dataset': general_config.dataset,
        'root_model': str(general_config.trained_model_path),
    }

    fixed_config = {
        'model': model,
        **general_params,
        **individual_params,
    }

    return fixed_config

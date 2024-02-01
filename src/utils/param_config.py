from src.config import general_config

def sweep_config(individual_params, models, method, metric=None, name=None, level=None):

    model = {}
    if len(models) == 1:
        model = {'model': {'value': models[0]}}
    elif len(models) > 1:
        model = {'model': {'values': models}}

    transform = {}
    for key, value in general_config.general_transform.items():
        transform[key] = {'value': value}

    general_params = {
        'selected_classes': {'value': general_config.selected_surface_classes},
        'transform': {'parameters': transform},
        'dataset': {'value': general_config.dataset},
        'label_type': {'value': general_config.label_type},
        'seed': {'value': general_config.seed},
        'validation_size': {'value': general_config.validation_size},
    }

    sweep_params = {
        **model,
        **general_params,
        **individual_params,
        'level': {'value': level},
    }

    sweep_config = {
        'name': name,
        **method,
        **metric,
        'parameters': sweep_params,
    }

    return sweep_config

def fixed_config(individual_params, model, level=None):

    general_params = {
        'selected_classes': general_config.selected_surface_classes,
        'transform': general_config.general_transform,
        'dataset': general_config.dataset,
        'label_type': general_config.label_type,
        'seed': general_config.seed,
        'validation_size': general_config.validation_size,
    }

    fixed_config = {
        'model': model,
        **general_params,
        **individual_params,
        'level': level,
    }

    return fixed_config

from src.config import general_config
from src import constants

def sweep_config(individual_params, models, method, metric=None, name=None, level=None):

    model = {}
    if len(models) == 1:
        model = {'model': {'value': models[0]}}
    elif len(models) > 1:
        model = {'model': {'values': models}}

    level, selected_classes = level_config(level=level)

    transform = {}
    for key, value in general_config.general_transform.items():
        transform[key] = {'value': value}

    general_params = {
        'selected_classes': {'value': selected_classes},
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

    level, selected_classes = level_config(level=level)

    general_params = {
        'selected_classes': selected_classes,
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

def level_config(level=None):
    selected_classes = general_config.selected_classes
    # TODO: catch 'no selected classes' given

    # exception error for level not surface/flatten/type_class
    if level is None:
        selected_classes = list(selected_classes.keys())
        level = constants.SURFACE  
    elif level == constants.SURFACE:
        selected_classes = list(selected_classes.keys())
    elif level == constants.FLATTEN:
        pass
    else:
        selected_classes = selected_classes[level]
        level = constants.SMOOTHNESS + '/' + level

    return level, selected_classes

import sys
sys.path.append('.')

from src import constants
from src.models import training
from experiments.config import train_config
from experiments.config import config_helper

project = constants.PROJECT_SURFACE_SWEEP

name = "efficientnet"

level = constants.SURFACE

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

models = ["efficientNetV2SLogsoftmax"]

individual_params = train_config.sweep_params

config_params = config_helper.sweep_config(individual_params=individual_params, models=models)

method = train_config.sweep_method

metric = train_config.sweep_metric_acc

sweep_counts = 2

training.run_sweep_training(config_params=config_params, method=method, metric=metric, project=project, name=name, level=level, sweep_counts=sweep_counts, wandb_mode=constants.WANDB_MODE_ON)
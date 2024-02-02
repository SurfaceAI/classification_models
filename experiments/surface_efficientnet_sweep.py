import sys
sys.path.append('.')

from src import constants
from src.models import training
from src.config import train_config

project = constants.PROJECT_SURFACE_SWEEP

name = "efficientnet"

level = constants.SURFACE

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

models = ["efficientNetV2SLogsoftmax"]

individual_params = train_config.sweep_params

method = train_config.sweep_method

metric = train_config.sweep_metric_acc

sweep_counts = 2

training.run_sweep_training(individual_params=individual_params, models=models, method=method, metric=metric, project=project, name=name, level=level, sweep_counts=sweep_counts)
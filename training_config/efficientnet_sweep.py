import sys
sys.path.append('.')

from utils import constants
from utils import training
from utils import general_config
from utils import sweep
from training_config import train_config

project = constants.PROJECT_SURFACE_SWEEP

name = "efficient net"

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

models = ["efficientNetV2SLogsoftmax"]

sweep_config = sweep.sweep_setup(individual_params=train_config.sweep_params, models=models, method=train_config.sweep_method, metric=train_config.sweep_metric_acc, name=name)

sweep_counts = 2

training.run_sweep(project=project, sweep_config=sweep_config, sweep_counts=sweep_counts)
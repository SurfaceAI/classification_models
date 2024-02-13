import sys
sys.path.append('.')

from src import constants
from src.models import training
from experiments.config import train_config
from experiments.config import config_helper

project = constants.PROJECT_SURFACE_FIXED

name = "efficientnet"

level = constants.SURFACE

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

model = "efficientNetV2SLogsoftmax" # constants.efficientnet ?

individual_params = train_config.default_params

config = config_helper.fixed_config(individual_params=individual_params, model=model)

training.run_fixed_training(config=config, project=project, name=name, level=level, wandb_mode=constants.WANDB_MODE_ON)
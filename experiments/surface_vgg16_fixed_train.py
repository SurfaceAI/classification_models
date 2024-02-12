import sys
sys.path.append('.')

from src import constants
from src.models import training
from experiments.config import train_config, config_helper

project = constants.PROJECT_SURFACE_FIXED

name = "VGG16"

level = constants.SURFACE

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

model = "vgg16" # constants.VGG16 ?

individual_params = train_config.fixed_params

config = config_helper.fixed_config(individual_params=individual_params, model=model)

training.run_fixed_training(config=config, project=project, name=name, level=level, wandb_mode=constants.WANDB_MODE_ON)
import sys
sys.path.append('.')

from src import constants
from src.models import training
from experiments.config import train_config
from experiments.config import config_helper

# TODO: project name combines project name + level + fixed/sweep training function
project = constants.PROJECT_FLATTEN_FIXED

name = "RatekeCNN"

level = constants.FLATTEN

model = "rateke" # constants.RATEKE?

individual_params = train_config.fixed_params

config = config_helper.fixed_config(individual_params=individual_params, model=model)

training.run_fixed_training(config=config, project=project, name=name, level=level, wandb_mode=constants.WANDB_MODE_ON)
import sys
sys.path.append('.')

from src import constants
from src.models import training
from src.config import train_config

# TODO: project name combines project name + level + fixed/sweep training function
project = constants.PROJECT_FLATTEN_FIXED

name = "RatekeCNN"

level = constants.FLATTEN

model = "rateke" # constants.RATEKE?

individual_params = train_config.fixed_params

training.run_fixed_training(individual_params=individual_params, model=model, project=project, name=name, level=level)
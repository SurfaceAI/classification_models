import sys
sys.path.append('.')

from src import constants
from src.models import training
from src.config import train_config

project = constants.PROJECT_SURFACE_FIXED

name = "RatekeCNN"

level = constants.SURFACE

model = "rateke" # constants.RATEKE?

individual_params = train_config.fixed_params

training.run_fixed_training(individual_params=individual_params, model=model, project=project, name=name, level=level)
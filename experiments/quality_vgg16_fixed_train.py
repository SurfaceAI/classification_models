import sys
sys.path.append('.')

from src import constants
from src.models import training
from src.config import train_config, general_config

project = constants.PROJECT_SURFACE_FIXED

name = "VGG16"

model = "vgg16" # TODO: constants.VGG16 ?

individual_params = train_config.fixed_params

for type_class in general_config.selected_classes.keys():
    training.run_fixed_training(individual_params=individual_params, model=model, project=project, name=name, level=type_class)
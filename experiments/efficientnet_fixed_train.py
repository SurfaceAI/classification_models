import sys
sys.path.append('.')

from src import constants
from src.models import training
from src.config import train_config

project = constants.PROJECT_SURFACE_FIXED

name = "efficient net"

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

model = "efficientNetV2SLogsoftmax"

individual_params = train_config.fixed_params

training.run_fixed_training(individual_params=individual_params, model=model, project=project, name=name)
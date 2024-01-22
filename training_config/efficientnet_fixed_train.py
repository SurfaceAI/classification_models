import sys
sys.path.append('.')

from utils import constants
from utils import training
from utils import general_config
from utils import sweep
from training_config import train_config

project = constants.PROJECT_SURFACE_FIXED

name = "efficient net"

# save_name = 'efficientnet_v2_s__nllloss.pt'   # which model will be saved in sweep?

model = "efficientNetV2SLogsoftmax"

config = sweep.fixed_setup(individual_params=train_config.fixed_params, model=model)

training.run_fixed(project=project, fixed_config=config, name=name)
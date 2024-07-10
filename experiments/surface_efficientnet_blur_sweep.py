import sys
sys.path.append('.')

import copy

from src.models import training
from experiments.config import train_config

# # run without blur
# config = copy.deepcopy(train_config.effnet_surface_blur_sweep_params)
# config["search_params"].pop("augment", None)
# training.run_training(config=config, is_sweep=True)

# run with blur
training.run_training(config=train_config.effnet_surface_high_blur_sweep_params, is_sweep=True)

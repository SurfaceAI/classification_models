import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

training.run_training(config=train_config.efficientnet_surface_params_rtk)

training.run_training(config=train_config.efficientnet_quality_params_rtk)

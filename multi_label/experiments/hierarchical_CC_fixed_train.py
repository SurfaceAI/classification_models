import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

training.run_training(config=train_config.vgg16_surface_params)

training.run_training(config=train_config.vgg16_quality_params)
import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

training.run_training(config=train_config.ordinal_methods_sweep_params, is_sweep=True)
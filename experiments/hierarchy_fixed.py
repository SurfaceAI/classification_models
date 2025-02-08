import sys
sys.path.append('.')

from src.models import training_hierarchical
from experiments.config import train_config

training_hierarchical.run_training(config=train_config.C_CNN_fixed_params)

import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

#training.run_training(config=train_config.GH_CNN)

seeds=[42, 43, 44]
for seed in seeds:
    training.run_training(config=train_config.GH_CNN, seed=seed)
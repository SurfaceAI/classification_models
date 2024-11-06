import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

seeds=[42, 43, 44]
for seed in seeds:
    training.run_training(config=train_config.H_NET, seed=seed)

#training.run_training(config=train_config.H_NET)
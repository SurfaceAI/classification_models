import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config


#multiple_seeds
seeds=[42, 43, 44]
for seed in seeds:
    training.run_training(config=train_config.B_CNN, seed=seed)
    
    
#single seed
#training.run_training(config=train_config.B_CNN)
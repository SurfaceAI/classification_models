import sys
sys.path.append('.')

from src.models import training
from src.utils import create_valid_dataset
from experiments.config import train_config

create_valid_dataset.save_train_valid_split(config=train_config.train_valid_split_params)

training.run_training(config=train_config.type_params)

training.run_training(config=train_config.quality_params)

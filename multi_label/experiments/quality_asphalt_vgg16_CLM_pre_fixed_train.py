import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

config = train_config.vgg16_asphalt_clm_params

training.run_training(config=config)

# for max_class_size in [50, 100, 200, 400, 600, None]:
#     config["max_class_size"] = max_class_size
#     training.run_training(config=config)

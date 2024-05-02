import sys
sys.path.append('.')

from src import constants as const
from src.models import training
from experiments.config import train_config, global_config

surfaces = [
        const.ASPHALT,
        const.CONCRETE,
        const.PAVING_STONES,
        const.SETT,
        const.UNPAVED,
        ]

for surface in surfaces:
    config = {
            **train_config.effnet_quality_regression_params,
            'epochs': 7,
            'learning_rate': 0.0005,
            'level': f'{surface}',
            'dataset': f'V11/annotated/{surface}',
            'selected_classes': global_config.global_config.get("selected_classes")[f'{surface}']
            }
    print(config)
    training.run_training(config=config)

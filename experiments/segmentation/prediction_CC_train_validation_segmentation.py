import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

config = predict_config.train_validation_segmentation_CC_v2
config['name'] = config.get('name') + '__mask_' + config.get('seg_mask_style') + '__crop_' + config.get('seg_crop_style')

prediction.run_dataset_predict_segmentation(config=config)

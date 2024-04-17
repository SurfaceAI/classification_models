import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

config = predict_config.train_validation_segmentation_CC_v2
config['name'] = config.get('name') + '__mask_' + config.get('seg_mask_style') + '__crop_' + config.get('seg_crop_style')
config['segmentation_selection_func'] = 'seg_sel_func_max_area_in_lower_half_crop'

prediction.run_dataset_predict_segmentation_train_validation(config=config)

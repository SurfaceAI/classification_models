import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

config = predict_config.train_validation_segmentation_CC
config['name'] = config.get('name') + '_' + config.get('segmentation')

prediction.run_image_per_image_predict_segmentation_train_validation(config=config)

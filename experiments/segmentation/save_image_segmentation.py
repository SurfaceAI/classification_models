import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

# prediction.run_image_segmentation_per_value(predict_config.train_validation_segmentation_CC_v2)
prediction.run_image_segmentation_per_image(predict_config.seg_analysis)

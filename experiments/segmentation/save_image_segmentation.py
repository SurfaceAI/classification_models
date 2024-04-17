import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.run_image_segmentation(predict_config.train_validation_segmentation_CC_v2)

import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.run_image_per_image_predict_segmentation_train_validation(predict_config.train_validation_segmentation_CC)

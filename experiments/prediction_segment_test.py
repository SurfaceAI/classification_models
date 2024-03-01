import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.run_image_per_image_predict_segmentation(predict_config.segmentation_CC_test)

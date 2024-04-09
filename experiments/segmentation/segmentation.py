import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.run_segmentation(predict_config.segmentation)

import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.cam_prediction(predict_config.cam_surface)
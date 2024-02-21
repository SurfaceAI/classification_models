import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.run_dataset_predict_csv(predict_config.vgg16_surface)
import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config

prediction.run_dataset_predict_csv(predict_config.B_CNN_CORN_GT)

#prediction.cam_prediction(predict_config.B_CNN)
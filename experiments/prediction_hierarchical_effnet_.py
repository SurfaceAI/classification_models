import sys
sys.path.append('.')

from src.models import prediction_hierarchical
from experiments.config import predict_config

prediction_hierarchical.run_dataset_predict_csv(predict_config.C_CNN_V1_0)

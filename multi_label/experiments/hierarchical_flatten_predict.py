import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config
from src.utils import helper

# model_names = [
#     "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed42.pt",
#     # "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed43.pt",
#     # "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed44.pt",
#     # "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed45.pt",
#     # "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed46.pt"
# ]

# # Iterate over each seed and run the prediction
# for model_name in model_names:
    
#     # Optionally, update the config to include the seed info
#     config = {
#         **predict_config.FLATTEN,
#         "model_dict": {"trained_model": model_name},
#         "seed": int(model_name.split("seed")[-1].split('.')[0])  # Extract seed from model name
#     }
    
#     # Run the prediction
#     prediction.run_dataset_predict_csv(config)

#prediction.run_dataset_predict_csv(predict_config.FLATTEN)

prediction.cam_prediction(predict_config.FLATTEN)
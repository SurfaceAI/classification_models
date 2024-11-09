import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config
from src.utils import helper


# Iterate over each seed and run the prediction
# seeds = [42]
# #

# # Iterate over each seed and run the prediction
# for seed in seeds:
#     helper.set_seed(seed)
#     print(f"Running prediction with seed {seed}")
    
#     # Optionally, update the config to include the seed info
#     config = {
#         **predict_config.H_NET,
#         "seed": seed  # Add the seed to config if you want to keep track of it
#     }
    
#     # Run the prediction
#     prediction.run_dataset_predict_csv(config)
    
prediction.cam_prediction(predict_config.H_NET)

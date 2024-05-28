import sys
sys.path.append('.')
sys.path.append('..')

from experiments.config import train_config
from src.utils import preprocessing
from src import constants



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import torchvision.utils as vutils
#from torchtnt.framework.callback import Callback

import wandb
import numpy as np
import os
import pandas as pd

from datetime import datetime
import time
import pickle

from src.models import prediction
from src.architecture.vgg16 import CustomVGG16
from experiments.config import predict_config
import torchvision.models as models 

config = predict_config.B_CNN

save_features = True


#random.seed(config.get('seed'))
torch.manual_seed(config.get('seed'))
np.random.seed(config.get('seed'))

device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

model_path = os.path.join(config.get('root_model'), config.get('model_dict')['trained_model'])


# prepare data
predict_data = prediction.prepare_data(config.get("root_data"), config.get("dataset"), config.get("transform"))

# df = pd.DataFrame()
# feature_dict = {}

df, pred_outputs, image_ids, features = prediction.recursive_predict_csv(model_dict=config.get("model_dict"), 
                                                                        model_root=config.get("root_model"), 
                                                                        data=predict_data, 
                                                                        batch_size=config.get("batch_size"), 
                                                                        device=device, 
                                                                        save_features=config.get("save_features"))

# save features
features_save_name = config.get('model_dict')['trained_model'][:-3] + '-' + config.get("dataset").replace('/', '_') + '-features'
with open(os.path.join(config.get('evaluation_path'), features_save_name), 'wb') as f_out:
    pickle.dump({'image_ids': image_ids, 'prediction': pred_outputs, 'coarse_features': features[0], 'fine_features': features[1]}, f_out, protocol=pickle.HIGHEST_PROTOCOL)
    #prediction.save_features(feature_dict, os.path.join(config.get("evaluation_path"), 'feature_maps'), features_save_name)
    

# save predictions
start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
saving_name = config.get('model_dict')['trained_model'][:-3] + '-' + config.get("dataset").replace('/', '_') + '-' + start_time + '.csv'

saving_path = prediction.save_predictions_csv(df=df, saving_dir=config.get("root_predict"), saving_name=saving_name)

print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')
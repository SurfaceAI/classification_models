
import sys
sys.path.append('.')

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

import sys
sys.path.append('.')

from src.models import prediction
from src.architecture.vgg16 import CustomVGG16
from experiments.config import predict_config
import torchvision.models as models 

config = predict_config.B_CNN_PRE



#random.seed(config.get('seed'))
torch.manual_seed(config.get('seed'))
np.random.seed(config.get('seed'))

device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

model_path = os.path.join(config.get('root_model'), config.get('model_dict')['trained_model'])



# prepare data
predict_data = prediction.prepare_data(config.get("root_data"), config.get("dataset"), config.get("transform"))

df = pd.DataFrame()
feature_dict = {}

pred_outputs, image_ids, features = prediction.recursive_predict_csv(model_dict=config.get("model_dict"), model_root=config.get("root_model"), data=predict_data, batch_size=config.get("batch_size"), device=device, df=df, feature_dict=feature_dict)

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








# def to_one_hot_tensor(y, num_classes):
#     y = torch.tensor(y)
#     return F.one_hot(y, num_classes)

# #--- coarse classes ---
# num_c = 5

# #--- fine classes ---
# num_classes  = 18


# # other parameters

# #--- file paths ---

# #weights_store_filepath = './B_CNN_weights/'
# train_id = '1'
# #model_name = 'weights_B_CNN_surfaceai'+train_id+'.h5'
# #model_path = os.path.join(weights_store_filepath, model_name)


# #functions
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # Define the neural network model
# class B_CNN_t_SNE(nn.B_CNN):
#     def __init__(self, num_c, num_classes):
#         # super(B_CNN_t_SNE, self).__init__(
            
#         #     state_dict = #load our weights from training here
#         # )
    
    
#     # def forward_reimpl(self, x):
#     #     x = self.block1_layer1(x) #[batch_size, 64, 256, 256]
#     #     x = self.block1_layer2(x) #[batch_size, 64, 128, 128]
        
#     #     x = self.block2_layer1(x)#[batch_size, 64, 128, 128] 
#     #     x = self.block2_layer2(x) #(batch_size, 128, 64, 64)
        
#     #     x = self.block3_layer1(x)
#     #     x = self.block3_layer2(x)
        
#     #     flat_fine = x.reshape(x.size(0), -1) 
        
#     #     x = self.block4_layer1(x)
#     #     x = self.block4_layer2(x) # output: [batch_size, 512 #channels, 16, 16 #height&width]
        
#     #     flat_coarse = x.reshape(x.size(0), -1)
       
#     #     return flat_coarse, flat_fine






# #here we define the label tree, left is the fine class (e.g. asphalt-excellent) and right the coarse (e.g.asphalt)
# parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])




# # Transform labels for coarse level
# for i in range(y_train.shape[0]):
#     y_c_train[i][parent[torch.argmax(y_train[i])]] = 1.0

# for j in range(y_valid.shape[0]):
#     y_c_valid[j][parent[torch.argmax(y_valid[j])]] = 1.0



# # Initialize the loss weights


# # Initialize the model, loss function, and optimizer
# model = B_CNN(num_c=5, num_classes=18)
# #model = VGG16_B_CNN(num_c=5, num_classes=18)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=config.get('learning_rate'), momentum=0.9)

# def prepare_data(data_root, dataset, transform):

#     data_path = os.path.join(data_root, dataset)
#     transform = preprocessing.transform(**transform)
#     predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

#     return predict_data



# def load_model(model_path, device):
#     model_state = torch.load(model_path, map_location=device)
#     model_cls = helper.string_to_object(model_state['config']['model'])
#     is_regression = model_state['config']["is_regression"]
#     valid_dataset = model_state['dataset']

#     if is_regression:
#         class_to_idx = valid_dataset.class_to_idx
#         classes = {str(i): cls for cls, i in class_to_idx.items()}
#         num_classes = 1
#     else:
#         classes = valid_dataset.classes
#         num_classes = len(classes)
#     model = model_cls(num_classes)
#     model.load_state_dict(model_state['model_state_dict'])

#     return model, classes, is_regression, valid_dataset

# def recursive_predict_csv(model_dict, model_root, data, batch_size, device, df, level, pre_cls=None):

#     # base:
#     if model_dict is None:
#         # predictions = None
#         pass
#     else:
#         model_path = os.path.join(model_root, model_dict['trained_model'])
#         model, classes, is_regression, valid_dataset = load_model(model_path=model_path, device=device)
        
#         pred_outputs, image_ids = predict(model, data, batch_size, is_regression, device)

#         # compare valid dataset 
#         # [image_id in valid_dataset ]
#         valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
#         is_valid_data = [1 if image_id in valid_dataset_ids else 0 for image_id in image_ids]
        
#         columns = ['Image', 'Prediction', 'is_in_validation', f'Level_{level}'] # is_in_valid_dataset / join
#         pre_cls_entry = []
#         if pre_cls is not None:
#             columns = columns + [f'Level_{level-1}']
#             pre_cls_entry = [pre_cls]
#         if is_regression:
#             pred_classes = ["outside" if str(pred.item()) not in classes.keys() else classes[str(pred.item())] for pred in pred_outputs.round().int()]
#             for image_id, pred, is_vd, cls in zip(image_ids, pred_outputs, is_valid_data, pred_classes):
#                 i = df.shape[0]
#                 df.loc[i, columns] = [image_id, pred.item(), is_vd, cls] + pre_cls_entry
#         else:
#             pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
#             for image_id, pred, is_vd in zip(image_ids, pred_outputs, is_valid_data):
#                 for cls, prob in zip(classes, pred.tolist()):
#                     i = df.shape[0]
#                     df.loc[i, columns] = [image_id, prob, is_vd, cls] + pre_cls_entry
#             # subclasses not for regression implemented
#             for cls in classes:
#                 sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
#                 sub_model_dict = model_dict.get('submodels', {}).get(cls)
#                 if not sub_indices or sub_model_dict is None:
#                     continue
#                 sub_data = Subset(data, sub_indices)
#                 recursive_predict_csv(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device, df=df, level=level+1, pre_cls=cls)


# def save_predictions_json(predictions, saving_dir, saving_name):
    
#     if not os.path.exists(saving_dir):
#         os.makedirs(saving_dir)

#     saving_path = os.path.join(saving_dir, saving_name)
#     with open(saving_path, "w") as f:
#         json.dump(predictions, f)

#     return saving_path

# def save_predictions_csv(df, saving_dir, saving_name):
    
#     if not os.path.exists(saving_dir):
#         os.makedirs(saving_dir)

#     saving_path = os.path.join(saving_dir, saving_name)
#     df.to_csv(saving_path, index=False)

#     return saving_path




# def predict(model, data, batch_size, is_regression, device):
#     model = B_CNN_t_SNE
#     model.to(device)
#     model.eval()

#     loader = DataLoader(
#         data, batch_size=batch_size
#     )
    
#     all_fine_labels = []
#     all_coarse_labels = []
#     all_images = []
    
#     with torch.no_grad():
        
#        for batch_index, (inputs, fine_labels) in enumerate(data_loader):
            
#             inputs, fine_labels = inputs.to(device), fine_labels.to(device)
#             coarse_labels = parent[fine_labels]
            
#             all_fine_labels += fine_labels
#             all_coarse_label += coarse_labels
#             all_images += #todo can I get ID somehow?
                        
#             coarse_output, fine_output = model.forward(inputs)

#             current_coarse_output = coarse_output.cpu().numpy()
#             current_fine_output = fine_output.cpu().numpy()
            
#             coarse_features = np.concatenate((coarse_output, current_coarse_output))
#             fine_features = np.concatenate((fine_output, current_fine_output))


#     return coarse_features, fine_features

            
# tsne_coarse = TSNE(n=2).fit_transform(coarse_features)

# tsne_fine = TSNE(n=2).fit_transform(fine_features)

            
   
    
   
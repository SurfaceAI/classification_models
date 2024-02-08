import sys
sys.path.append('.')
sys.path.append('..')
import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from src.utils import preprocessing
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import time
from src.utils import parser
from src import constants
from experiments.config import general_config
from src.architecture import Rateke_CNN
from PIL import Image


def run_dataset_prediction(name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size):
    # TODO: config instead of data_root etc.?

    # decide flatten or surface or CC based on model_dict input!

    # load device
    device = torch.device(
        f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    data_path = os.path.join(data_root, dataset)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    predictions = recursive_predict(model_dict=model_dict, model_root=model_root, data=predict_data, batch_size=batch_size, device=device)

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = name + '-' + dataset.replace('/', '_') + '-' + start_time + '.json'

    saving_path = save_predictions(predictions=predictions, saving_dir=predict_dir, saving_name=saving_name)

    print(f'Images {dataset} predicted and saved: {saving_path}')


def recursive_predict(model_dict, model_root, data, batch_size, device):

    # base:
    if model_dict is None:
        predictions = None
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        model, classes = load_model(model_path=model_path)
        
        prediction_outputs, image_ids = predict(model, data, batch_size, device)
        # TODO: output/logits to prob function based on model last layer/parser?
        # prediction_props = 
        pred_classes = [classes[idx.item()] for idx in torch.argmax(prediction_outputs, dim=1)]

        predictions = {}
        for image_id, pred_prob, pred_cls in zip(image_ids, prediction_outputs, pred_classes):
            predictions[image_id] = {
                'label': pred_cls,
                'classes': {
                    cls: {'prob': prob} for cls, prob in zip(classes, pred_prob.tolist())
                }
            }

        for cls in classes:
            sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
            sub_model_dict = model_dict.get('submodels', {}).get(cls)
            if not sub_indices or sub_model_dict is None:
                continue
            sub_data = Subset(data, sub_indices)
            sub_predictions = recursive_predict(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device)

            if sub_predictions is not None:
                for image_id, value in sub_predictions.items():
                    predictions[image_id]['classes'][cls]['classes'] = value['classes']
                    predictions[image_id]['label'] = predictions[image_id]['label'] + '_' + value['label']
    
    return predictions


def predict(model, data, batch_size, device):
    model.to(device)
    model.eval()

    loader = DataLoader(
        data, batch_size=batch_size
    )
    
    batch_predictions = []
    ids = []
    with torch.no_grad():
        
        for inputs, id_s in loader:
            inputs = inputs.to(device)
    
            outputs = model(inputs)
            batch_predictions.append(outputs)
            ids.extend(id_s)

    predictions = torch.cat(batch_predictions, dim=0)

    return predictions, ids

def load_model(model_path):
    model_state = torch.load(model_path)
    model_name = model_state['config']['model']
    classes = model_state['dataset'].classes

    model_cfg = parser.model_name_to_config(model_name)
    model_cls = model_cfg.get('model_cls')
    
    model = model_cls(len(classes))
    model.load_state_dict(model_state['model_state_dict'])

    return model, classes

def save_predictions(predictions, saving_dir, saving_name):
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    saving_path = os.path.join(saving_dir, saving_name)
    with open(saving_path, "w") as f:
        json.dump(predictions, f)

    return saving_path



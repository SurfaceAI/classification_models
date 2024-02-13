import sys
sys.path.append('.')
sys.path.append('..')

import torch
import os
import json
from src.utils import preprocessing
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import time
from src.utils import parser
import pandas as pd
import argparse


def run_dataset_prediction_json(name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size):
    # TODO: config instead of data_root etc.?

    # decide flatten or surface or CC based on model_dict input!

    # load device
    device = torch.device(
        f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    data_path = os.path.join(data_root, dataset)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    predictions = recursive_predict_json(model_dict=model_dict, model_root=model_root, data=predict_data, batch_size=batch_size, device=device)

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = name + '-' + dataset.replace('/', '_') + '-' + start_time + '.json'

    saving_path = save_predictions_json(predictions=predictions, saving_dir=predict_dir, saving_name=saving_name)

    print(f'Images {dataset} predicted and saved: {saving_path}')

def run_dataset_prediction_csv(name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size):
    # TODO: config instead of data_root etc.?

    # decide flatten or surface or CC based on model_dict input!

    # load device
    device = torch.device(
        f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    data_path = os.path.join(data_root, dataset)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    level = 0
    columns = ['Image', 'Probability', f'Level_{level}']
    df = pd.DataFrame(columns=columns)

    recursive_predict_csv(model_dict=model_dict, model_root=model_root, data=predict_data, batch_size=batch_size, device=device, df=df, level=level)

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = name + '-' + dataset.replace('/', '_') + '-' + start_time + '.csv'

    saving_path = save_predictions_csv(df=df, saving_dir=predict_dir, saving_name=saving_name)

    print(f'Images {dataset} predicted and saved: {saving_path}')


def run_labeled_dataset_prediction_csv(name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size):
    # TODO: config instead of data_root etc.?

    # decide flatten or surface or CC based on model_dict input!

    # load device
    device = torch.device(
        f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    data_path = os.path.join(data_root, dataset)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    level = 0
    columns = ['Image', 'Probability', f'Level_{level}']
    df = pd.DataFrame(columns=columns)

    recursive_predict_csv(model_dict=model_dict, model_root=model_root, data=predict_data, batch_size=batch_size, device=device, df=df, level=level)

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = name + '-' + dataset.replace('/', '_') + '-' + start_time + '.csv'

    saving_path = save_predictions_csv(df=df, saving_dir=predict_dir, saving_name=saving_name)

    print(f'Images {dataset} predicted and saved: {saving_path}')

def recursive_predict_json(model_dict, model_root, data, batch_size, device):

    # base:
    if model_dict is None:
        predictions = None
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        model, classes, logits_to_prob, is_regression = load_model(model_path=model_path)
        
        pred_probs, image_ids = predict(model, data, batch_size, logits_to_prob, device)
        # TODO: is_regression
        pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_probs, dim=1)]

        predictions = {}
        for image_id, pred_prob, pred_cls in zip(image_ids, pred_probs, pred_classes):
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
            sub_predictions = recursive_predict_json(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device)

            if sub_predictions is not None:
                for image_id, value in sub_predictions.items():
                    predictions[image_id]['classes'][cls]['classes'] = value['classes']
                    predictions[image_id]['label'] = predictions[image_id]['label'] + '__' + value['label']
    
    return predictions

def recursive_predict_csv(model_dict, model_root, data, batch_size, device, df, level, pre_cls=None):

    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        model, classes, logits_to_prob, is_regression = load_model(model_path=model_path)
        
        pred_outputs, image_ids = predict(model, data, batch_size, logits_to_prob, is_regression, device)

        # compare valid dataset 
        [image_id in valid_dataset ]
        
        columns = ['Image', 'Prediction', f'Level_{level}'] # is_in_valid_dataset / join
        pre_cls_entry = []
        if pre_cls is not None:
            columns = columns + [f'Level_{level-1}']
            pre_cls_entry = [pre_cls]
        if is_regression:
            pred_classes = ["outside" if str(pred.item()) not in classes.keyes else classes[str(pred.item())] for pred in pred_outputs.round()]
            for image_id, pred, cls in zip(image_ids, pred_outputs, pred_classes):
                i = df.shape[0]
                df.loc[i, columns] = [image_id, pred.item(), cls] + pre_cls_entry
        else:
            pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
            for image_id, pred in zip(image_ids, pred_outputs):
                for cls, prob in zip(classes, pred.tolist()):
                    i = df.shape[0]
                    df.loc[i, columns] = [image_id, prob, cls] + pre_cls_entry
            # subclasses not for regression implemented
            for cls in classes:
                sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
                sub_model_dict = model_dict.get('submodels', {}).get(cls)
                if not sub_indices or sub_model_dict is None:
                    continue
                sub_data = Subset(data, sub_indices)
                recursive_predict_csv(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device, df=df, level=level+1, pre_cls=cls)



def predict(model, data, batch_size, logits_to_prob, is_regression, device):
    model.to(device)
    model.eval()

    loader = DataLoader(
        data, batch_size=batch_size
    )
    
    outputs = []
    ids = []
    with torch.no_grad():
        
        for batch_inputs, batch_ids in loader:
            batch_inputs = batch_inputs.to(device)
    
            outputs = model(batch_inputs)
            if is_regression:
                outputs = outputs.flatten()
            else:
                outputs = logits_to_prob(outputs)

            outputs.append(outputs)
            ids.extend(batch_ids)
            break

    pred_probs = torch.cat(outputs, dim=0)

    return pred_probs, ids

def load_model(model_path):
    model_state = torch.load(model_path)
    model_name = model_state['config']['model']
    is_regression = model_state['config']["is_regression"]
    

    model_cfg = parser.model_name_to_config(model_name)
    model_cls = model_cfg.get('model_cls')
    logits_to_prob = model_cfg.get('logits_to_prob')
    
    if is_regression:
        idx_to_classes = model_state['dataset'].idx_to_classes
        classes = {str(i): cls for cls, i in idx_to_classes.items()}
        num_classes = 1
    else:
        classes = model_state['dataset'].classes
        num_classes = len(classes)
    model = model_cls(num_classes)
    model.load_state_dict(model_state['model_state_dict'])

    return model, classes, logits_to_prob, is_regression

def save_predictions_json(predictions, saving_dir, saving_name):
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    saving_path = os.path.join(saving_dir, saving_name)
    with open(saving_path, "w") as f:
        json.dump(predictions, f)

    return saving_path

def save_predictions_csv(df, saving_dir, saving_name):
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    saving_path = os.path.join(saving_dir, saving_name)
    df.to_csv(saving_path, index=False)

    return saving_path


# def main():
#     # command line args
#     # name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size
#     arg_parser = argparse.ArgumentParser(description='Model Prediction')
#     arg_parser.add_argument('saving_type', type=str, help='Required: saving type of predictions: csv or json')
#     arg_parser.add_argument('name', type=str, help='Required: name used for saving file name')
#     arg_parser.add_argument('data_root', type=str, help='Required: root to dataset folder')
#     arg_parser.add_argument('dataset', type=str, help='Required: dataset folder')
#     arg_parser.add_argument('transform', type=dict, help='Required: transformation dictionary for dataset')
#     arg_parser.add_argument('model_root', type=str, help='Required: root where to find the trained models')
#     arg_parser.add_argument('model_dict', type=dict, help='Required: dictionary defining trained models to use for prediction')
#     arg_parser.add_argument('predict_dir', type=str, help='Required: directory to save prediction')
#     arg_parser.add_argument('gpu_kernel', type=int, help='Required: gpu kernel')
#     arg_parser.add_argument('batch_size', type=int, help='Required: batch size used for prediction')
    
#     args = arg_parser.parse_args()

#     # csv or json
#     if args.saving_type == 'csv':
#         run_dataset_prediction_csv(args.name,
#                                    args.data_root,
#                                    args.dataset,
#                                    args.transform,
#                                    args.model_root,
#                                    args.model_dict,
#                                    args.predict_dir,
#                                    args.gpu_kernel,
#                                    args.batch_size)
#     elif args.saving_type == 'json':
#         run_dataset_prediction_json(args.name,
#                                    args.data_root,
#                                    args.dataset,
#                                    args.transform,
#                                    args.model_root,
#                                    args.model_dict,
#                                    args.predict_dir,
#                                    args.gpu_kernel,
#                                    args.batch_size)
#     else:
#         print('no valid saving format')

# if __name__ == "__main__":
#     main()
import sys
sys.path.append('.')
sys.path.append('..')

import torch
import os
import json
from src.utils import preprocessing
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime
import time
from src.utils import helper
from src import constants as const
from experiments.config import global_config
from src.architecture import Rateke_CNN
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from src.architecture import c_cnn


def run_dataset_predict_csv(config):
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    data = prepare_data(config.get("root_data"), config.get("dataset"), config.get("transform"))

    level = 0
    columns = ['Image', 'Prediction', 'Level', f'Level_{level}']
    # TODO: rewrite csv creation without pandas dataframe
    df = pd.DataFrame(columns=columns)

    model_dict=config.get("model_dict")
    model_root=config.get("root_model")
    batch_size=config.get("batch_size")

    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        # level_name = model_dict.get('level', '')
        model, class_to_idx_local, head_fine, valid_dataset = load_model(model_path=model_path, device=device)
        
        pred_outputs_coarse_val, pred_outputs_fine_val, pred_outputs_coarse_idx, pred_outputs_fine_idx, image_ids = predict(model, data, batch_size, device)

        # compare valid dataset 
        # [image_id in valid_dataset ]
        valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in helper.get_attribute(valid_dataset, "samples")]
        is_valid_data = [1 if image_id in valid_dataset_ids else 0 for image_id in image_ids]

        classes = {value["local_index"]: {"type": key} for key, value in class_to_idx_local.items()}
        for key, value in classes.items():
            classes[key]["quality"] = {v["local_index"]: k for k, v in class_to_idx_local[value["type"]]["quality"].items()}

        # Surface
        level = 0
        level_name = const.TYPE
        columns = ['Image', 'Prediction', 'Level', 'is_in_validation', f'Level_{level}']
        coarse_classes = [classes[key]["type"] for key in sorted(classes.keys())]
        pre_classes = [classes[idx.item()]["type"] for idx in pred_outputs_coarse_idx]
        df_tmp = pd.DataFrame(columns=columns, index=range(pred_outputs_coarse_val.shape[0] * pred_outputs_coarse_val.shape[1]))
        i = 0
        for image_id, pred, is_vd in tqdm(zip(image_ids, pred_outputs_coarse_val, is_valid_data), desc="write df"):
            for cls, prob in zip(coarse_classes, pred.tolist()):
                df_tmp.iloc[i] = [image_id, prob, level_name, is_vd, cls]
                i += 1
        print(df_tmp.shape)
        df = pd.concat([df, df_tmp], ignore_index=True)
        print(df.shape)

        # Quality
        level = 1
        level_name = const.QUALITY
        columns = ['Image', 'Prediction', 'Level', 'is_in_validation', f'Level_{level}', f'Level_{level-1}'] # is_in_valid_dataset / join
        
        if head_fine == const.HEAD_REGRESSION:
            pred_classes = ["outside" if idx.item() not in classes[coarse_idx.item()]["quality"].keys() else classes[coarse_idx.item()]["quality"][idx.item()] for idx, coarse_idx in zip(pred_outputs_fine_idx, pred_outputs_coarse_idx)]
            df_tmp = pd.DataFrame(columns=columns, index=range(pred_outputs_fine_val.shape[0]))
            i = 0
            for image_id, pred, is_vd, cls, pre_cls in tqdm(zip(image_ids, pred_outputs_fine_val, is_valid_data, pred_classes, pre_classes), desc="write df"):
                df_tmp.iloc[i] = [image_id, pred.item(), level_name, is_vd, cls, pre_cls]
                i += 1
            print(df_tmp.shape)
            df = pd.concat([df, df_tmp], ignore_index=True)
            print(df.shape)
        elif head_fine == const.HEAD_CLASSIFICATION:
            # pred_classes = [classes[coarse_idx]["quality"][idx.item()] for idx, coarse_idx in zip(pred_outputs_fine_idx, pred_outputs_coarse_idx)]
            df_tmp = pd.DataFrame(columns=columns, index=range(pred_outputs_fine_val.shape[0] * pred_outputs_fine_val.shape[1]))
            i = 0
            for image_id, pred, is_vd, pre_cls, coarse_idx in tqdm(zip(image_ids, pred_outputs_fine_val, is_valid_data, pre_classes, pred_outputs_coarse_idx), desc="write df"):
                fine_classes = [classes[coarse_idx.item()]["quality"][key] for key in sorted(classes[coarse_idx.item()]["quality"].keys())]
                for cls, prob in zip(fine_classes, pred.tolist()):
                    df_tmp.iloc[i] = [image_id, prob, level_name, is_vd, cls, pre_cls]
                    i += 1
            print(df_tmp.shape)
            df = pd.concat([df, df_tmp], ignore_index=True)
            print(df.shape)
        else:
            raise ValueError(f"Fine head {head_fine} not applicable!")

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = config.get("name") + '-' + config.get("dataset").replace('/', '_') + '-' + start_time + '.csv'

    saving_path = save_predictions_csv(df=df, saving_dir=config.get("root_predict"), saving_name=saving_name)

    print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')

# prediction without gt
def predict(model, data, batch_size, device):
    model.to(device)
    model.eval()

    loader = DataLoader(
        data, batch_size=batch_size
    )
    
    outputs_coarse_val = []
    outputs_fine_val = []
    outputs_coarse_idx = []
    outputs_fine_idx = []
    ids = []
    with torch.no_grad():
        
        for batch_inputs, batch_ids in tqdm(loader, desc="predict batches"):
            batch_inputs = batch_inputs.to(device)
    
            batch_outputs_coarse, batch_outputs_fine = model.forward(batch_inputs)
            batch_outputs_coarse_val, batch_outputs_fine_val = model.get_prediction_values(batch_outputs_coarse, batch_outputs_fine)
            batch_outputs_coarse_idx, batch_outputs_fine_idx = model.get_prediction_indices(batch_outputs_coarse, batch_outputs_fine)

            outputs_coarse_val.append(batch_outputs_coarse_val)
            outputs_fine_val.append(batch_outputs_fine_val)
            outputs_coarse_idx.append(batch_outputs_coarse_idx)
            outputs_fine_idx.append(batch_outputs_fine_idx)
            ids.extend(batch_ids)

            break # TODO: debug only

    pred_outputs_coarse_val = torch.cat(outputs_coarse_val, dim=0)
    pred_outputs_fine_val = torch.cat(outputs_fine_val, dim=0)
    pred_outputs_coarse_idx = torch.cat(outputs_coarse_idx, dim=0)
    pred_outputs_fine_idx = torch.cat(outputs_fine_idx, dim=0)

    return pred_outputs_coarse_val, pred_outputs_fine_val, pred_outputs_coarse_idx, pred_outputs_fine_idx, ids


def prepare_data(data_root, dataset, transform): # TODO differentiate gt

    data_path = os.path.join(data_root, dataset)
    transform = preprocessing.transform(**transform)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    return predict_data

def load_model(model_path, device): # TODO adapt
    model_state = torch.load(model_path, map_location=device)
    model_cls = helper.string_to_object(model_state['config']['model'])
    # is_regression = model_state['config']["is_regression"]
    # avg_pool = model_state['config'].get("avg_pool", 1)
    num_last_blocks = model_state['config']["num_last_blocks"]
    head_fine = model_state['config']["head_fine"]
    # is_regression = False
    valid_dataset = model_state['dataset']

    class_to_idx_global = helper.get_attribute(valid_dataset, "class_to_idx")
    class_to_idx_local = defaultdict(dict)
    idx_global_to_local_mapping = defaultdict(dict)
    num_f = []
    for cls, class_index in sorted(class_to_idx_global.items()):
        t_q_split = cls.split("__", 1)
        type_class, quality_class = t_q_split[0], t_q_split[1]
        # quality_index = const.SMOOTHNESS_INT[quality_class]
        # class_to_idx_local[type_class]["quality"][quality_class]["global_index"] = class_index
        class_to_idx_local[type_class].setdefault("quality", {})[quality_class] = {
            "global_index": class_index
        }
    for i, type_class in enumerate(class_to_idx_local.keys()):
        class_to_idx_local[type_class]["local_index"] = i
        class_to_idx_local[type_class]["global_index"] = []
        for j, quality_class in enumerate(
            class_to_idx_local[type_class]["quality"].keys()
        ):
            if head_fine == const.HEAD_CLASSIFICATION:
                class_index = j
            elif head_fine == const.HEAD_REGRESSION:
                class_index = float(const.SMOOTHNESS_INT[quality_class])
            else:
                print("Fine head not applicable!")
                raise ValueError(f"Fine head {head_fine} not applicable!")
            class_to_idx_local[type_class]["quality"][quality_class][
                "local_index"
            ] = class_index
            global_index = class_to_idx_local[type_class]["quality"][quality_class][
                "global_index"
            ]
            class_to_idx_local[type_class]["global_index"].append(global_index)
            idx_global_to_local_mapping[global_index] = {
                "coarse": i,
                "fine": class_index,
            }
        num_f.append(len(class_to_idx_local[type_class]["global_index"]))

    # instanciate model
    # model = model_cls(num_classes, avg_pool)
    model = c_cnn.C_CNN(
        base_model=model_cls,
        number_of_last_blocks=num_last_blocks,
        head_fine=head_fine,
        num_f=num_f,
    )
    model.load_state_dict(model_state['model_state_dict'])

    return model, class_to_idx_local, head_fine, valid_dataset

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


# def run_dataset_prediction_json(name, data_root, dataset, transform, model_root, model_dict, predict_dir, gpu_kernel, batch_size):
#     # TODO: config instead of data_root etc.?

#     # decide flatten or surface or CC based on model_dict input!

#     # load device
#     device = torch.device(
#         f"cuda:{gpu_kernel}" if torch.cuda.is_available() else "cpu"
#     )

#     # prepare data
#     data_path = os.path.join(data_root, dataset)
#     predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

#     predictions = recursive_predict_json(model_dict=model_dict, model_root=model_root, data=predict_data, batch_size=batch_size, device=device)

#     # save predictions
#     start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
#     saving_name = name + '-' + dataset.replace('/', '_') + '-' + start_time + '.json'

#     saving_path = save_predictions_json(predictions=predictions, saving_dir=predict_dir, saving_name=saving_name)

#     print(f'Images {dataset} predicted and saved: {saving_path}')

# def recursive_predict_json(model_dict, model_root, data, batch_size, device):

#     # base:
#     if model_dict is None:
#         predictions = None
#     else:
#         model_path = os.path.join(model_root, model_dict['trained_model'])
#         model, classes, logits_to_prob, is_regression = load_model(model_path=model_path)
        
#         pred_probs, image_ids = predict(model, data, batch_size, logits_to_prob, device)
#         # TODO: is_regression
#         pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_probs, dim=1)]

#         predictions = {}
#         for image_id, pred_prob, pred_cls in zip(image_ids, pred_probs, pred_classes):
#             predictions[image_id] = {
#                 'label': pred_cls,
#                 'classes': {
#                     cls: {'prob': prob} for cls, prob in zip(classes, pred_prob.tolist())
#                 }
#             }

#         for cls in classes:
#             sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
#             sub_model_dict = model_dict.get('submodels', {}).get(cls)
#             if not sub_indices or sub_model_dict is None:
#                 continue
#             sub_data = Subset(data, sub_indices)
#             sub_predictions = recursive_predict_json(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device)

#             if sub_predictions is not None:
#                 for image_id, value in sub_predictions.items():
#                     predictions[image_id]['classes'][cls]['classes'] = value['classes']
#                     predictions[image_id]['label'] = predictions[image_id]['label'] + '__' + value['label']
    
#     return predictions

def main():
    '''predict images in folder
    
    command line args:
    - config: with
        - name
        - data_root
        - dataset
        - transform
        - model_root
        - model_dict
        - predict_dir
        - gpu_kernel
        - batch_size
    - saving_type: (Optional) csv (dafault) or json
    '''
    arg_parser = argparse.ArgumentParser(description='Model Prediction')
    arg_parser.add_argument('config', type=helper.dict_type, help='Required: configuration for prediction')
    arg_parser.add_argument('--type', type=str, default='csv', help='Optinal: saving type of predictions: csv (default) or json')
    
    args = arg_parser.parse_args()

    # csv or json
    if args.type == 'csv':
        run_dataset_predict_csv(args.config)
    # elif args.saving_type == 'json':
    #     run_dataset_predict_json(args.config)
    else:
        print('no valid saving format')

if __name__ == "__main__":
    main()
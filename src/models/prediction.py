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
from src.utils import helper
from src import constants
from experiments.config import global_config
from Archive.architectures import Rateke_CNN
from PIL import Image
import pandas as pd
import argparse
import pickle 
from collections import OrderedDict
from src import constants as const
from coral_pytorch.dataset import corn_label_from_logits



def run_dataset_predict_csv(config):
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    predict_data = prepare_data(config.get("root_data"), config.get("dataset"), config.get("transform"))

    # if config.get('level') is not "hierarchical":
    #     level_no = 0
    #     columns = ['Image', 'Prediction', f'Level_{level_no}']
        #df = pd.DataFrame(columns=columns)

    df, pred_outputs, image_ids, features = recursive_predict_csv(model_dict=config.get("model_dict"), 
                          model_root=config.get("root_model"), 
                          data=predict_data, 
                          batch_size=config.get("batch_size"), 
                          device=device, 
                          level=config.get('level'), 
                          save_features=config.get('save_features'))

    # save features
    features_save_name = config.get("name") + '-' + config.get("dataset").replace('/', '_')
    save_features(features, os.path.join(config.get("evaluation_path"), 'feature_maps'), features_save_name)

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = config.get("name") + '-' + config.get("dataset").replace('/', '_') + '-' + start_time + '.csv'

    saving_path = save_predictions_csv(df=df, saving_dir=config.get("root_predict"), saving_name=saving_name)

    print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')

def recursive_predict_csv(model_dict, model_root, data, batch_size, device, level, save_features, level_no=None, pre_cls=None):

    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        model, classes, head, level_no, valid_dataset = load_model(model_path=model_path, device=device)
        
    
        pred_outputs, image_ids, features = predict(model, data, batch_size, head, level, device, save_features) 
        
        # compare valid dataset 
        # [image_id in valid_dataset ]
        valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
        is_valid_data = [1 if image_id in valid_dataset_ids else 0 for image_id in image_ids]
        
        df = pd.DataFrame()        
        if level == const.HIERARCHICAL: 
            columns =  ['Image', 'Coarse_Prediction', 'Coarse_Probability', 'Fine_Prediction', 'Fine_Probability', 'is_in_validation']
            #todo: add regression
            pred_coarse_outputs = pred_outputs[0]
            pred_fine_outputs = pred_outputs[1]
            
            coarse_classes = classes[0]
            fine_classes = classes[1]
            
            pred_coarse_classes = [coarse_classes[idx.item()] for idx in torch.argmax(pred_coarse_outputs, dim=1)]
            
            if head == const.CLASSIFICATION:
                pred_fine_classes = [fine_classes[idx.item()] for idx in torch.argmax(pred_fine_outputs, dim=1)]
            elif head == const.REGRESSION:
                pred_fine_classes = [fine_classes[idx.item()] for idx in pred_fine_outputs.round().int()]
            elif head == const.CLM:
                pred_fine_classes = [fine_classes[idx.item()] for idx in torch.argmax(pred_fine_outputs, dim=1)]
            elif head == const.CORN:
                pred_fine_classes = [fine_classes[idx.item()] for idx in corn_label_from_logits(pred_fine_outputs)]
                
            coarse_probs, _ = torch.max(pred_coarse_outputs, dim=1)
            fine_probs, _ = torch.max(pred_fine_outputs, dim=1)

            for image_id, coarse_pred, coarse_prob, fine_pred, fine_prob, is_vd, in zip(image_ids, pred_coarse_classes, coarse_probs.tolist(), pred_fine_classes, fine_probs.tolist(), is_valid_data):
                i = df.shape[0]
                df.loc[i, columns] = [float(image_id), coarse_pred, coarse_prob, fine_pred, fine_prob, is_vd]
          
          
        #classifier chain  
        else:
            level_no = 0
            columns = ['Image', 'Prediction', 'is_in_validation', f'Level_{level_no}'] # is_in_valid_dataset / join
            pre_cls_entry = []
            if pre_cls is not None:
                columns = columns + [f'Level_{level_no-1}']
                pre_cls_entry = [pre_cls]
                
            if head == const.REGRESSION:
                pred_classes = ["outside" if str(pred.item()) not in classes.keys() else classes[str(pred.item())] for pred in pred_outputs.round().int()]
                for image_id, pred, is_vd, cls in zip(image_ids, pred_outputs, is_valid_data, pred_classes):
                    i = df.shape[0]
                    df.loc[i, columns] = [image_id, pred.item(), is_vd, cls] + pre_cls_entry
            else:
                pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
                for image_id, pred, is_vd in zip(image_ids, pred_outputs, is_valid_data):
                    for cls, prob in zip(classes, pred.tolist()):
                        i = df.shape[0]
                        df.loc[i, columns] = [image_id, prob, is_vd, cls] + pre_cls_entry
                # subclasses not for regression implemented
                for cls in classes:
                    sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
                    sub_model_dict = model_dict.get('submodels', {}).get(cls)
                    if not sub_indices or sub_model_dict is None:
                        continue
                    sub_data = Subset(data, sub_indices)
                    recursive_predict_csv(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device, df=df, level_no=level_no+1, pre_cls=cls)
                    
        return df, pred_outputs, image_ids, features


def predict(model, data, batch_size, head, level, device, save_features):
    model.to(device)
    model.eval()

    loader = DataLoader(
        data, batch_size=batch_size
    )
    
    if level ==  const.HIERARCHICAL:
        coarse_outputs = []
        fine_outputs = []
    else:
        outputs = []
        
    ids = []
    
    #where we store intermediate outputs 
    if save_features is True:
        
        feature_dict = {}

        h_1 = model.block3_layer2.register_forward_hook(helper.make_hook("h1_features", feature_dict))
        h_2 = model.block4_layer2.register_forward_hook(helper.make_hook("h2_features", feature_dict))

        all_coarse_features = []
        all_fine_features = []
    
    with torch.no_grad():
        
        for index, (batch_inputs, batch_ids) in enumerate(loader):
            batch_inputs = batch_inputs.to(device)

            if level == const.HIERARCHICAL:
                coarse_batch_outputs, fine_batch_outputs = model(batch_inputs)
                
                coarse_batch_outputs = model.get_class_probabilities(coarse_batch_outputs)
                
                if head == const.CLASSIFICATION:
                    fine_batch_outputs = model.get_class_probabilities(fine_batch_outputs)
                elif head == const.REGRESSION:
                    fine_batch_outputs = fine_batch_outputs.flatten()
                else:
                    pass # CLM outputs probs already and corn output is being transformed later
                 
                coarse_outputs.append(coarse_batch_outputs)
                fine_outputs.append(fine_batch_outputs)
             
            #Classifier Chain   
            else:
                batch_outputs = model(batch_inputs)
                
                if head == const.CLASSIFICATION:
                    batch_outputs = model.get_class_probabilities(batch_outputs) 
                elif head == const.REGRESSION:
                    batch_outputs = batch_outputs.flatten()
                else:
                    pass
                    
                outputs.append(batch_outputs)
            
            ids.extend(batch_ids)
            
            if save_features:
                #flatten to vector
                for feature in feature_dict:
                    feature_dict[feature] = feature_dict[feature].view(feature_dict[feature].size(0), -1)
                    
                all_coarse_features.append(feature_dict['h1_features'])
                all_fine_features.append(feature_dict['h2_features'])
              
            # if index == 1:
            #     break 
    # h_1.remove()
    # h_2.remove()
    
    if level == const.HIERARCHICAL:
        pred_coarse_outputs = torch.cat(coarse_outputs, dim=0)
        pred_fine_outputs = torch.cat(fine_outputs, dim=0)
        all_coarse_features = torch.cat(all_coarse_features, dim=0)
        all_fine_features = torch.cat(all_fine_features, dim=0)
        
        if save_features: 
            h_1.remove()
            h_2.remove()
            all_features = [all_coarse_features, all_fine_features]
            return (pred_coarse_outputs, pred_fine_outputs), ids, all_features

    else:
        pred_outputs = torch.cat(outputs, dim=0)
        if save_features:
            h_1.remove()
            h_2.remove()
            return pred_outputs, ids, feature_dict
        else:
            return pred_outputs, ids,

def prepare_data(data_root, dataset, transform):

    data_path = os.path.join(data_root, dataset)
    transform = preprocessing.transform(**transform)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    return predict_data

def load_model(model_path, device):
    model_state = torch.load(model_path, map_location=device)
    model_cls = helper.string_to_object(model_state['config']['model'])
    head = model_state['config']["head"]
    level = model_state['config']["level"]
    valid_dataset = model_state['dataset']
    
    #Hierarhical
    if level == const.HIERARCHICAL: 
        fine_classes = valid_dataset.classes  #TODO: Adapt this to classes with clm and corn (-> dict like in to_train_list training.py)
        coarse_classes = list(OrderedDict.fromkeys(class_name.split('__')[0] for class_name in fine_classes))
        num_c = len(coarse_classes)
        num_classes = len(fine_classes) 
        model = model_cls(num_c = num_c, num_classes=num_classes, head=head) 
        model.load_state_dict(model_state['model_state_dict'])
        
        return model, (coarse_classes, fine_classes), head, level, valid_dataset
    
    #CC    
    else:
        if head == const.REGRESSION:
            class_to_idx = valid_dataset.class_to_idx
            classes = {str(i): cls for cls, i in class_to_idx.items()}
            num_classes = 1
        #add clm and corn
        else:
            classes = valid_dataset.classes  
            num_classes = len(classes)
        model = model_cls(num_classes=num_classes) 
        model.load_state_dict(model_state['model_state_dict'])
        
        return model, classes, head, level, valid_dataset

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

def save_features(features_dict, saving_dir, saving_name):
    
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    saving_path = os.path.join(saving_dir, saving_name)
    torch.save(features_dict, saving_path)


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
#         model, classes, logits_to_prob, head = load_model(model_path=model_path)
        
#         pred_probs, image_ids = predict(model, data, batch_size, logits_to_prob, device)
#         # TODO: head
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
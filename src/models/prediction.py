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
#from Archive.architectures import Rateke_CNN
from PIL import Image
import pandas as pd
import argparse
import pickle 
from collections import OrderedDict
from src import constants as const
from coral_pytorch.dataset import corn_label_from_logits
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torch.utils.data import DataLoader


def cam_prediction(config):
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    normalize_transform = transforms.Normalize(*config.get("transform")['normalize'])
    non_normalize_transform = {
        **config.get("transform"),
        'normalize': None,
    }
    predict_data = prepare_data(config.get("root_data"), config.get("dataset"), config.get("metadata"), non_normalize_transform, ds_type=config.get("ds_type"))
    
    model_path = os.path.join(config.get("root_model"), config.get("model_dict")['trained_model'])
    model, classes, head, level, valid_dataset, hierarchy_method = load_model(model_path=model_path, device=device)
    image_folder = os.path.join(config.get("root_predict"), config.get("dataset"))
    os.makedirs(image_folder, exist_ok=True)
        
    cam_subfolder = os.path.join(image_folder, 'cam', os.path.splitext(os.path.basename(model_path))[0])
    os.makedirs(cam_subfolder, exist_ok=True)
    
    if level == const.HIERARCHICAL:
        save_cam_hierarchical(model, predict_data, normalize_transform, classes, valid_dataset, head, level, hierarchy_method, device, cam_subfolder)
    elif level == const.FLATTEN or level == const.CC:
        save_cam_flattened(model, predict_data, normalize_transform, classes, valid_dataset, head, hierarchy_method, level, device, cam_subfolder)


    print(f'Images {config.get("dataset")} predicted and saved with CAM: {image_folder}')


def run_dataset_predict_csv(config):
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )
    # prepare data
    predict_data = prepare_data(config.get("root_data"), config.get("dataset"), config.get("metadata"), config.get("transform"), ds_type=config.get("ds_type"))
    print(len(predict_data))
    #Filter data 
    # if config.get('level') == const.CC:
    #     if config.get('save_features'):
    #         df, all_features = predict_surface_and_quality(
    #             model_root = config.get("root_model"),
    #             model_dict=config["model_dict"],
    #             data=predict_data,
    #             device=device,
    #             batch_size=config.get("batch_size"),
    #             save_features=config.get('save_features'),
    #             seed=config["seed"],
    #         )
    #     else:
    #         df = predict_surface_and_quality(
    #             model_root = config.get("root_model"),
    #             model_dict=config["model_dict"],
    #             data=predict_data,
    #             device=device,
    #             batch_size=config.get("batch_size"),
    #             save_features=config.get('save_features'),
    #             seed=config["seed"],
    #         )

    # else:   
    level_no = 0
  
    if config.get('level') == const.CC:
        level_no = 0
        # columns = ['Image', 'Prediction', 'Level', f'Level_{level_no}']
        # df = pd.DataFrame(columns=columns)
    if config.get('save_features'):
        df, pred_outputs, image_ids, all_features = recursive_predict_csv(model_dict=config.get("model_dict"), 
                            model_root=config.get("root_model"), 
                            data=predict_data, 
                            batch_size=config.get("batch_size"), 
                            device=device, 
                            level=config.get('level'), 
                            hierarchy_method=config.get('hierarchy_method'),
                            save_features=config.get('save_features'),
                            level_no=level_no,
                            seed=config["seed"],)
    else: 
        df, pred_outputs, image_ids = recursive_predict_csv(model_dict=config.get("model_dict"), 
                            model_root=config.get("root_model"), 
                            data=predict_data, 
                            batch_size=config.get("batch_size"), 
                            device=device, 
                            level=config.get('level'),  
                            hierarchy_method=config.get('hierarchy_method'),
                            save_features=config.get('save_features'),
                            level_no=level_no,
                            seed=config["seed"])
    
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    # save features
    if config.get('save_features'):
        if config.get('level') == const.CC:
            for feature_key, features in all_features.items():
                features_dict = {
                'image_ids': image_ids,
                'pred_outputs': pred_outputs,
                'features': features
                }
                # Create a unique name for each feature set
                features_save_name = config.get("model_dict")['trained_model'] + '-' + config.get('level') +'-' + config.get('head') +'-'+ feature_key + '-' + config.get("dataset").replace('\\', '_') + '-' + config.get('ds_type') +  '-' + start_time
                # Save the features
                #save_features(features, os.path.join(config.get("root_predict"), 'feature_maps'), features_save_name)
                with open(os.path.join(config.get("root_predict"), 'feature_maps', features_save_name), 'wb') as f_out:
                    pickle.dump(features_dict, f_out, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'CC features saved: {feature_key}')
                
        elif config.get('level') == const.HIERARCHICAL:
            features_dict = {
                'image_ids': image_ids,
                'pred_outputs': pred_outputs,
                'coarse_features': all_features[0] if len(features) > 0 else None,
                'fine_features': all_features[1] if len(features) > 1 else None
            }
            features_save_name = config.get("model_dict")['trained_model'] + '-' + config.get('level') + '-' + config.get("dataset").replace('\\', '_') + '-' + config.get('ds_type') + '-' + start_time
            with open(os.path.join(config.get("root_predict"), 'feature_maps', features_save_name), 'wb') as f_out:
                pickle.dump(features_dict, f_out, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Hierarchical features saved')
                
                
        else:
            features_dict = {
                'image_ids': image_ids,
                'pred_outputs': pred_outputs,
                'features': all_features if len(all_features) > 0 else None,
            }
            features_save_name = config.get("model_dict")['trained_model'] + '-' + config.get('level') + '-' + config.get("dataset").replace('\\', '_') + '-' + config.get('ds_type') + '-' + start_time
            with open(os.path.join(config.get("root_predict"), 'feature_maps', features_save_name), 'wb') as f_out:
                pickle.dump(features_dict, f_out, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'flattened features saved')
            
            #features_save_name = config.get("model_dict")['trained_model'] + '-' + config.get("dataset").replace('\\', '_')
            #save_features(features, os.path.join(config.get("root_predict"), 'feature_maps'), features_save_name)

    df['Seed'] = config.get('seed')
    # save predictions
    saving_name = config.get("model_dict")['trained_model'] + '-' + config.get('head') +'-' + config.get("dataset").replace('\\', '_') + '-' + config.get('ds_type') + '-' + start_time + '-'+'.csv'

    saving_path = save_predictions_csv(df=df, saving_dir=os.path.join(config.get("root_predict")), saving_name=saving_name)
    #print(df)

    print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')

def recursive_predict_csv(model_dict, model_root, data, batch_size, device, level, hierarchy_method, save_features, df=None, level_no=None, pre_cls=None, seed=None):
    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        model, classes, head, level, valid_dataset, hierarchy_method = load_model(model_path=model_path, device=device)
        print(classes)
        #level = const.FLATTEN
        print(level)
        print(head)
        #classes = ["excellent", "good", "intermediate", "bad"]

        if save_features:
            pred_outputs, image_ids, features = predict(model, data, batch_size, head, level, device, save_features, seed) 
        else:
            pred_outputs, image_ids = predict(model, data, batch_size, head, level, device, save_features, seed) 
        
        # compare valid dataset 
        # [image_id in valid_dataset ]
        valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
        is_valid_data = [1 if image_id in valid_dataset_ids else 0 for image_id in image_ids]
        
        if level == const.HIERARCHICAL:   
            df = pd.DataFrame() #todo: add regression
            pred_coarse_outputs = pred_outputs[0]
            pred_fine_outputs = pred_outputs[1]
            
            coarse_classes = classes[0]
            fine_classes = classes[1]
            #fine_classes = sorted(classes[1], key=lambda x: const.FLATTENED_INT[x]) #ordered according to integer values
            
            pred_coarse_classes = [coarse_classes[idx.item()] for idx in torch.argmax(pred_coarse_outputs, dim=1)]
            
            if hierarchy_method == const.MODELSTRUCTURE:
                columns =  ['Image', 'Coarse_Prediction', 'Coarse_Probability', 'Fine_Prediction', 'Fine_Probability', 'is_in_validation']            
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
                   
            elif hierarchy_method == const.GROUNDTRUTH:
                columns =  ['Image', 'Coarse_Prediction', 'Fine_Prediction', 'is_in_validation']

                pred_fine_classes = helper.compute_fine_prediction_hierarchical_GT(pred_fine_outputs, pred_coarse_classes, hierarchy_method, head, fine_classes)
                
                for image_id, coarse_pred, fine_pred, is_vd, in zip(image_ids, pred_coarse_classes, pred_fine_classes, is_valid_data):
                    i = df.shape[0]
                    df.loc[i, columns] = [float(image_id), coarse_pred, fine_pred, is_vd]
                    
            if save_features:          
                return df, pred_outputs, image_ids, features
            else:
                return df, pred_outputs, image_ids
            
        #classifier chain  
        elif level == const.FLATTEN or level ==  const.ASPHALT:
            df = pd.DataFrame()   
            columns =  ['Image', 'Fine_Prediction', 'Fine_Probability', 'is_in_validation']
            #todo: add regression
            
            #coarse_classes = classes[0]
            #classes = sorted(classes, key=lambda x: const.FLATTENED_INT[x]) #ordered according to integer values

            #pred_classes = [classes[idx.item()] for idx in torch.argmax(outputs, dim=1)]
            #sorted(classes[1], key=lambda x: const.FLATTENED_INT[x])
            
            if head == const.CLASSIFICATION:
                pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
            elif head == const.REGRESSION:
                pred_classes = [classes[idx.item()] for idx in pred_outputs.round().int()]
            elif head == const.CLM:
                pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
            elif head == const.CORN:
                pred_classes = [classes[idx.item()] for idx in corn_label_from_logits(pred_outputs)]
                
            probs, _ = torch.max(pred_outputs, dim=1)

            for image_id, pred, prob, is_vd, in zip(image_ids, pred_classes, probs.tolist(), is_valid_data):
                i = df.shape[0]
                df.loc[i, columns] = [float(image_id), pred, prob, is_vd]
                
            if save_features:          
                return df, pred_outputs, image_ids, features
            else:
                return df, pred_outputs, image_ids
         
        #CC   
        else:
            if save_features:
                all_features = {}
            #df = pd.DataFrame(columns=columns)
            level_name = model_dict.get('level', '')
            columns = ['Image', 'Prediction', 'Level', 'is_in_validation', f'Level_{level_no}'] # is_in_valid_dataset / join
            pre_cls_entry = []
            if pre_cls is not None:
                columns = columns + [f'Level_{level_no-1}']
                pre_cls_entry = [pre_cls]
              
            #print(pred_outputs)
            print(f"Level hier: {level}")
            print(f"level_name:{level_name}")
          
            if level_name == const.QUALITY:
                if head == const.REGRESSION:
                    #pred_classes = [classes[idx.item()] for idx in pred_outputs.round().int()]
                    pred_classes = ["outside" if str(pred.item()) not in classes.keys() else classes[str(pred.item())] for pred in pred_outputs.round().int()]
                elif head == const.CORN:
                    pred_classes = [classes[idx.item()] for idx in corn_label_from_logits(pred_outputs)]                                 
                else:
                    pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
            else:
                pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
            
            #print(classes)
            if level_name == const.TYPE:
                df_tmp = pd.DataFrame(columns=columns, index=range(pred_outputs.shape[0] * pred_outputs.shape[1]))
                i = 0
                for image_id, pred, is_vd in tqdm(zip(image_ids, pred_outputs, is_valid_data), desc="write df"):
                    for cls, prob in zip(classes, pred.tolist()):
                        df_tmp.iloc[i] = [image_id, prob, level_name, is_vd, cls] + pre_cls_entry
                        i += 1
                print(df_tmp.shape)
                df = pd.concat([df, df_tmp], ignore_index=True)
                print(df.shape)
                
            else:
                df_tmp = pd.DataFrame(columns=columns, index=range(0, len(pred_classes)))
                i = 0
                for image_id, pred, is_vd in tqdm(zip(image_ids, pred_classes, is_valid_data), desc="write df"):
                    df_tmp.iloc[i] = [image_id, pred, level_name, is_vd, pre_cls_entry] + pre_cls_entry
                    i += 1
                print(df_tmp.shape)
                df = pd.concat([df, df_tmp], ignore_index=True)
                print(df.shape)
            # subclasses not for regression implemented
            for cls in classes:
                sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
                sub_model_dict = model_dict.get('submodels', {}).get(cls)
                # print(sub_model_dict)
                # print("Predicted coarse classes:", pred_classes)
                # print("Submodel indices:", sub_indices)
                #print("Data passed to submodels:", [data[idx] for idx in sub_indices][:5]) 
                if not sub_indices or sub_model_dict is None:
                    continue
                sub_data = Subset(data, sub_indices)
                if save_features:
                    df, pred_outputs, image_ids, features = recursive_predict_csv(model_dict=sub_model_dict, 
                                            model_root=model_root, 
                                            data=sub_data, 
                                            batch_size=batch_size, 
                                            device=device, 
                                            level=level, 
                                            hierarchy_method=hierarchy_method, 
                                            save_features=save_features, 
                                            df=df, 
                                            level_no=level_no+1, 
                                            pre_cls=cls)
                    all_features[cls] = features
                else:
                    df, pred_outputs, image_ids = recursive_predict_csv(model_dict=sub_model_dict, 
                                            model_root=model_root, 
                                            data=sub_data, 
                                            batch_size=batch_size, 
                                            device=device, 
                                            level=level, 
                                            hierarchy_method=hierarchy_method, 
                                            save_features=save_features, 
                                            df=df, 
                                            level_no=level_no+1, 
                                            pre_cls=cls)
                # for image_id, pred, is_vd in zip(image_ids, pred_outputs, is_valid_data):
                #     for cls, prob in zip(classes, pred.tolist()):
                #         i = df.shape[0]
                #         df.loc[i, columns] = [image_id, prob, is_vd, cls] + pre_cls_entry
                # # subclasses not for regression implemented
                # for cls in classes:
                #     sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
                #     sub_model_dict = model_dict.get('submodels', {}).get(cls)
                #     if not sub_indices or sub_model_dict is None:
                #         continue
                #     sub_data = Subset(data, sub_indices)
                #     recursive_predict_csv(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device, level=const.SMOOTHNESS, 
                #                           hierarchy_method=hierarchy_method, save_features=save_features, df=df, level_no=level_no+1, pre_cls=cls)
        if save_features:     
            all_features[level_name] = features     
            return df, pred_outputs, image_ids, all_features
        else:
            return df, pred_outputs, image_ids
        
def predict(model, data, batch_size, head, level, device, save_features, seed):
        
    model.to(device)
    model.eval()

    loader = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle = False,
    )
    
    if level ==  const.HIERARCHICAL:
        coarse_outputs = []
        fine_outputs = []
        
        if save_features is True:
            feature_dict = {}
            
            #das alles nochmal überprüfen
            
            if 'B_CNN' in model.__module__:            
                h_1 = model.features[17].register_forward_hook(helper.make_hook("h1_features", feature_dict))
                h_2 = model.features[30].register_forward_hook(helper.make_hook("h2_features", feature_dict))
                
            elif 'C_CNN' in model.__module__:
                h_1 = model.block5_coarse[-1].register_forward_hook(helper.make_hook("h1_features", feature_dict))
                h_2 = model.block5_fine[-1].register_forward_hook(helper.make_hook("h2_features", feature_dict))
                
            elif 'GH_CNN' or 'H_NET' in model.__module__:
                h_1 = model.features[30].register_forward_hook(helper.make_hook("h1_features", feature_dict))
                h_2 = model.features[30].register_forward_hook(helper.make_hook("h2_features", feature_dict))
                
            else:
                print('No hooks for this model implemented')
                
        all_coarse_features = []
        all_fine_features = []
    
        
    else:
        outputs = []
        if save_features is True:
            feature_dict = {}
            if 'vgg16' in model.__module__:
                h_1 = model.features[30].register_forward_hook(helper.make_hook("h1_features", feature_dict))
                
        all_features = []
            
        
    ids = []
    
    #where we store intermediate outputs 
   

        #h_1 = model.block3_layer2.register_forward_hook(helper.make_hook("h1_features", feature_dict))
        #h_2 = model.block4_layer2.register_forward_hook(helper.make_hook("h2_features", feature_dict))
    
    with torch.no_grad():
        
        for index, (batch_inputs, batch_ids) in enumerate(loader):
            batch_inputs = batch_inputs.to(device)

            if level == const.HIERARCHICAL:
                model_inputs = (batch_inputs, None)
                coarse_batch_outputs, fine_batch_outputs = model(model_inputs)
                
                if 'C_CNN' in model.__module__:
                    print("CPWM:")
                    print(model.coarse_condition.weight.data)
                
                coarse_batch_outputs = model.get_class_probabilities(coarse_batch_outputs)
                
                if head == const.CLASSIFICATION:
                    fine_batch_outputs = model.get_class_probabilities(fine_batch_outputs)
                elif head == const.REGRESSION:
                    fine_batch_outputs = fine_batch_outputs.flatten()
                else:
                    pass # CLM outputs probs already and corn output is being transformed later
                 
                coarse_outputs.append(coarse_batch_outputs)
                fine_outputs.append(fine_batch_outputs)
                
            
            #Classifier Chain  or FLatten
            else:
                batch_outputs = model(batch_inputs)
                
                if level == "surface":
                    batch_outputs = model.get_class_probabilities(batch_outputs) 

                else:
                    if head == const.CLM:
                      # Access the last layer of the Sequential
                        clm_layer = model.classifier[-1][-1]  # Extract the CLM layer
                        thresholds_b = clm_layer.thresholds_b.detach().cpu().numpy()
                        thresholds_a = clm_layer.thresholds_a.detach().cpu().numpy()
                        converted_thresholds = clm_layer.convert_thresholds(
                            clm_layer.thresholds_b, clm_layer.thresholds_a, clm_layer.min_distance
                        ).detach().cpu().numpy()
                        
                        print("Raw thresholds (b):", thresholds_b)
                        print("Raw thresholds (a):", thresholds_a)
                        print("Converted thresholds:", converted_thresholds)
                        
                    if head == const.CLASSIFICATION:
                        batch_outputs = model.get_class_probabilities(batch_outputs) 
                    elif head == const.REGRESSION:
                        batch_outputs = batch_outputs.flatten()
                    else:
                        pass
                # if head == const.CLASSIFICATION:
                #     batch_outputs = model.get_class_probabilities(batch_outputs) 
                # elif head == const.REGRESSION:
                #     batch_outputs = batch_outputs.flatten()
                # else:
                #     pass
                    
                outputs.append(batch_outputs)
            
            ids.extend(batch_ids)
            
            if save_features:
                #flatten to vector
                if level == const.HIERARCHICAL:
                    for feature in feature_dict:
                        feature_dict[feature] = feature_dict[feature].view(feature_dict[feature].size(0), -1)
                        
                    all_coarse_features.append(feature_dict['h1_features'])
                    all_fine_features.append(feature_dict['h2_features'])
                
                else: 
                    for feature in feature_dict:
                        feature_dict[feature] = feature_dict[feature].view(feature_dict[feature].size(0), -1)
                        
                    all_features.append(feature_dict['h1_features'])
              
            # if index == 0:
            #     break 
    # h_1.remove()
    # h_2.remove()
    
    if level == const.HIERARCHICAL:
        pred_coarse_outputs = torch.cat(coarse_outputs, dim=0)
        pred_fine_outputs = torch.cat(fine_outputs, dim=0)
        
        if save_features: 
            all_coarse_features = torch.cat(all_coarse_features, dim=0)
            all_fine_features = torch.cat(all_fine_features, dim=0)
            h_1.remove()
            h_2.remove()
            all_features = [all_coarse_features, all_fine_features]
            return (pred_coarse_outputs, pred_fine_outputs), ids, all_features
        else:
            return (pred_coarse_outputs, pred_fine_outputs), ids

    else:
        pred_outputs = torch.cat(outputs, dim=0)
        if save_features:
            all_features = torch.cat(all_features, dim=0)
            h_1.remove()
            return pred_outputs, ids, all_features
        else:
            return pred_outputs, ids

def prepare_data(data_root, dataset, metadata, transform, ds_type=None):

    data_path = os.path.join(data_root, dataset, "s_1024")
    #data_path = os.path.join(data_root, dataset, "annotated", "asphalt")
    transform = preprocessing.transform(**transform)
    metadata_path = os.path.join(data_root, dataset, "metadata", metadata)
    predict_data = preprocessing.PredictImageFolder(
        root=data_path,
        csv_file=metadata_path,
        transform=transform,
        ds_type=ds_type,
    )
    
    return predict_data

# def create_dataloader(predict_data, batch_size, seed=None):
#     return DataLoader(
#         predict_data,
#         batch_size=batch_size,
#         shuffle=False,  
#         worker_init_fn=lambda _: helper.set_seed(seed) if seed is not None else None
#     )

def load_model(model_path, device):
    model_state = torch.load(model_path, map_location=device)
    model_cls = helper.string_to_object(model_state['config']['model'])
    hierarchy_method = model_state['config']["hierarchy_method"]
    head = model_state['config']["head"]
    level = model_state['config']["level"]
    valid_dataset = model_state['dataset']
    
    #Hierarhical
    if level == const.HIERARCHICAL: 
        fine_classes = valid_dataset.classes  #TODO: Adapt this to classes with clm and corn (-> dict like in to_train_list training.py)
        fine_classes = sorted(fine_classes, key=custom_sort_key)
        coarse_classes = list(OrderedDict.fromkeys(class_name.split('__')[0] for class_name in fine_classes))
        num_c = len(coarse_classes)
        num_classes = len(fine_classes) 
        model = model_cls(num_c = num_c, num_classes=num_classes, head=head, hierarchy_method=hierarchy_method) 
        model.load_state_dict(model_state['model_state_dict'])
        
        return model, (coarse_classes, fine_classes), head, level, valid_dataset, hierarchy_method, 
    
    #FLATTEN
    elif level == const.FLATTEN:
        classes = valid_dataset.classes
        classes = sorted(classes, key=custom_sort_key)
        num_classes = len(classes)
        model = model_cls(num_classes=num_classes, head=head) 
        model.load_state_dict(model_state['model_state_dict'])
        
        return model, classes, head, level, valid_dataset, hierarchy_method  
    #CC 
    #Surface type
    elif level == const.SURFACE:
        classes = valid_dataset.classes
        num_classes = len(classes)
        model = model_cls(num_classes=num_classes, head=head) 
        model.load_state_dict(model_state['model_state_dict'])
        
        return model, classes, head, level, valid_dataset, hierarchy_method
    #quality type
    else:
        if head == const.REGRESSION:
            class_to_idx = valid_dataset.class_to_idx
            classes = {str(i): cls for cls, i in class_to_idx.items()}
            num_classes = 1
        #add clm and corn
        else:
            classes = valid_dataset.classes
            classes = sorted(classes, key=custom_sort_key)
            #classes = sorted(classes, key=lambda cls: constants.SMOOTHNESS_INT.get(cls, float('inf')))
            num_classes = len(classes)
        model = model_cls(num_classes=num_classes, head=head) 
        model.load_state_dict(model_state['model_state_dict'])
        
        return model, classes, head, level, valid_dataset, hierarchy_method

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
    
    
def save_cam_hierarchical(model, data, normalize_transform, classes, valid_dataset, head, level, hierarchy_method, device, cam_folder):

    #feature_layer = model.fine_classifier[-4]
    if 'C_CNN' in model.__module__:
        feature_layer = model.block5_fine[0][-3]
    else:
        feature_layer = model.features[-3]
    #out_weights_coarse = model.coarse_classifier[-1].weight #TODO adapt for multiple heads 
    out_weights_fine_reduced = helper.get_fine_weights(model, level, head)
    
    model.to(device)
    model.eval()
   
    coarse_classes, fine_classes = classes
    #fine_classes = sorted(fine_classes, key=lambda x: const.FLATTENED_INT[x])

    valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
    
    with torch.no_grad():
        with helper.ActivationHook(feature_layer) as activation_hook:
        
            for image, image_id in data:
                input = normalize_transform(image).unsqueeze(0).to(device)
                model_inputs = (input, None)
                
                coarse_output, fine_output = model(model_inputs)
                coarse_output = model.get_class_probabilities(coarse_output).squeeze(0)
                coarse_pred_value = torch.max(coarse_output, dim=0).values.item()
                coarse_idx = torch.argmax(coarse_output, dim=0).item()
                coarse_pred_class = coarse_classes[coarse_idx]
                # TODO: wie sinnvoll ist class activation map bei regression?
                #if head == const.REGRESSION:                    
                    # fine_output = fine_output.flatten().squeeze(0)
                    # fine_pred_value = fine_output.item()
                    # fine_idx = 0
                    # fine_pred_class = "outside" if str(round(fine_pred_value)) not in fine_classes.keys() else fine_classes[str(round(fine_pred_value))]
                    
                #elif head == const.CORN TODO: add corn
                #elif head == const.CLM:
                if head == const.CLASSIFICATION:                  
                    fine_output = model.get_class_probabilities(fine_output).squeeze(0)
                
                fine_idx, fine_pred_class = helper.compute_fine_prediction_hierarchical(fine_output=fine_output, 
                                                                                                            coarse_filter=coarse_idx, 
                                                                                                            hierarchy_method=hierarchy_method, 
                                                                                                            head=head,
                                                                                                            fine_classes=fine_classes)

                # create cam
                activations = activation_hook.activation[0]
                cam_maps = {}
                if head == const.CLASSIFICATION:
                    cam_map_fine = torch.einsum('ck,kij->cij', out_weights_fine_reduced, activations)
                    cam_maps["combined"] = cam_map_fine
                else:
                    # Store each CAM in the dictionary
                    cam_maps["asphalt"] = helper.generate_cam(activations, out_weights_fine_asphalt)
                    cam_maps["concrete"] = helper.generate_cam(activations, out_weights_fine_concrete)
                    cam_maps["paving_stones"] = helper.generate_cam(activations, out_weights_fine_paving_stones)
                    cam_maps["sett"] = helper.generate_cam(activations, out_weights_fine_sett)
                    cam_maps["unpaved"] = helper.generate_cam(activations, out_weights_fine_unpaved)

                #coarse_text = 'validation_data: {}\nprediction: {}\nvalue: {:.3f}'.format('True' if image_id in valid_dataset_ids else 'False', coarse_pred_class, coarse_pred_value)
                #fine_text = 'validation_data: {}\nprediction: {}\nvalue: {:.3f}'.format('True' if image_id in valid_dataset_ids else 'False', fine_pred_class, fine_pred_value)

                #n_coarse_classes = len(coarse_classes)
                if head == const.REGRESSION:
                    n_fine_classes = 1 
                elif head == const.CORN:
                    if coarse_idx == 0 or coarse_idx == 1 or coarse_idx == 2:
                        n_fine_classes = 3
                    else:
                        n_fine_classes = 2
                
                else:
                    n_fine_classes = len(fine_classes)
                
                to_pil = ToPILImage()  # Initialize the transform
                original_pil_image = to_pil(image.cpu())  
                original_image_path = os.path.join(cam_folder, f"{image_id}_original.jpg")
                original_pil_image.save(original_image_path)
                print(f"Original image saved at {original_image_path}")
                 
                for class_key, cam_map in cam_maps.items():
                    for i in range(n_fine_classes):
                        class_name = fine_classes[i]

                        # Create a figure for each CAM
                        fig, ax = plt.subplots()
                        
                        # Show the CAM for the specific class and fine class
                        ax.imshow(cam_map[i].detach().cpu(), alpha=1.0, extent=(0, 48, 48, 0), interpolation='bicubic', cmap='magma')
                        ax.axis('off')
                        
                        # Save with different filenames for predicted vs non-predicted classes
                        if i == fine_idx:
                            class_image_path = os.path.join(cam_folder, f"{image_id}_{class_key}_{class_name}_predicted_cam.jpg")
                            ax.set_title(f"Predicted: {class_name}")
                        else:
                            class_image_path = os.path.join(cam_folder, f"{image_id}_{class_key}_{class_name}_cam.jpg")
                        
                        plt.savefig(class_image_path, bbox_inches='tight', pad_inches=0)
                        plt.close()

                print(f"CAM images saved in {cam_folder}")  

def save_cam_flattened(model, data, normalize_transform, classes, valid_dataset, head, hierarchy_method, level, device, cam_folder):
    # Feature layer for generating CAM
    feature_layer = model.features[-3]  # Adjust if your feature extraction layer differs
    #out_weights_fine = model.classifier[-1].weight  # Assuming a single output layer for classification
    out_weights_fine = helper.get_fine_weights(model, level, head)
    model.to(device)
    model.eval()
    
    classes = sorted(classes, key=lambda x: const.FLATTENED_INT[x])  # Sort fine classes if needed

    # Extract valid dataset IDs for validation images
    valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
    
    with torch.no_grad():
        with helper.ActivationHook(feature_layer) as activation_hook:
            for image, image_id in data:
                # Preprocess the input image
                input_image = normalize_transform(image).unsqueeze(0).to(device)
                
                # Forward pass
                fine_output = model(input_image)
                if head == const.CLASSIFICATION:
                    fine_output = model.get_class_probabilities(fine_output).squeeze(0)
                
                # Determine predicted class and value
                fine_pred_value = torch.max(fine_output, dim=0).values.item()
                fine_idx = torch.argmax(fine_output, dim=0).item()
                fine_pred_class = classes[fine_idx]
                
                # Generate CAMs
                activations = activation_hook.activation[0]
                cam_map_fine = torch.einsum('ck,kij->cij', out_weights_fine, activations)
                
                # Save original image
                to_pil = ToPILImage()
                original_pil_image = to_pil(image.cpu())
                original_image_path = os.path.join(cam_folder, f"{image_id}_original.jpg")
                original_pil_image.save(original_image_path)
                print(f"Original image saved at {original_image_path}")
                
                # Create and save CAM images for each class
                for i in range(len(classes)):
                    class_name = classes[i]
                    fig, ax = plt.subplots()
                    
                    # Display the CAM
                    ax.imshow(cam_map_fine[i].detach().cpu(), alpha=1.0, extent=(0, 48, 48, 0), interpolation='bicubic', cmap='magma')
                    ax.axis('off')
                    
                    # Save with different filenames for predicted vs non-predicted classes
                    if i == fine_idx:
                        class_image_path = os.path.join(cam_folder, f"{image_id}_{class_name}_predicted_cam.jpg")
                        ax.set_title(f"Predicted: {class_name}")
                    else:
                        class_image_path = os.path.join(cam_folder, f"{image_id}_{class_name}_cam.jpg")
                    
                    plt.savefig(class_image_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                print(f"CAM images saved in {cam_folder}")

                #break
                
                
                
                #fine cam
                # fig, ax = plt.subplots(1, n_fine_classes+1, figsize=((n_fine_classes+1)*2.5, 2.5))

                # ax[0].imshow(image.permute(1, 2, 0))
                # ax[0].axis('off')

                # for i in range(1, n_fine_classes+1):
                    
                #     # merge original image with cam
                    
                #     ax[i].imshow(image.permute(1, 2, 0))

                #     #ax[i].imshow(cam_map_coarse[i-1].detach(), alpha=0.75, extent=(0, image.shape[2], image.shape[1], 0),
                #             #interpolation='bicubic', cmap='magma')
                    
                #     ax[i].imshow(cam_map_fine[i-1].detach(), alpha=0.75, extent=(0, image.shape[2], image.shape[1], 0),
                #             interpolation='bicubic', cmap='magma')

                #     ax[i].axis('off')

                    # if i - 1 == idx:
                    #     # draw prediction on image
                    #     ax[i].text(10, 80, text, color='white', fontsize=6)
                    # else:
                    #     t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], output[i - 1].item())
                    #     ax[i].text(10, 80, t, color='white', fontsize=6)

                #     t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], fine_output[i - 1].item())
                #     ax[i].fine_text(10, 60, t, color='white', fontsize=6)

                #     # save image
                #     # image_path = os.path.join(image_folder, "{}_cam.png".format(image_id))
                #     # plt.savefig(image_path)

                #     # show image
                # plt.show()
                # plt.close() 
            
        
# def predict_surface_and_quality(model_root, model_dict, data, device, batch_size, save_features, seed):
    
#     all_CC_features = {}
#     # Load surface type model
#     surface_model_path = os.path.join(model_root, model_dict['trained_model'])
#     surface_model, surface_classes, head, level, valid_dataset, hierarchy_method = load_model(surface_model_path, device)

#     # Predict surface type
#     if save_features:
#         surface_predictions, image_ids, surface_features = predict(surface_model, data, batch_size, head, level, device, save_features=save_features, seed=seed)
#         all_CC_features['surface'] = surface_features

#     else:
#         surface_predictions, image_ids = predict(surface_model, data, batch_size, head, level, device, save_features=save_features, seed=seed)

#     # Get predicted classes for surface type
#     predicted_surface_classes = [surface_classes[idx.item()] for idx in torch.argmax(surface_predictions, dim=1)]

#     valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
#     is_valid_data = [1 if image_id in valid_dataset_ids else 0 for image_id in image_ids]
        
#     # Group images by predicted surface type
#     grouped_images = {}
#     for image_id, pred_surface_class, is_valid in zip(image_ids, predicted_surface_classes, is_valid_data):
#         if pred_surface_class not in grouped_images:
#             grouped_images[pred_surface_class] = []
#         grouped_images[pred_surface_class].append((image_id, is_valid))

#     # Dictionary to store results
#     results = []

#     # Predict quality for each group
#     for surface_type, images in grouped_images.items():
#         submodel_dict = model_dict['submodels'].get(surface_type)
#         if not submodel_dict:
#             continue  # Skip if no submodel is defined for this surface type

#         # Load the submodel for this surface type
#         submodel_path = os.path.join(model_root, submodel_dict['trained_model'])
#         submodel, quality_classes, _, _, _, _ = load_model(submodel_path, device) #TODO ggf sortieren noch
        
#         image_id_to_index = {image_id: idx for idx, (image_data, image_id) in enumerate(data)}
#         sub_data_indices = [image_id_to_index[image_id] for image_id, _ in images if image_id in image_id_to_index]
#         subset = Subset(data, sub_data_indices)
#         # image_id_to_index = {}
        # sub_data = [image_id for image_id, _ in images]
        
        # for idx, (image_data, image_id) in enumerate(data):
        #     image_id_to_index[image_id] = idx

        # # Step 2: Get the indices for the subset based on your sub_data list of image IDs
        # sub_data_indices = [image_id_to_index[image_id] for image_id in sub_data if image_id in image_id_to_index] #TODO: nicht ganz sicher, ob das so richtig ist
        # # Step 3: Create a subset using the indices
        # subset = Subset(data, sub_data_indices)

        # Prepare data for this specific surface type
#         sub_data = [image_id for image_id, _ in images]  # Extract image IDs #TODO hier weitermachen
#         image_id_to_index = {os.path.splitext(os.path.basename(sample[0]))[0]: idx for idx, sample in enumerate(data.samples)}

# # Filter the indices using your `sub_data`
#         sub_data_indices = [image_id_to_index[image_id] for image_id in sub_data if image_id in image_id_to_index]

# Create the subset using these indices
        #subset = Subset(data, sub_data_indices)
        # sub_data_loader = DataLoader(
        #     subset,
        #     batch_size=batch_size,sc
        #     shuffle=False
        # )
        # Predict quality using the submodel
        if save_features:
            quality_predictions, _ , quality_features= predict(submodel, subset, batch_size, head, level, device, save_features=save_features, seed=seed)
            all_CC_features[surface_type] = quality_features

        else:
            quality_predictions, _ = predict(submodel, subset, batch_size, head, level, device, save_features=save_features, seed=seed)
        
        predicted_quality_classes = [quality_classes[idx.item()] for idx in torch.argmax(quality_predictions, dim=1)]

        # Store results
        for (image_id, is_valid), quality_pred in zip(images, predicted_quality_classes):
            results.append({
                'image_id': image_id,
                'is_in_validation': is_valid,
                'surface_type': surface_type,
                'quality_prediction': quality_pred
            })

    # Convert results to DataFrame (or any desired output format)
    df_results = pd.DataFrame(results)
    if save_features:
        return df_results, all_CC_features
    else:
        return df_results


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

def custom_sort_key(class_name):
    if "__" in class_name:  # Case for full classes like "asphalt__bad"
        surface, condition = class_name.split("__")
        return (surface, const.condition_order[condition])
    else:  # Case for standalone conditions like "bad", "excellent"
        return const.condition_order[class_name]


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
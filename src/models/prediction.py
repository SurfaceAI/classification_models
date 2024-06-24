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
from src import constants
from experiments.config import global_config
from src.architecture import Rateke_CNN
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import argparse
import matplotlib.pyplot as plt

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
    predict_data = prepare_data(config.get("root_data"), config.get("dataset"), non_normalize_transform)

    model_path = os.path.join(config.get("root_model"), config.get("model_dict")['trained_model'])
    model, classes, is_regression, valid_dataset = load_model(model_path=model_path, device=device)
    image_folder = os.path.join(config.get("root_predict"), config.get("dataset"))
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    save_cam(model, predict_data, normalize_transform, classes, valid_dataset, is_regression, device, image_folder)

    print(f'Images {config.get("dataset")} predicted and saved with CAM: {image_folder}')

    
def run_dataset_predict_csv(config):
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    predict_data = prepare_data(config.get("root_data"), config.get("dataset"), config.get("transform"))

    level = 0
    columns = ['Image', 'Prediction', 'Level', f'Level_{level}']
    df = pd.DataFrame(columns=columns)

    recursive_predict_csv(model_dict=config.get("model_dict"), model_root=config.get("root_model"), data=predict_data, batch_size=config.get("batch_size"), device=device, df=df, level=level)

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = config.get("name") + '-' + config.get("dataset").replace('/', '_') + '-' + start_time + '.csv'

    saving_path = save_predictions_csv(df=df, saving_dir=config.get("root_predict"), saving_name=saving_name)

    print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')

def recursive_predict_csv(model_dict, model_root, data, batch_size, device, df, level, pre_cls=None):

    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict['trained_model'])
        level_name = model_dict.get('level', '')
        model, classes, is_regression, valid_dataset = load_model(model_path=model_path, device=device)
        
        pred_outputs, image_ids = predict(model, data, batch_size, is_regression, device)

        # compare valid dataset 
        # [image_id in valid_dataset ]
        valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
        is_valid_data = [1 if image_id in valid_dataset_ids else 0 for image_id in image_ids]
        
        columns = ['Image', 'Prediction', 'Level', 'is_in_validation', f'Level_{level}'] # is_in_valid_dataset / join
        pre_cls_entry = []
        if pre_cls is not None:
            columns = columns + [f'Level_{level-1}']
            pre_cls_entry = [pre_cls]
        if is_regression:
            pred_classes = ["outside" if str(pred.item()) not in classes.keys() else classes[str(pred.item())] for pred in pred_outputs.round().int()]
            for image_id, pred, is_vd, cls in zip(image_ids, pred_outputs, is_valid_data, pred_classes):
                i = df.shape[0]
                df.loc[i, columns] = [image_id, pred.item(), level_name, is_vd, cls] + pre_cls_entry
        else:
            pred_classes = [classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)]
            for image_id, pred, is_vd in zip(image_ids, pred_outputs, is_valid_data):
                for cls, prob in zip(classes, pred.tolist()):
                    i = df.shape[0]
                    df.loc[i, columns] = [image_id, prob, level_name, is_vd, cls] + pre_cls_entry
            # subclasses not for regression implemented
            for cls in classes:
                sub_indices = [idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls]
                sub_model_dict = model_dict.get('submodels', {}).get(cls)
                if not sub_indices or sub_model_dict is None:
                    continue
                sub_data = Subset(data, sub_indices)
                recursive_predict_csv(model_dict=sub_model_dict, model_root=model_root, data=sub_data, batch_size=batch_size, device=device, df=df, level=level+1, pre_cls=cls)



def predict(model, data, batch_size, is_regression, device):
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
    
            batch_outputs = model(batch_inputs)
            if is_regression:
                batch_outputs = batch_outputs.flatten()
            else:
                batch_outputs = model.get_class_probabilies(batch_outputs)

            outputs.append(batch_outputs)
            ids.extend(batch_ids)

    pred_outputs = torch.cat(outputs, dim=0)

    return pred_outputs, ids

def save_cam(model, data, normalize_transform, classes, valid_dataset, is_regression, device, image_folder):

    feature_layer = model.features
    out_weights = model.classifier[-1].weight

    model.to(device)
    model.eval()

    valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
    
    with torch.no_grad() and helper.ActivationHook(feature_layer) as activation_hook:
        
        for image, image_id in data:
            input = normalize_transform(image).unsqueeze(0).to(device)
            
            output = model(input)
            # TODO: wie sinnvoll ist class activation map bei regression?
            if is_regression:
                output = output.flatten().squeeze(0)
                pred_value = output.item()
                idx = 0
                pred_class = "outside" if str(round(pred_value)) not in classes.keys() else classes[str(round(pred_value))]
            else:
                output = model.get_class_probabilies(output).squeeze(0)
                pred_value = torch.max(output, dim=0).values.item()
                idx = torch.argmax(output, dim=0).item()
                pred_class = classes[idx]

            # create cam
            activations = activation_hook.activation[0]
            cam_map = torch.einsum('ck,kij->cij', out_weights, activations).cpu()

            text = 'validation_data: {}\nprediction: {}\nvalue: {:.3f}'.format('True' if image_id in valid_dataset_ids else 'False', pred_class, pred_value)
            
            n_classes = 1 if is_regression else len(classes)

            # fig, ax = plt.subplots(1, n_classes+1, figsize=((n_classes+1)*2.5, 2.5))

            # ax[0].imshow(image.permute(1, 2, 0))
            # ax[0].axis('off')

            # for i in range(1, n_classes+1):
                
            #     # merge original image with cam
                
            #     ax[i].imshow(image.permute(1, 2, 0))

            #     ax[i].imshow(cam_map[i-1].detach(), alpha=0.75, extent=(0, image.shape[2], image.shape[1], 0),
            #             interpolation='bicubic', cmap='magma')

            #     ax[i].axis('off')

            #     # if i - 1 == idx:
            #     #     # draw prediction on image
            #     #     ax[i].text(10, 80, text, color='white', fontsize=6)
            #     # else:
            #     #     t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], output[i - 1].item())
            #     #     ax[i].text(10, 80, t, color='white', fontsize=6)

            #     t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], output[i - 1].item())
            #     ax[i].text(10, 60, t, color='white', fontsize=6)

            #     # save image
            #     # image_path = os.path.join(image_folder, "{}_cam.png".format(image_id))
            #     # plt.savefig(image_path)

            #     # show image
            # plt.show()
            # plt.close()
           
            for i in range(n_classes):
                class_name = classes[i]
                # Erstelle eine neue Figur für jedes Bild
                fig, ax = plt.subplots()
                # ax.imshow(image.permute(1, 2, 0))
                ax.imshow(cam_map[i].detach(), alpha=1.0, extent=(0, 48, 48, 0),
                        interpolation='bicubic', cmap='magma')
                ax.axis('off')
                
                # Speichere das Bild mit dem Klassennamen als Präfix
                class_image_path = os.path.join(image_folder, f"{image_id}_{class_name}_cam.jpg")
                plt.savefig(class_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            print(f"Images saved in {image_folder}")   
            # break        


def prepare_data(data_root, dataset, transform):

    data_path = os.path.join(data_root, dataset)
    transform = preprocessing.transform(**transform)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    return predict_data

def load_model(model_path, device):
    model_state = torch.load(model_path, map_location=device)
    model_cls = helper.string_to_object(model_state['config']['model'])
    is_regression = model_state['config']["is_regression"]
    # is_regression = False
    valid_dataset = model_state['dataset']

    if is_regression:
        class_to_idx = valid_dataset.class_to_idx
        classes = {str(i): cls for cls, i in class_to_idx.items()}
        num_classes = 1
    else:
        classes = valid_dataset.classes
        num_classes = len(classes)
    model = model_cls(num_classes)
    model.load_state_dict(model_state['model_state_dict'])

    return model, classes, is_regression, valid_dataset

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
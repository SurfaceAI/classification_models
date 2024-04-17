import sys

sys.path.append(".")
sys.path.append("..")

import torch
import os
import json
from src.utils import preprocessing
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime
import time
from src.utils import helper, mapillary_requests, mapillary_detections
from src import constants
from experiments.config import global_config
from src.architecture import Rateke_CNN
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from shapely import box, intersection
from functools import partial

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
    predict_data = prepare_data(
        config.get("root_data"), config.get("dataset"), config.get("transform")
    )

    level = 0
    columns = ["Image", "Prediction", f"Level_{level}"]
    df = pd.DataFrame(columns=columns)

    recursive_predict_csv(
        model_dict=config.get("model_dict"),
        model_root=config.get("root_model"),
        data=predict_data,
        batch_size=config.get("batch_size"),
        device=device,
        df=df,
        level=level,
    )

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = (
        config.get("name")
        + "-"
        + config.get("dataset").replace("/", "_")
        + "-"
        + start_time
        + ".csv"
    )

    saving_path = save_predictions_csv(
        df=df, saving_dir=config.get("root_predict"), saving_name=saving_name
    )

    print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')

def run_image_segmentation(config):
    # prepare data
    segment_data = prepare_data(config.get("root_data"), config.get("dataset"))

    segmentation_selection = partial(helper.string_to_object(config.get("segmentation_selection_func")), config=config)

    saving_folder = os.path.join(config.get("root_data"), config.get("segmentation_folder"), config.get("segmentation_selection_func"))

    # for debugging only
    count = 0

    for image, image_id in segment_data:
        # for debugging only
        count += 1
        # if count > 5:
        #     break
        print(image_id)

        image_segmentation_file = os.path.join(config.get("root_data"), config.get("segmentation_folder"), 'detections', '{}_{}.geojson'.format(image_id, config.get("saving_postfix")))
        with open(image_segmentation_file, 'r') as file:
            detections = json.load(file)
        detections = mapillary_requests.extract_detections_from_image(
            detections
        )

        segmentation_properties_list = segmentation_selection(detections)
        
        for value, polygon in segmentation_properties_list:

            if value == 'not_completely_segmented':
                continue
        
            # for debugging only
            print(value)

            image_det = image.copy()
            draw_det = ImageDraw.Draw(image_det)
            
            draw_det.polygon(
                np.multiply(polygon.exterior.coords, image_det.size)
                .flatten()
                .tolist(),
                fill=config.get("segment_color")[value],
                outline="blue",
            )

            composite = Image.blend(image, image_det, 0.2)

            # composite.show()
        
            # save image with detections
            if not os.path.exists(saving_folder):
                os.makedirs(saving_folder)
            image_path = os.path.join(saving_folder, "{}_{}.png".format(image_id, value))
            composite.save(image_path)
            # composite.close()

    print(f'Images {config.get("dataset")} segmented and saved.')

def run_dataset_predict_segmentation(config):
    
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

    segmentation_selection = partial(helper.string_to_object(config.get("segmentation_selection_func")), config=config)

    individual_transform_generator = preprocessing.extract_segmentation_properties(segmentation_path=os.path.join(config.get("root_data"), config.get("segmentation_folder"), 'detections'),
                                                                                   postfix=config.get("saving_postfix"),
                                                                                   segmentation_selection=segmentation_selection,
                                                                                   mask_style=config.get("seg_mask_style"),
                                                                                   crop_style=config.get("seg_crop_style"),
                                                                                   )

    # prepare data
    data = segmentation_data(config.get("root_data"), config.get("dataset"), config.get("transform"), individual_transform_generator)

    level = 0
    columns = ["Image", "Segment", "Prediction", f"Level_{level}"]
    df = pd.DataFrame(columns=columns)

    recursive_predict_csv_segment(
        model_dict=config.get("model_dict"),
        model_root=config.get("root_model"),
        data=data,
        batch_size=config.get("batch_size"),
        device=device,
        df=df,
        level=level,
    )

    # save predictions
    start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    saving_name = (
        config.get("name")
        + "-"
        + config.get("dataset").replace("/", "_")
        + "-"
        + start_time
        # + ".csv"
    )

    saving_path = save_predictions_csv(
        df=df, saving_dir=config.get("root_predict"), saving_name=saving_name + ".csv"
    )
    save_config(config=config, saving_dir=config.get("root_predict"), saving_name=saving_name + "_config.json")

    print(f'Images {config.get("dataset")} predicted and saved: {saving_path}')
    
def recursive_image_per_image_predict_csv(
    model_dict,
    model_root,
    image_id,
    segment,
    image_transformed,
    device,
    df,
    level,
    pre_cls=None,
):
    text = ""
    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict["trained_model"])
        model, classes, is_regression, valid_dataset = load_model(
            model_path=model_path, device=device
        )

        pred_output = predict_image_per_image(
            model, image_transformed, is_regression, device
        )

        # compare valid dataset
        # [image_id in valid_dataset ]
        valid_dataset_ids = [
            os.path.splitext(os.path.split(id[0])[-1])[0]
            for id in valid_dataset.samples
        ]
        is_valid_data = 1 if image_id in valid_dataset_ids else 0

        columns = [
            "Image",
            "Segment",
            "Prediction",
            "is_in_validation",
            f"Level_{level}",
        ]  # is_in_valid_dataset / join
        pre_cls_entry = []
        if pre_cls is not None:
            columns = columns + [f"Level_{level-1}"]
            pre_cls_entry = [pre_cls]

        pred_value, pred_class = convert_forward_step_outputs(outputs=pred_output, classes=classes, is_regression=is_regression)
        pred_value, pred_class = pred_value[0], pred_class[0]

        if is_regression:
            i = df.shape[0]
            df.loc[i, columns] = [
                image_id,
                segment,
                pred_value,
                is_valid_data,
                pred_class,
            ] + pre_cls_entry
            text = cls + ": " + f"{pred_value:.3f}"
        else:
            for cls, prob in zip(classes, pred_value):
                i = df.shape[0]
                df.loc[i, columns] = [
                    image_id,
                    segment,
                    prob,
                    is_valid_data,
                    cls,
                ] + pre_cls_entry
            # subclasses not for regression implemented
            sub_model_dict = model_dict.get("submodels", {}).get(pred_class)
            sub_text = recursive_image_per_image_predict_csv(
                model_dict=sub_model_dict,
                model_root=model_root,
                image_id=image_id,
                segment=segment,
                image_transformed=image_transformed,
                device=device,
                df=df,
                level=level + 1,
                pre_cls=pred_class,
            )
            text = (
                pred_class
                + ": "
                + f"{max(pred_value):.3f}"
                + "\n"
                + sub_text
            )

    return text

def recursive_predict_csv(
    model_dict, model_root, data, batch_size, device, df, level, pre_cls=None
):
    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict["trained_model"])
        model, classes, is_regression, valid_dataset = load_model(
            model_path=model_path, device=device
        )

        pred_outputs, image_ids = predict(
            model, data, batch_size, is_regression, device
        )

        # compare valid dataset
        # [image_id in valid_dataset ]
        valid_dataset_ids = [
            os.path.splitext(os.path.split(id[0])[-1])[0]
            for id in valid_dataset.samples
        ]
        is_valid_data = [
            1 if image_id in valid_dataset_ids else 0 for image_id in image_ids
        ]

        columns = [
            "Image",
            "Prediction",
            "is_in_validation",
            f"Level_{level}",
        ]  # is_in_valid_dataset / join
        pre_cls_entry = []
        if pre_cls is not None:
            columns = columns + [f"Level_{level-1}"]
            pre_cls_entry = [pre_cls]

        pred_values, pred_classes = convert_forward_step_outputs(outputs=pred_outputs, classes=classes, is_regression=is_regression)
            
        if is_regression:
            for image_id, pred, is_vd, cls in zip(
                image_ids, pred_values, is_valid_data, pred_classes
            ):
                i = df.shape[0]
                df.loc[i, columns] = [image_id, pred, is_vd, cls] + pre_cls_entry
        else:
            for image_id, pred, is_vd in zip(image_ids, pred_values, is_valid_data):
                for cls, prob in zip(classes, pred):
                    i = df.shape[0]
                    df.loc[i, columns] = [image_id, prob, is_vd, cls] + pre_cls_entry
            # subclasses not for regression implemented
            for cls in classes:
                sub_indices = [
                    idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls
                ]
                sub_model_dict = model_dict.get("submodels", {}).get(cls)
                if not sub_indices or sub_model_dict is None:
                    continue
                sub_data = Subset(data, sub_indices)
                recursive_predict_csv(
                    model_dict=sub_model_dict,
                    model_root=model_root,
                    data=sub_data,
                    batch_size=batch_size,
                    device=device,
                    df=df,
                    level=level + 1,
                    pre_cls=cls,
                )

def recursive_predict_csv_segment(
    model_dict, model_root, data, batch_size, device, df, level, pre_cls=None
):
    # base:
    if model_dict is None:
        # predictions = None
        pass
    else:
        model_path = os.path.join(model_root, model_dict["trained_model"])
        model, classes, is_regression, valid_dataset = load_model(
            model_path=model_path, device=device
        )

        pred_outputs, image_ids, segments = predict_segment(
            model, data, batch_size, is_regression, device
        )

        # compare valid dataset
        # [image_id in valid_dataset ]
        valid_dataset_ids = [
            os.path.splitext(os.path.split(id[0])[-1])[0]
            for id in valid_dataset.samples
        ]
        is_valid_data = [
            1 if image_id in valid_dataset_ids else 0 for image_id in image_ids
        ]

        columns = [
            "Image",
            "Segment",
            "Prediction",
            "is_in_validation",
            f"Level_{level}",
        ]  # is_in_valid_dataset / join
        pre_cls_entry = []
        if pre_cls is not None:
            columns = columns + [f"Level_{level-1}"]
            pre_cls_entry = [pre_cls]

        pred_values, pred_classes = convert_forward_step_outputs(outputs=pred_outputs, classes=classes, is_regression=is_regression)
            
        if is_regression:
            for image_id, segment, pred, is_vd, cls in zip(
                image_ids, segments, pred_values, is_valid_data, pred_classes
            ):
                i = df.shape[0]
                df.loc[i, columns] = [image_id, segment, pred, is_vd, cls] + pre_cls_entry
        else:
            for image_id, segment, pred, is_vd in zip(image_ids, segments, pred_values, is_valid_data):
                for cls, prob in zip(classes, pred):
                    i = df.shape[0]
                    df.loc[i, columns] = [image_id, segment, prob, is_vd, cls] + pre_cls_entry
            # subclasses not for regression implemented
            for cls in classes:
                sub_indices = [
                    idx for idx, pred_cls in enumerate(pred_classes) if pred_cls == cls
                ]
                sub_model_dict = model_dict.get("submodels", {}).get(cls)
                if not sub_indices or sub_model_dict is None:
                    continue
                sub_data = Subset(data, sub_indices)
                recursive_predict_csv_segment(
                    model_dict=sub_model_dict,
                    model_root=model_root,
                    data=sub_data,
                    batch_size=batch_size,
                    device=device,
                    df=df,
                    level=level + 1,
                    pre_cls=cls,
                )


def predict_image_per_image(model, image, is_regression, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        input = torch.unsqueeze(image, 0)

        output = batch_forward_step(model_to_device=model,
                                    batch_inputs=input,
                                    is_regression=is_regression,
                                    device=device)

    # return torch.squeeze(output, 0)
    return output


def predict(model, data, batch_size, is_regression, device):
    model.to(device)
    model.eval()

    loader = DataLoader(data, batch_size=batch_size)

    outputs = []
    ids = []
    with torch.no_grad():
        for batch_inputs, batch_ids in loader:
            batch_outputs = batch_forward_step(model_to_device=model, batch_inputs=batch_inputs, is_regression=is_regression, device=device)
            outputs.append(batch_outputs)
            ids.extend(batch_ids)

    pred_outputs = torch.cat(outputs, dim=0)

    return pred_outputs, ids

def predict_segment(model, data, batch_size, is_regression, device):
    model.to(device)
    model.eval()

    loader = DataLoader(data, batch_size=batch_size)

    outputs = []
    ids = []
    segments = []
    with torch.no_grad():
        for batch_inputs, batch_ids, batch_segments in loader:

            # for debugging pnly
            # helper.imshow(batch_inputs[0])

            batch_outputs = batch_forward_step(model_to_device=model, batch_inputs=batch_inputs, is_regression=is_regression, device=device)
            outputs.append(batch_outputs)
            ids.extend(batch_ids)
            segments.extend(batch_segments)


    pred_outputs = torch.cat(outputs, dim=0)

    return pred_outputs, ids, segments

def batch_forward_step(model_to_device, batch_inputs, is_regression, device):
    batch_inputs = batch_inputs.to(device)
    batch_outputs = model_to_device(batch_inputs)

    if is_regression:
        batch_outputs = batch_outputs.flatten()
    else:
        batch_outputs = model_to_device.get_class_probabilies(batch_outputs)

    return batch_outputs

def convert_forward_step_outputs(outputs, classes, is_regression):
    # batch
    if is_regression:
        pred_values = [output.item() for output in outputs]
        rounded_pred_values = [str(round(pred_value)) for pred_value in pred_values]
        pred_classes = [classes.get(value, "outside") for value in rounded_pred_values]    
    else:
        pred_values = [output.tolist() for output in outputs]
        pred_classes = [
            classes[idx.item()] for idx in torch.argmax(outputs, dim=1)
        ]

    return pred_values, pred_classes

def save_cam(model, data, normalize_transform, classes, valid_dataset, is_regression, device, image_folder):

    feature_layer = model.features
    out_weights = model.classifier[-1].weight

    model.to(device)
    model.eval()

    valid_dataset_ids = [os.path.splitext(os.path.split(id[0])[-1])[0] for id in valid_dataset.samples]
    
    with torch.no_grad() and helper.ActivationHook(feature_layer) as activation_hook:
        
        for image, image_id in data:
            input = normalize_transform(image).unsqueeze(0)
            
            pred_output = batch_forward_step(model_to_device=model, batch_inputs=input, is_regression=is_regression, device=device)
            pred_value, pred_class = convert_forward_step_outputs(outputs=pred_output, classes=classes, is_regression=is_regression)
            pred_value, pred_class = pred_value[0], pred_class[0]

            # TODO: wie sinnvoll ist class activation map bei regression?

            # create cam
            activations = activation_hook.activation[0]
            cam_map = torch.einsum('ck,kij->cij', out_weights, activations)

            text = 'validation_data: {}\nprediction: {}\nvalue: {:.3f}'.format('True' if image_id in valid_dataset_ids else 'False', pred_class, pred_value)
            
            n_classes = 1 if is_regression else len(classes)

            fig, ax = plt.subplots(1, n_classes+1, figsize=((n_classes+1)*2.5, 2.5))

            ax[0].imshow(image.permute(1, 2, 0))
            ax[0].axis('off')

            for i in range(1, n_classes+1):
                
                # merge original image with cam
                
                ax[i].imshow(image.permute(1, 2, 0))

                ax[i].imshow(cam_map[i-1].detach(), alpha=0.75, extent=(0, image.shape[2], image.shape[1], 0),
                        interpolation='bicubic', cmap='magma')

                ax[i].axis('off')

                # if i - 1 == idx:
                #     # draw prediction on image
                #     ax[i].text(10, 80, text, color='white', fontsize=6)
                # else:
                #     t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], output[i - 1].item())
                #     ax[i].text(10, 80, t, color='white', fontsize=6)

                t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], pred_output[i - 1].item())
                ax[i].text(10, 60, t, color='white', fontsize=6)

                # save image
                # image_path = os.path.join(image_folder, "{}_cam.png".format(image_id))
                # plt.savefig(image_path)

                # show image
            plt.show()
            plt.close()


def prepare_data(data_root, dataset, transform=None):
    data_path = os.path.join(data_root, dataset)
    if transform is not None:
        transform = preprocessing.transform(**transform)
    predict_data = preprocessing.PredictImageFolder(root=data_path, transform=transform)

    return predict_data

def segmentation_data(data_root, dataset, transform=None, individual_transform_generator=None):
    data_path = os.path.join(data_root, dataset)
    if transform is not None:
        transform = preprocessing.transform(**transform)
    predict_data = preprocessing.PredictIndividualTransformImageFolder(root=data_path, transform=transform, individual_transform_generator=individual_transform_generator)

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
    model.load_state_dict(model_state["model_state_dict"])

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

def save_config(config, saving_dir, saving_name):
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    saving_path = os.path.join(saving_dir, saving_name)
    with open(saving_path, 'w') as file:
        json.dump(config, file)

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
    """predict images in folder

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
    """
    arg_parser = argparse.ArgumentParser(description="Model Prediction")
    arg_parser.add_argument(
        "config", type=helper.dict_type, help="Required: configuration for prediction"
    )
    arg_parser.add_argument(
        "--type",
        type=str,
        default="csv",
        help="Optinal: saving type of predictions: csv (default) or json",
    )

    args = arg_parser.parse_args()

    # csv or json
    if args.type == "csv":
        run_dataset_predict_csv(args.config)
    # elif args.saving_type == 'json':
    #     run_dataset_predict_json(args.config)
    else:
        print("no valid saving format")


if __name__ == "__main__":
    main()

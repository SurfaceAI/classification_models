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

def run_segmentation(config):
    # # load device
    # device = torch.device(
    #     f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    # )

    # prepare data
    data_root = config.get("root_data")
    # if config.get("mode") == "testing":
    #     data_root = config.get("data_testing_path")
    segment_data = prepare_data(data_root, config.get("dataset"))

    # for debugging only
    count = 0

    for image, image_id in segment_data:
        # for debugging only
        count += 1
        # if count > 5:
        #     break
        print(image_id)
        # if image_id in [1246612526078070]:
        #     continue

        # detection
        detections = mapillary_requests.extract_detections_from_image(
            mapillary_requests.request_image_data_from_image_entity(
                image_id,
                mapillary_requests.load_mapillary_token(
                    token_path=config.get("mapillary_token_path")
                ),
                url=False,
                detections=True,
            )
        )

        # for segmentation analysis only
        image_det = image.copy().transpose(Image.FLIP_TOP_BOTTOM)
        draw_det = ImageDraw.Draw(image_det)
        # text_list = []

        for det in detections:
            if det["value"] in config.get("segment_color").keys():
                # for debugging only
                print(det["value"])
                # if det["value"] == 'construction--flat--crosswalk-plain':
                #     print(det["value"])

                # segments rescaled to [0, 1]
                rescaled_segments = [
                    np.divide(segment, 4096)
                    for segment in mapillary_detections.decode_detection_geometry(
                        det["geometry"]
                    )
                ]

                # detection_polygons = [
                #     mapillary_detections.convert_to_polygon(segment)
                #     for segment in rescaled_segments
                # ]
                # merged_polygon = mapillary_detections.merge_polygons(detection_polygons)

                # reject detection if coverage area of detection is low (sum of segments (alternative: max segment))
                # threshold = 0.05
                # threshold = 0
                # if mapillary_detections.calculate_polygon_area(merged_polygon) < threshold:
                #     continue

                # to avoid fringed edges/smooth edges
                # convex_hull = mapillary_detections.generate_polygon_convex_hull(
                #     merged_polygon
                # )

                # # Create Mask Image (TODO: include in transformation?)
                # mask = Image.new("L", image.size, 0)
                # draw = ImageDraw.Draw(mask)
                # rescaled_convex = (
                #     np.multiply(convex_hull.exterior.coords, image.size)
                #     .flatten()
                #     .tolist()
                # )
                # draw.polygon(rescaled_convex, fill=255)
                # mask = mask.copy().transpose(Image.FLIP_TOP_BOTTOM)
                # image_rgba = image.copy()
                # image_rgba.putalpha(mask)
                # color_layer = Image.new("RGB", image.size, (0, 0, 0)).convert("RGBA")
                # image_masked = Image.alpha_composite(color_layer, image_rgba).convert(
                #     "RGB"
                # )

                # bounding box for cropping
                # bbox = mapillary_detections.generate_polygon_bbox(merged_polygon)
                # top = 1 - bbox[3]
                # left = bbox[0]
                # height = bbox[3] - bbox[1]
                # width = bbox[2] - bbox[0]

                # transform = {
                #     **config.get("transform"),
                #     "crop": (top, left, height, width),
                # }
                # transform = preprocessing.transform(**transform)
                # image_transformed = transform(image_masked)

                # for debugging only
                # helper.imshow(image_transformed)

                # text = f'{det["value"]}'

                # for segmentation analysis only
                # draw polygons
                # for segment in rescaled_segments:
                #     draw_det.polygon(np.multiply(segment, image_det.size).flatten().tolist(), fill=config.get('segment_color')[det['value']], outline="blue")
                for seg in rescaled_segments:
                    draw_det.polygon(
                        np.multiply(seg, image_det.size)
                        .flatten()
                        .tolist(),
                        fill=config.get("segment_color")[det["value"]],
                        outline="blue",
                    )
                # draw bbox
                # left = bbox[0] * image_det.size[0]
                # right = bbox[2] * image_det.size[0]
                # top = bbox[1] * image_det.size[1]
                # upper = bbox[3] * image_det.size[1]
                # draw_det.rectangle(
                #     [left, top, right, upper],
                #     outline=config.get("segment_color")[det["value"]],
                # )
                # label prediction
                # text_list.append([text, (left + 5, image_det.size[1] - upper + 5)])
                # draw_det.text((left+5, top+5), text=text)

        # for segmentation analysis only
        image_det = image_det.transpose(Image.FLIP_TOP_BOTTOM)
        composite = Image.blend(image, image_det, 0.2)
        composite_draw = ImageDraw.Draw(composite)
        # for label in text_list:
        #     composite_draw.text(label[1], label[0])
        composite.show()
        
        # # save image with detections
        # image_folder = os.path.join(data_root, config.get("dataset"), "segmentation")
        # if not os.path.exists(image_folder):
        #     os.makedirs(image_folder)
        # image_path = os.path.join(image_folder, "{}_segment.png".format(image_id))
        # # with open(image_path, 'wb') as handler:
        # #     handler.write(image)
        # composite.save(image_path)
        composite.close()
        

    # # save predictions
    # start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    # saving_name = (
    #     config.get("name")
    #     + "-"
    #     + config.get("dataset").replace("/", "_")
    #     + "-"
    #     + start_time
    #     + ".csv"
    # )

    # saving_path = save_predictions_csv(
    #     df=df, saving_dir=config.get("root_predict"), saving_name=saving_name
    # )

    print(f'Images {config.get("dataset")} segmented and saved.')

def run_image_per_image_predict_segmentation(config):
    # load device
    device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

    # prepare data
    predict_data = prepare_data(config.get("root_data"), config.get("dataset"))

    level = 0
    columns = ["Image", "Segment", "Prediction", f"Level_{level}"]
    df = pd.DataFrame(columns=columns)

    # for debugging only
    count = 0

    for image, image_id in predict_data:
        # for debugging only
        count += 1
        if count > 20:
            break
        print(image_id)

        # detection
        detections = mapillary_requests.extract_detections_from_image(
            mapillary_requests.request_image_data_from_image_entity(
                image_id,
                mapillary_requests.load_mapillary_token(
                    token_path=config.get("mapillary_token_path")
                ),
                url=False,
                detections=True,
            )
        )

        # for segmentation analysis only
        image_det = image.copy().transpose(Image.FLIP_TOP_BOTTOM)
        draw_det = ImageDraw.Draw(image_det)
        text_list = []

        # TODO: save image in csv plain (to be registered if no valid segmentation)
        
        for det in detections:
            if det["value"] in config.get("segment_color").keys():
                # for debugging only
                print(det["value"])

                # segments rescaled to [0, 1]
                rescaled_segments = [
                    np.divide(segment, 4096)
                    for segment in mapillary_detections.decode_detection_geometry(
                        det["geometry"]
                    )
                ]

                detection_polygons = [
                    mapillary_detections.convert_to_polygon(segment)
                    for segment in rescaled_segments
                ]
                merged_polygon = mapillary_detections.merge_polygons(detection_polygons)

                # reject detection if coverage area of detection is low (sum of segments (alternative: max segment))
                if mapillary_detections.calculate_polygon_area(merged_polygon) < 0.05:
                    continue

                # to avoid fringed edges/smooth edges
                convex_hull = mapillary_detections.generate_polygon_convex_hull(
                    merged_polygon
                )

                # Create Mask Image (TODO: include in transformation?)
                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                rescaled_convex = (
                    np.multiply(convex_hull.exterior.coords, image.size)
                    .flatten()
                    .tolist()
                )
                draw.polygon(rescaled_convex, fill=255)
                mask = mask.copy().transpose(Image.FLIP_TOP_BOTTOM)
                image_rgba = image.copy()
                image_rgba.putalpha(mask)
                color_layer = Image.new("RGB", image.size, (0, 0, 0)).convert("RGBA")
                image_masked = Image.alpha_composite(color_layer, image_rgba).convert(
                    "RGB"
                )

                # bounding box for cropping
                bbox = mapillary_detections.generate_polygon_bbox(convex_hull)
                top = 1 - bbox[3]
                left = bbox[0]
                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]

                transform = {
                    **config.get("transform"),
                    "crop": (top, left, height, width),
                }
                transform = preprocessing.transform(**transform)
                image_transformed = transform(image_masked)

                # for debugging only
                # helper.imshow(image_transformed)

                text = recursive_image_per_image_predict_csv(
                    model_dict=config.get("model_dict"),
                    model_root=config.get("root_model"),
                    image_id=image_id,
                    segment=det["value"],
                    image_transformed=image_transformed,
                    device=device,
                    df=df,
                    level=level,
                )

                # for segmentation analysis only
                # draw polygons
                # for segment in rescaled_segments:
                #     draw_det.polygon(np.multiply(segment, image_det.size).flatten().tolist(), fill=config.get('segment_color')[det['value']], outline="blue")
                draw_det.polygon(
                    np.multiply(merged_polygon.exterior.coords, image_det.size)
                    .flatten()
                    .tolist(),
                    fill=config.get("segment_color")[det["value"]],
                    outline="blue",
                )
                # draw bbox
                left = bbox[0] * image_det.size[0]
                right = bbox[2] * image_det.size[0]
                top = bbox[1] * image_det.size[1]
                upper = bbox[3] * image_det.size[1]
                draw_det.rectangle(
                    [left, top, right, upper],
                    outline=config.get("segment_color")[det["value"]],
                )
                # label prediction
                text_list.append([text, (left + 5, image_det.size[1] - upper + 5)])
                # draw_det.text((left+5, top+5), text=text)

        # for segmentation analysis only
        image_det = image_det.transpose(Image.FLIP_TOP_BOTTOM)
        composite = Image.blend(image, image_det, 0.2)
        composite_draw = ImageDraw.Draw(composite)
        for label in text_list:
            composite_draw.text(label[1], label[0])
        # composite.show()
        # save image with detections
        image_folder = os.path.join(config.get("root_predict"), config.get("dataset"))
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_path = os.path.join(image_folder, "{}_segment.png".format(image_id))
        # with open(image_path, 'wb') as handler:
        #     handler.write(image)
        composite.save(image_path)

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
        if is_regression:
            cls = (
                "outside"
                if str(pred_output.round().int().item()) not in classes.keys()
                else classes[str(pred_output.round().int().item())]
            )
            i = df.shape[0]
            df.loc[i, columns] = [
                image_id,
                segment,
                pred_output.item(),
                is_valid_data,
                cls,
            ] + pre_cls_entry
            text = cls + ": " + f"{pred_output.item():.3f}"
        else:
            pred_class = classes[torch.argmax(pred_output, dim=0).item()]
            for cls, prob in zip(classes, pred_output.tolist()):
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
                pre_cls=cls,
            )
            text = (
                pred_class
                + ": "
                + f"{torch.max(pred_output, dim=0).values.item():.3f}"
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
        if is_regression:
            pred_classes = [
                "outside"
                if str(pred.item()) not in classes.keys()
                else classes[str(pred.item())]
                for pred in pred_outputs.round().int()
            ]
            for image_id, pred, is_vd, cls in zip(
                image_ids, pred_outputs, is_valid_data, pred_classes
            ):
                i = df.shape[0]
                df.loc[i, columns] = [image_id, pred.item(), is_vd, cls] + pre_cls_entry
        else:
            pred_classes = [
                classes[idx.item()] for idx in torch.argmax(pred_outputs, dim=1)
            ]
            for image_id, pred, is_vd in zip(image_ids, pred_outputs, is_valid_data):
                for cls, prob in zip(classes, pred.tolist()):
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


def predict_image_per_image(model, image_transformed, is_regression, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        input = torch.unsqueeze(image_transformed, 0).to(device)
        output = model(input)
        if is_regression:
            output = output.flatten()
        else:
            output = model.get_class_probabilies(output)

    return torch.squeeze(output, 0)


def predict(model, data, batch_size, is_regression, device):
    model.to(device)
    model.eval()

    loader = DataLoader(data, batch_size=batch_size)

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

                t = '\n\nprediction: {}\nvalue: {:.3f}'.format(classes[i - 1], output[i - 1].item())
                ax[i].text(10, 60, t, color='white', fontsize=6)

                # save image
                # image_path = os.path.join(image_folder, "{}_cam.png".format(image_id))
                # plt.savefig(image_path)

                # show image
            plt.show()
            plt.close()


def prepare_data(data_root, dataset, transform):

def prepare_data(data_root, dataset, transform=None):
    data_path = os.path.join(data_root, dataset)
    if transform is not None:
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

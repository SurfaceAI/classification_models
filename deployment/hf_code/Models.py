import os
# import sys
from functools import partial
from pathlib import Path
import logging

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torchvision import models, transforms
import pandas as pd
from collections import defaultdict


class ModelInterface:
    """
    Interface for managing image classification and regression tasks.

    """
    def __init__(self, config):
        """
        Initialize the ModelInterface.

        Parameters:
            config (dict): Configuration dictionary containing the following keys:
                - gpu_kernel (int): GPU index to use for computations. Defaults to the first available GPU if available, otherwise CPU.
                - transform_surface (dict): Parameters for surface type and quality image transformations, including resize, crop, and normalization settings.
                - transform_road_type (dict): Parameters for road type image transformations, similar to surface transformations.
                - model_root (str): Directory path where model files are stored locally. Defaults to folder name 'models'.
                - models (dict): Dictionary mapping prediction levels (e.g., 'road_type', 'surface_type') to model file names.
                - hf_model_repo (str): Hugging Face repository ID for downloading models if not found locally.
        """
        self.device = self._validate_device(config.get('gpu_kernel', ''))
        self.model_root = Path(config.get("model_root", "models"))
        self.models = config.get("models")
        self.hf_model_repo = config.get("hf_model_repo", "")
        self._validate_models()
        self._default_normalization = (NORM_MEAN, NORM_SD)
        self.transform_surface = self._validate_transform(config.get("transform_surface", None), "surface_type")
        self.transform_road_type = self._validate_transform(config.get("transform_road_type", None), "road_type")

    def _validate_device(self, gpu_kernel):
        try:
            cuda = "cuda" if gpu_kernel == '' else f"cuda:{gpu_kernel}"
            return torch.device(
                cuda if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            logging.warning(f"An unexpected error occurred while selecting GPU: {e}\n"
                          + "Falling back to CPU.")
            return torch.device("cpu")

    def _validate_models(self):
        """
        Check if model files exist and download from hugging face if not.
        """
        if self.models is None:
            raise TypeError("No models are defined.")

        log_model_not_defined = "No model for '{level_string}' is defined. Prediction is skipped."

        # check surface type model
        level = "surface_type"
        model_file = self.models.get(level)
        if model_file is None:
            logging.warning(log_model_not_defined.format(level_string=model_to_info_string[level]))
        else:
            self.download_model(model_file)
            _, surface_class_to_idx, _ = self.load_model(model=model_file)

        # check quality models
        level = "surface_quality"
        sub_models = self.models.get(level)
        if model_file is None:
            logging.warning(log_model_not_defined.format(level_string=model_to_info_string[level]))
        else:
            for surface_type in surface_class_to_idx:
                model_file = sub_models.get(surface_type)
                if model_file is None:
                    logging.warning(log_model_not_defined.format(level_string=surface_type))
                else:
                    self.download_model(model_file)
                    self.load_model(model=model_file)

        # check road type model
        level = "road_type"
        model_file = self.models.get(level)
        if model_file is None:
            logging.warning(log_model_not_defined.format(level_string=model_to_info_string[level]))
        else:
            self.download_model(model_file)
            self.load_model(model=model_file)

    def _validate_transform(self, transform, level):
        """
        Validate the transformation for a given model type if the model exists.

        Parameters:
            - transform (dict): transformation.
            - level (str): model level.

        Returns:
            dict: transformation.
        """
        if (level in self.models) and (transform is None):
            logging.warning(f"No transformation for {model_to_info_string[level]} prediction defined.")
            transform = {}
        
        if "normalize" not in transform:
            logging.info(f"No normalization parameters for {model_to_info_string[level]} prediction provided. Using default values.")
            transform["normalize"] = self._default_normalization
        
        return transform
  

    def download_model(self, model):
        """
        Download a model from Hugging Face repository.

        Parameters:
            - model (str): Model file name.

        Returns:
            None
        """
        model_path = self.model_root / model
        # load model data from hugging face if not locally available
        if not os.path.exists(model_path):
            logging.info(
                f"Model file not found at {model_path}. Downloading from Hugging Face..."
            )
            try:
                os.makedirs(self.model_root, exist_ok=True)
                model_path = hf_hub_download(
                    repo_id=self.hf_model_repo, filename=model, local_dir=self.model_root
                )
                logging.info(f"Model file downloaded successfully to {model_path}.")
            except Exception as e:
                logging.error(f"An unexpected error occurred while downloading the model {model}: {e}")
                raise e


    @staticmethod
    def custom_crop(img, crop_style=None):
        """
        Crop an image according to the specified style.

        Parameters:
            - img (PIL.Image): Input image to be cropped.
            - crop_style (str, optional): Style of cropping (e.g., 'lower_middle_half').

        Returns:
            PIL.Image: Cropped image.
        """
        im_width, im_height = img.size
        if crop_style == CROP_LOWER_MIDDLE_HALF:
            top = im_height / 2
            left = im_width / 4
            height = im_height / 2
            width = im_width / 2
        elif crop_style == CROP_LOWER_HALF:
            top = im_height / 2
            left = 0
            height = im_height / 2
            width = im_width
        else:  # None, or not valid
            logging.warning(f"Cropping method {crop_style} is not defined. Image is not cropped.")
            return img

        cropped_img = transforms.functional.crop(img, top, left, height, width)
        return cropped_img

    def transform(
        self,
        resize=None,
        crop=None,
        to_tensor=True,
        normalize=None,
    ):
        """
        Create a PyTorch image transformation function based on specified parameters.

        Parameters:
            - resize ((int, int) or int, optional): Target size for resizing, e.g. (height, width). If int, then used for both height and width.
            - crop (str, optional): crop style e.g. 'lower_middle_third'
            - to_tensor (bool, optional): Converts the PIL Image (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            - normalize (tuple of lists [r, g, b], optional): Mean and standard deviation for normalization.

        Returns:
            PyTorch image transformation function.
        """
        transform_list = []

        if crop is not None:
            transform_list.append(
                transforms.Lambda(partial(self.custom_crop, crop_style=crop))
            )

        if resize is not None:
            if isinstance(resize, int):
                resize = (resize, resize)
            transform_list.append(transforms.Resize(resize))

        if to_tensor:
            transform_list.append(transforms.ToTensor())

        if normalize is not None:
            transform_list.append(transforms.Normalize(*normalize))

        composed_transform = transforms.Compose(transform_list)
        return composed_transform

    def preprocessing(self, img_data_raw, transform):
        """
        Preprocess raw image data using a specified transformation.

        Parameters:
            - img_data_raw (list): List of raw images to preprocess.
            - transform (dict): Dictionary of transformation parameters.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if not img_data_raw:
            raise ValueError("Image data is empty.")
        
        transform = self.transform(**transform)
        img_data = torch.stack([transform(img) for img in img_data_raw])
        return img_data

    def load_model(self, model):
        """
        Load a model from local storage.

        Parameters:
            - model (str): Model file name.

        Returns:
            nn.Module: Loaded model.
            dict: Mapping of classes to indices.
            bool: Whether the model is for regression.
        """
        model_path = self.model_root / model
        try:
            model_state = torch.load(model_path, map_location=self.device)
            model_name = model_state["model_name"]
            is_regression = model_state["is_regression"]
            class_to_idx = model_state["class_to_idx"]
            num_classes = 1 if is_regression else len(class_to_idx.items())
            model_state_dict = model_state["model_state_dict"]
            model_cls = model_mapping[model_name]
            model = model_cls(num_classes=num_classes)
            model.load_state_dict(model_state_dict)
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the model {model_path}: {e}")
            raise e

        return model, class_to_idx, is_regression

    def predict(self, model, data):
        """
        Perform predictions using the specified model and input data.

        Parameters:
            - model (nn.Module): The model to use for predictions.
            - data (torch.Tensor): Batch of input data.

        Returns:
            torch.Tensor: Predicted values or class probabilities.
        """
        model.to(self.device)
        model.eval()

        image_batch = data.to(self.device)

        with torch.no_grad():
            batch_outputs = model(image_batch)
            # batch_classes, batch_values = model.get_class_and_value(batch_outputs)
            batch_values = model.get_class_probabilities(batch_outputs)

        return batch_values
    
    @staticmethod
    def predict_to_classes(batch_values, class_to_idx):
        """
        Map predicted values to classes.

        Parameters:
            - batch_values (torch.Tensor): Batch of prediction values.
            - class_to_idx (dict): Mapping from class names to indices.

        Returns:
            list: List of predicted values.
            list: List of predicted classes.
        """
        idx_to_class = {i: cls for cls, i in class_to_idx.items()}
        
        if len(list(batch_values.shape)) < 2:
            classes = [
                idx_to_class[
                    min(
                        max(idx.item(), min(list(class_to_idx.values()))),
                        max(list(class_to_idx.values())),
                    )
                ]
                for idx in batch_values.round().int()
            ]
            values = batch_values.tolist()
        else:
            classes = [idx_to_class[idx.item()] for idx in torch.argmax(batch_values, dim=1)]
            values = batch_values.tolist()
            
        return values, classes

    def batch_classifications(self, img_data_raw, img_ids=None):
        """
        Perform batch classification for multiple prediction levels (road type, surface type, surface quality).

        Parameters:
            - img_data_raw (list): List of raw images to classify.
            - img_ids (list, optional): List of IDs corresponding to the images. Defaults to indices.

        Returns:
            list: Combined list of image ids and predictions across levels.
        """
        if not img_data_raw:
            logging.info("Input data is empty. No predictions performed.")
            return []
        
        # default image ids
        if img_ids is None:
            img_ids = range(len(img_data_raw))

        # road type
        level = "road_type"
        model_file = self.models.get(level)
        if model_file is not None:
            model, class_to_idx, _ = self.load_model(model=model_file)
            data = self.preprocessing(img_data_raw, self.transform_road_type)
            values = self.predict(model, data)
            road_values, road_classes = self.predict_to_classes(values, class_to_idx)

        # surface type
        level = "surface_type"
        model_file = self.models.get(level)
        if model_file is not None:
            model, class_to_idx, _ = self.load_model(model=model_file)
            data = self.preprocessing(img_data_raw, self.transform_surface)
            values = self.predict(model, data)
            surface_values, surface_classes = self.predict_to_classes(values, class_to_idx)

            # surface quality
            level = "surface_quality"
            sub_models = self.models.get(level)
            if sub_models is not None:
                surface_indices = defaultdict(list)
                for i, surface_type in enumerate(surface_classes):
                    surface_indices[surface_type].append(i)

                quality_values = [None] * len(img_data_raw)
                quality_classes = [None] * len(img_data_raw)
                for surface_type, indices in surface_indices.items():
                    model_file = sub_models.get(surface_type)
                    if model_file is not None:
                        model, class_to_idx, _ = self.load_model(model=model_file)
                        values = self.predict(model, data[indices])
                        values, classes = self.predict_to_classes(values, class_to_idx)
                        for idx, vl, cls in zip(indices, values, classes):
                            quality_values[idx] = vl
                            quality_classes[idx] = cls

        # final results combination
        final_results = [
            [
                img_ids[i],
                road_classes[i],
                road_values[i],
                surface_classes[i],
                surface_values[i],
                quality_classes[i],
                quality_values[i],
            ]
            for i in range(len(img_data_raw))
        ]

        return final_results


class CustomEfficientNetV2SLinear(nn.Module):
    """
    Custom implementation of EfficientNetV2-S with a linear classifier for classification or regression tasks.

    Attributes:
        features (nn.Sequential): Feature extractor from EfficientNetV2-S.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        classifier (nn.Sequential): Fully connected layers for classification.
        is_regression (bool): Whether the model is configured for regression tasks.
        criterion (callable): Loss function used for training the model.
    """

    def __init__(self, num_classes, avg_pool=1):
        super(CustomEfficientNetV2SLinear, self).__init__()

        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        # adapt output layer
        in_features = model.classifier[-1].in_features * (avg_pool * avg_pool)
        fc = nn.Linear(in_features, num_classes, bias=True)
        model.classifier[-1] = fc

        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(avg_pool)
        self.classifier = model.classifier
        if num_classes == 1:
            self.criterion = nn.MSELoss
            self.is_regression = True
        else:
            self.criterion = nn.CrossEntropyLoss
            self.is_regression = False

    def get_class_probabilities(self, x):
        if self.is_regression:
            x = x.flatten()
        else:
            x = nn.functional.softmax(x, dim=1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    # def get_optimizer_layers(self):
    #     return self.classifier


# Constants
EFFNET_LINEAR = "efficientNetV2SLinear"
CROP_LOWER_MIDDLE_HALF = "lower_middle_half"
CROP_LOWER_HALF = "lower_half"
NORM_MEAN = [0.42834484577178955, 0.4461250305175781, 0.4350937306880951]
NORM_SD = [0.22991590201854706, 0.23555299639701843, 0.26348039507865906]


model_mapping = {
    EFFNET_LINEAR: CustomEfficientNetV2SLinear,
}

model_to_info_string = {
    "surface_type": "surface type",
    "road_type": "road type",
    "surface_quality": "quality",
}
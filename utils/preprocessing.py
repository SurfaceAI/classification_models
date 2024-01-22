import sys
sys.path.append('.')

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import copy
import os
from utils import general_config, constants
from tqdm import tqdm
from pathlib import Path
    


class PartialImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 selected_classes=None
                 ):
        self.selected_classes = selected_classes
        super(PartialImageFolder, self).__init__(root,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 loader=loader,
                                                 is_valid_file=is_valid_file)
    
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if self.selected_classes is not None:
            classes = [c for c in classes if c in self.selected_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def create_train_validation_datasets(dataset, label_type, selected_classes, validation_size, general_transform, augmentation, random_state):

    # data path
    data_root = general_config.training_data_path
    data_path = os.path.join(data_root, dataset, label_type)

    # create complete dataset
    complete_dataset = PartialImageFolder(data_path, selected_classes=selected_classes)

    if general_transform.get('normalize') is not None:
        general_transform['normalize'] = load_normalization(general_transform.get('normalize'), dataset, label_type)
    
    train_transform = transform(**general_transform, **augmentation)
    valid_transform = transform(**general_transform)

    train_dataset, valid_dataset = train_validation_split_datasets(complete_dataset, validation_size, train_transform, valid_transform, random_state)

    return train_dataset, valid_dataset


def load_normalization(normalization_type, dataset, label_type):

    tuple_mean_sd = None
    if normalization_type == 'imagenet':
        tuple_mean_sd = (constants.IMAGNET_MEAN, constants.IMAGNET_SD)
    elif normalization_type == 'from_data':
        mean_name = f'{dataset}_{label_type}_mean'.upper()
        mean_value = getattr(constants, mean_name, None)
        sd_name = f'{dataset}_{label_type}_sd'.upper()
        sd_value = getattr(constants, sd_name, None)
        if mean_value is None or sd_value is None:
            mean_value, sd_value = calculate_dataset_normalization(dataset, label_type)
        tuple_mean_sd =(mean_value, sd_value)

    return tuple_mean_sd
    

# TODO: more efficient method?
def calculate_dataset_normalization(dataset, label_type):

    # calculate normalization parameters
    data_root = general_config.training_data_path
    data_path = os.path.join(data_root, dataset, label_type)

    image_size = constants.H256_W256

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    image_dataset = datasets.ImageFolder(data_path, transform = transform)
    # TODO: does larger batch make any difference?
    image_loader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    std_sum = torch.zeros(3)
    mean_sum = torch.zeros(3)

    for input, _ in tqdm(image_loader, desc=f'{dataset}_{label_type} normalization'):
        std_image, mean_image = torch.std_mean(input, dim=[0, 2, 3])
        std_sum += std_image
        mean_sum += mean_image

    total_std = (std_sum / len(image_loader.sampler)).tolist()
    total_mean = (mean_sum / len(image_loader.sampler)).tolist()
    
    # write values to constants file
    folder = Path(__file__).parent
    
    with open(os.path.join(folder, 'constants.py'), 'a') as f:
        f.write(f'\n{f"{dataset}_{label_type}_mean".upper()} = {total_mean}\n')
        f.write(f'{f"{dataset}_{label_type}_sd".upper()} = {total_std}\n')
    
    return total_mean, total_std


def train_validation_split_datasets(complete_dataset, validation_size, train_transform, valid_transform, random_state):

    (samples_train, samples_valid,
     targets_train, targets_valid,
     imgs_train, imgs_valid) = train_test_split(complete_dataset.samples, complete_dataset.targets, complete_dataset.imgs, test_size=validation_size, random_state=random_state, stratify=complete_dataset.targets)
    
    train_dataset = copy.deepcopy(complete_dataset)
    train_dataset.samples = samples_train
    train_dataset.targets = targets_train
    train_dataset.imgs = imgs_train
    train_dataset.transform = train_transform

    valid_dataset = copy.deepcopy(complete_dataset)
    valid_dataset.samples = samples_valid
    valid_dataset.targets = targets_valid
    valid_dataset.imgs = imgs_valid
    valid_dataset.transform = valid_transform

    return train_dataset, valid_dataset

def custom_crop(img, crop_style=None):

    im_width, im_height = img.size
    if crop_style == 'lower_middle_third':
        top = im_height / 3 * 2
        left = im_width / 3
        height = im_height - top
        width = im_width / 3
    else: # None, or not valid
        return img
    
    cropped_img = transforms.functional.crop(img, top, left, height, width)
    return cropped_img
    
def transform(resize=None,
              crop=None,
              to_tensor=True,
              normalize=None,
              random_rotation=None,
              random_horizontal_flip=None,
              random_vertical_flip=None,
              color_jitter=None,
              gaussian_blur=None,
              ):
    """
    Create a PyTorch image transformation function based on specified parameters.

    Parameters:
        - resize (tuple or None): Target size for resizing, e.g. (height, width).
        - crop (string): crop style e.g. 'lower_middle_third'
        - to_tensor (bool): Converts the PIL Image (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        - normalize (tuple of lists [r, g, b] or None): Mean and standard deviation for normalization.
        - random_rotation (float (non-negative) or None): Maximum rotation angle in degrees for random rotation.
        - random_horizontal_flip (bool): Flip right-left with 0.5 probability.
        - random_vertical_flip (bool):  Flip top-bottom with 0.5 probability.
        - color_jitter (tuple of 4 or None): Randomly change the brightness, contrast, saturation and hue.
            - brightness between 0 and 1 or None
            - contrast between 0 and 1 or None
            - saturation between 0 and 1 or None
            - hue between 0 and 0.5 or None
        - gaussian_blur (int or None): Blurs image with randomly chosen Gaussian blur.

    Returns:
        PyTorch image transformation function.
    """
    transform_list = []

    if random_rotation is not None:
        transform_list.append(transforms.RandomRotation(random_rotation))

    if crop is not None:
        transform_list.append(transforms.Lambda(lambda img: custom_crop(img, crop)))
        
    if resize is not None:
        transform_list.append(transforms.Resize(resize))

    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if random_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())

    if color_jitter is not None:
        transform_list.append(transforms.ColorJitter(*color_jitter))

    if gaussian_blur is not None:
        transform_list.append(transforms.GaussianBlur(gaussian_blur))

    if to_tensor:
        transform_list.append(transforms.ToTensor())

    if normalize is not None:
        transform_list.append(transforms.Normalize(*normalize))

    composed_transform = transforms.Compose(transform_list)

    return composed_transform

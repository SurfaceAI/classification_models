import sys

sys.path.append(".")

import copy
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from tqdm import tqdm
from itertools import chain
import pandas as pd

from experiments.config import global_config
from src import constants as const


class PartialImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root,
        is_regression,
        selected_classes=None,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
    ):
        self.selected_classes = selected_classes
        self.is_regression = is_regression
        super(PartialImageFolder, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )

        if self.selected_classes is not None:
            classes = [c for c in classes if c in self.selected_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        if self.is_regression:
            class_to_idx = {
                cls_name: const.SMOOTHNESS_INT[cls_name] for cls_name in classes
            }
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

def find_flatten_classes(directory, selected_classes):
    type_classes = sorted(
        entry.name for entry in os.scandir(directory) if entry.is_dir()
    )

    # only take the ones that are in selected_classes
    if selected_classes is not None:
        type_classes = [
            c for c in type_classes if c in selected_classes.keys()
        ]
    if not type_classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    flattened_classes = []
    # for each type class, we find all the sub-classes and store them separately
    for type_class in type_classes:
        type_directory = os.path.join(directory, type_class)
        quality_classes = sorted(
            entry.name for entry in os.scandir(type_directory) if entry.is_dir()
        )
        quality_classes = [
            (type_class + "__" + c)
            for c in quality_classes
            if c in selected_classes[type_class]
        ]
        flattened_classes.extend(quality_classes)

    flattened_class_to_idx = {
        cls_name: i for i, cls_name in enumerate(flattened_classes)
    }

    return flattened_classes, flattened_class_to_idx

def make_flatten_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_flatten_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]

        t_q_split = target_class.split("__", 1)
        type_class, quality_class = t_q_split[0], t_q_split[1]

        target_dir = os.path.join(directory, type_class, quality_class)

        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                # if is_valid_file(path):
                item = path, class_index
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

class FlattenFolders(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        selected_classes=None,
    ):
        self.selected_classes = selected_classes

        super(FlattenFolders, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory):
        # find type classes
        return find_flatten_classes(directory, self.selected_classes)

    @staticmethod
    def make_dataset(
        directory,
        class_to_idx=None,
        extensions=None,
        is_valid_file=None,
        allow_empty=False,
    ):
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_flatten_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

# VisionDataset instead of Dataset only used due to __repr__
class PredictImageFolder(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        extensions = datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None
        samples = self.make_dataset(self.root, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions
        self.samples = samples

    @staticmethod
    def make_dataset(
        directory: str,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[str]:
        """Generates a list of samples of a form "path_to_sample" """
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return datasets.folder.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    try:
                        Image.open(path)
                    except:
                        print(f'Corrupted image: {path}')
                        continue
                    else:
                        instances.append(path)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, id) where id is file name w/o extension.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        id = os.path.splitext(os.path.split(path)[-1])[0]

        return sample, id

    def __len__(self) -> int:
        return len(self.samples)


def create_train_validation_datasets(
    data_root,
    dataset,
    metadata,
    train_valid_split_list,
    selected_classes,
    validation_size,
    general_transform,
    augmentation,
    random_state,
    is_regression,
    level=None,
    type_class=None,
):
    # TODO: only a single argument instead of level + type_class?

    # data path
    data_path = os.path.join(data_root, dataset)

    # train validation split file
    if metadata is not None and train_test_split is not None:
        train_valid_split_list = os.path.join(data_root, metadata, train_valid_split_list)
    else:
        train_valid_split_list = None

    # flatten if level is flatten
    if level == const.FLATTEN:
        complete_dataset = FlattenFolders(data_path, selected_classes=selected_classes)
    # surface or smoothness for surface type if level is not flatten
    else:
        if type_class is not None:
            data_path = os.path.join(data_path, type_class)

        # create complete dataset
        complete_dataset = PartialImageFolder(
            data_path,
            is_regression=is_regression,
            selected_classes=selected_classes,
        )

    if general_transform.get("normalize") is not None:
        general_transform["normalize"] = load_normalization(
            general_transform.get("normalize"), data_root, dataset
        )

    train_transform = transform(**general_transform, **augmentation)
    valid_transform = transform(**general_transform)

    train_dataset, valid_dataset = train_validation_split_datasets(
        complete_dataset,
        train_valid_split_list,
        validation_size,
        train_transform,
        valid_transform,
        random_state,
    )

    return train_dataset, valid_dataset


def load_normalization(normalization, data_root, dataset):
    if isinstance(normalization, (tuple, list)):
        tuple_mean_sd = tuple(normalization)
    elif normalization == "imagenet":
        tuple_mean_sd = (const.IMAGNET_MEAN, const.IMAGNET_SD)
    elif normalization == "from_data":
        dataset_name = dataset.replace("/", "_")
        mean_name = f"{dataset_name.upper()}_MEAN"
        mean_value = getattr(const, mean_name, None)
        sd_name = f"{dataset_name.upper()}_SD"
        sd_value = getattr(const, sd_name, None)
        if mean_value is None or sd_value is None:
            mean_value, sd_value = calculate_dataset_normalization(
                data_root, dataset
            )  # TODO: check if already computed?
        tuple_mean_sd = (mean_value, sd_value)
    else:
        tuple_mean_sd = None

    return tuple_mean_sd


# TODO: more efficient method?
def calculate_dataset_normalization(data_root, dataset):
    # calculate normalization parameters
    data_path = os.path.join(data_root, dataset)

    image_size = const.H256_W256

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    image_dataset = datasets.ImageFolder(data_path, transform=transform)
    # TODO: does larger batch make any difference?
    image_loader = DataLoader(image_dataset, batch_size=1, shuffle=False)

    std_sum = torch.zeros(3)
    mean_sum = torch.zeros(3)

    for input, _ in tqdm(image_loader, desc=f"{dataset} normalization"):
        std_image, mean_image = torch.std_mean(input, dim=[0, 2, 3])
        std_sum += std_image
        mean_sum += mean_image

    total_std = (std_sum / len(image_loader.sampler)).tolist()
    total_mean = (mean_sum / len(image_loader.sampler)).tolist()

    # write values to constants file
    folder = Path(__file__).parent.parent

    with open(os.path.join(folder, "constants.py"), "a") as f:
        dataset_name = dataset.replace("/", "_")
        f.write(f'\n{f"{dataset_name.upper()}_MEAN"} = {total_mean}\n')
        f.write(f'{f"{dataset_name.upper()}_SD"} = {total_std}\n')

    return total_mean, total_std


def train_validation_split_datasets(
    complete_dataset, train_valid_split_list, validation_size, train_transform, valid_transform, random_state
):
    if train_valid_split_list is not None:
        print(f"Train validation split with {train_valid_split_list}.")
        df = pd.read_csv(train_valid_split_list,
                         dtype={
                             'is_in_validation': bool,
                             'image_id': str,
                             })
        image_id_to_index = {os.path.splitext(os.path.basename(path))[0]: idx for idx, (path, _) in enumerate(complete_dataset.samples)}
    
        train_indices = []
        valid_indices = []
        
        for image_id in image_id_to_index.keys():
            index = image_id_to_index[image_id]
            if image_id in df['image_id'].values and df[df['image_id'] == image_id]['is_in_validation'].values[0]:
                valid_indices.append(index)
            else:
                train_indices.append(index)
        (
            samples_train,
            samples_valid,
            targets_train,
            targets_valid,
            imgs_train,
            imgs_valid,
        ) = train_valid_split(
            complete_dataset.samples,
            complete_dataset.targets,
            complete_dataset.imgs,
            train_valid_split_tuple=(train_indices, valid_indices),
        )
    else:
        (
            samples_train,
            samples_valid,
            targets_train,
            targets_valid,
            imgs_train,
            imgs_valid,
        ) = train_test_split(
            complete_dataset.samples,
            complete_dataset.targets,
            complete_dataset.imgs,
            test_size=validation_size,
            random_state=random_state,
            stratify=complete_dataset.targets,
        )

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

def train_valid_split(*arrays, train_valid_split_tuple):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    
    train_indices, valid_indices = train_valid_split_tuple
    n_samples = len(train_indices) + len(valid_indices)

    for a in arrays:
        if len(a) != n_samples:
            raise ValueError("All input arrays must have the same length as sum of train and validation indices")

    return list(
        chain.from_iterable(
            ([a[idx] for idx in train_indices], [a[idx] for idx in valid_indices]) for a in arrays
        )
    )


def custom_crop(img, crop_style=None):
    im_width, im_height = img.size
    if crop_style == "lower_middle_third":
        top = im_height / 3 * 2
        left = im_width / 3
        height = im_height - top
        width = im_width / 3
    elif crop_style == "lower_middle_half":
        top = im_height / 2
        left = im_width / 4
        height = im_height / 2
        width = im_width / 2
    elif crop_style == "lower_half":
        top = im_height / 2
        left = 0
        height = im_height / 2
        width = im_width
    else:  # None, or not valid
        return img

    cropped_img = transforms.functional.crop(img, top, left, height, width)
    return cropped_img


def transform(
    preresize=None,
    resize=None,
    crop=None,
    to_tensor=True,
    normalize=None,
    random_rotation=None,
    random_horizontal_flip=None,
    random_vertical_flip=None,
    color_jitter=None,
    gaussian_blur_kernel=None,
    gaussian_blur_sigma=None,
    gaussian_blur_fixed=False,
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
        - gaussian_blur_kernel (odd int or None): Kernel size for image Gaussian blur.
        - gaussian_blur_sigma (float or None): Sigma or max sigma for randomly chosen Gaussian blur.
        - gaussian_blur_fixed (boolean): True for fixed sigma, False for range

    Returns:
        PyTorch image transformation function.
    """
    transform_list = []

    if random_rotation is not None:
        transform_list.append(transforms.RandomRotation(random_rotation))

    # if crop is not None:
    #     transform_list.append(transforms.Lambda(lambda img: custom_crop(img, crop)))

    if crop is not None:
        transform_list.append(transforms.Lambda(partial(custom_crop, crop_style=crop)))

    if preresize is not None:
        transform_list.append(transforms.Resize(preresize))

    if resize is not None:
        transform_list.append(transforms.Resize(resize))

    if random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if random_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())

    if color_jitter is not None:
        transform_list.append(transforms.ColorJitter(*color_jitter))

    if gaussian_blur_kernel is not None:
        if gaussian_blur_sigma is not None:
            if gaussian_blur_fixed == False:
                gaussian_blur_sigma = (0.01, gaussian_blur_sigma)
        else:
            gaussian_blur_sigma = (0.1, 2.0) # pytorch default value
        transform_list.append(transforms.GaussianBlur(gaussian_blur_kernel, gaussian_blur_sigma))

    if to_tensor:
        transform_list.append(transforms.ToTensor())

    if normalize is not None:
        transform_list.append(transforms.Normalize(*normalize))

    composed_transform = transforms.Compose(transform_list)

    return composed_transform

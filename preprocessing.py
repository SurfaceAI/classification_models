from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

def train_validation_spilt_datasets(root, validation_size, train_transform, valid_transform, random_state):
    full_dataset = datasets.ImageFolder(root)

    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_idx, valid_idx = next(stratified_splitter.split(full_dataset, full_dataset.targets))

    # Aufteilung des Datasets basierend auf den Indizes
    train_dataset = Subset(full_dataset, train_idx)
    valid_dataset = Subset(full_dataset, valid_idx)

    # (samples_train, samples_valid,
    #  targets_train, targets_valid,
    #  imgs_train, imgs_valid) = train_test_split(dataset.samples, dataset.targets, dataset.imgs, test_size=validation_size, random_state=random_state, stratify=dataset.targets)
    
    # train_dataset = dataset
    # train_dataset.samples = samples_train
    # train_dataset.targets = targets_train
    # train_dataset.imgs = imgs_train
    # train_dataset.transform = train_transform

    # valid_dataset = dataset
    # valid_dataset.samples = samples_valid
    # valid_dataset.targets = targets_valid
    # valid_dataset.imgs = imgs_valid
    # valid_dataset.transform = valid_transform

    return train_dataset, valid_dataset

# def transform(height=256, width=256, rotation=0, horizontal_flip=None, ):
#     train_transforms = transforms.Compose([transforms.RandomRotation(10),
#                                            transforms.Resize((image_height, image_width)),
#                                            transforms.RandomHorizontalFlip(),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.485, 0.456, 0.406],
#                                                                     [0.229, 0.224, 0.225])])
    
def transform(resize=None,
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

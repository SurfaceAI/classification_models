import sys
sys.path.append('./')

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import copy

def compute_mean_std(train_dataset):
    """
    Compute mean and standard deviation of the training dataset.
    """
    mean = np.zeros(3)  # Assuming 3 channels for RGB images
    std = np.zeros(3)

    for path, target in train_dataset.samples:
        img = Image.open(path).convert("RGB")
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))

    mean /= len(train_dataset)
    std /= len(train_dataset)

    return mean, std


def train_validation_spilt_datasets(root, validation_size, train_transform, valid_transform, random_state):

    # create complete dataset
    complete_dataset = datasets.ImageFolder(root)

    # # split indices for training and validation sets
    # stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    # train_idx, valid_idx = next(stratified_splitter.split(complete_dataset, complete_dataset.targets))

    # # split datasets based on indices
    # train_dataset = Subset(complete_dataset, train_idx)
    # train_dataset.dataset.transform = train_transform
    # valid_dataset = Subset(complete_dataset, valid_idx)
    # valid_dataset.dataset.transform = valid_transform

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
    
    train_mean, train_std = compute_mean_std(train_dataset)#unklar wie ich diese Werte weitergeben soll

    return train_dataset, valid_dataset, train_mean, train_std

def custom_crop(img):
    cropped_img = transforms.functional.crop(img, 512, 256, 256, 256)
    return cropped_img
    
def transform(crop=None,
              resize=None,
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
        
    if crop is not None:
        transform_list.append(transforms.Lambda(custom_crop))
        
    # if crop is not None:
    #     transform_list.append(crop_image(*crop))
        
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


# def crop_image(image):
    
#     height, width, _ = image.shape #get original height and width

#     left = (width - 256) // 2
#     upper = 2 * height // 3
#     right = left + 256
#     lower = height
    
#     cropped_image = image[upper:lower, left:right]
    
#     return cropped_image

# import cv2
# import numpy as np
 
# img = cv2.imread('training_data/annotated_images/asphalt/bad/129525449781518.jpg')

# cv2.imshow("img", img)
# print(img.shape) # Print image shape

# height, width, _ = img.shape #get original height and width

# left = (width - 256) // 2
# #upper = height - 224
# upper = 2 * height // 3
# right = left + 256
# lower = height
 
# # Cropping an image
# cropped_image = img[upper:lower, left:right]
# cropped_image.shape
# cv2.imshow("cropped", cropped_image)
 
# # Display cropped image
# cv2.imshow("cropped", cropped_image)
 
# # Save the cropped image
# cv2.imwrite("Cropped Image.jpg", cropped_image)
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('image view', cropped_image)
# k = cv2.waitKey(0) & 0xFF #without this, the execution would crush the kernel on windows
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# #plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))




# import torch 
# import torchvision.transforms as transforms 
# from PIL import Image 


# image
  
# # create an transform for crop the image 
# transform = transforms.CenterCrop(200) 
  
# # use above created transform to crop  
# # the image 
# image_crop = transform(image) 
  
# # display result 
# image_crop.show() 
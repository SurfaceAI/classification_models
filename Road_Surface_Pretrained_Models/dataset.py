from torchvision import datasets
from sklearn.model_selection import train_test_split

def train_validation_spilt_datasets(root, validation_size, train_transform, valid_transform):
    dataset = datasets.ImageFolder(root)

    (samples_train, samples_valid,
     targets_train, targets_valid,
     imgs_train, imgs_valid) = train_test_split(dataset.samples, dataset.targets, dataset.imgs, test_size=validation_size)
    
    train_dataset = dataset
    train_dataset.samples = samples_train
    train_dataset.targets = targets_train
    train_dataset.imgs = imgs_train
    train_dataset.transform = train_transform

    valid_dataset = dataset
    valid_dataset.samples = samples_valid
    valid_dataset.targets = targets_valid
    valid_dataset.imgs = imgs_valid
    valid_dataset.transform = valid_transform

    return train_dataset, valid_dataset

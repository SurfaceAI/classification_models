import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, images, labels, img_names, cls):
        self.num_examples = images.shape[0]
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.cls = cls

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
            image = self.images[index]
            label = self.labels[index]
            img_name = self.img_names[index]
            cls = self.cls[index]

            return torch.Tensor(image), torch.Tensor(label), img_name, cls


def load_train(train_path, image_size, classes, max_index):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    
    for index, fields in enumerate(classes):
        if index > max_index:
            break
        
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Region Of Interest (ROI)
            height, width = image.shape[:2]
            newHeight = int(round(height/2))
            image = image[newHeight-5:height-50, 0:width]
            
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            images.append(image)
              
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)       
               
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)

    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls
            
            
        
            
def load_test(test_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read test images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(test_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            # Region Of Interest (ROI)
            height, width = image.shape[:2]
            newHeight = int(round(height/2))
            image = image[newHeight-5:height-50, 0:width]
                      
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
           
            if index == 0:
                images.append(image)
                
                label = np.zeros(len(classes))
                label[index] = 1.0

                labels.append(label)       
                
                flbase = os.path.basename(fl)

                img_names.append(flbase)
                

                cls.append(fields)
                              
            elif index == 1:
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                img_names.append(flbase)
                cls.append(fields)
            
              
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls



### This is the version without sklearn train_test_split function

# def read_train_sets(train_path, img_size, classes, validation_size, max_index):
    
#     images, labels, img_names, cls = load_train(train_path, img_size, classes, max_index)
#     images, labels, img_names, cls = shuffle(images, labels, img_names, cls, random_state=1)  
    
    
#     if isinstance(validation_size, float):
#         validation_size = int(validation_size * images.shape[0])

#     validation_images = images[:validation_size]
#     validation_labels = labels[:validation_size]
#     validation_img_names = img_names[:validation_size]
#     validation_cls = cls[:validation_size]

#     train_images = images[validation_size:]
#     train_labels = labels[validation_size:]
#     train_img_names = img_names[validation_size:]
#     train_cls = cls[validation_size:] 

#     data = Dataset()
#     data.train = CustomDataset(train_images, train_labels, train_img_names, train_cls)
#     data.valid = CustomDataset(validation_images, validation_labels, validation_img_names, validation_cls)

#     return data


### This is the version with sklearn train_test_split function

def read_train_sets(train_path, img_size, classes, validation_size, max_index):
    images, labels, img_names, cls = load_train(train_path, img_size, classes, max_index)
    #images, labels, img_names, cls = shuffle(images, labels, img_names, cls, random_state=1) 
    # shuffling is included in the train_test_split function 

    # Using train_test_split to split data into training and validation sets
    train_images, validation_images, train_labels, validation_labels, train_img_names, validation_img_names, train_cls, validation_cls = \
        train_test_split(images, labels, img_names, cls, test_size=validation_size, random_state=1)

    # Create data dictionary with train and valid keys
    data = Dataset()
    data.train = CustomDataset(train_images, train_labels, train_img_names, train_cls)
    data.valid = CustomDataset(validation_images, validation_labels, validation_img_names, validation_cls)

    return data



  

#function to create folders in the current directory
def create_folders(folder_names):
    try:
        # Get the current working directory
        current_directory = os.getcwd()

        # Iterate through the list of folder names and create each folder
        for folder_name in folder_names:
            folder_path = os.path.join(current_directory, folder_name)

            # Check if the folder already exists before creating it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_name}' created successfully at '{folder_path}'")
            else:
                print(f"Folder '{folder_name}' already exists at '{folder_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")



#save images
def save_image(image, folder, filename):
    # # Resize the image (you can add more preprocessing steps here)
    # resized_image = cv2.resize(image, (128, 128))  

    # Save the preprocessed image
    save_path = os.path.join(folder, filename + ".jpg")
    cv2.imwrite(save_path, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))




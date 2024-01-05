import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
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


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()
  
                 
  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls, random_state=1)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets


def read_test_sets(test_path, image_size, classes, random_seed):
    class Datasets(object):
        pass
    data_sets= Datasets()
    
    images, labels, img_names, cls = load_test(test_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls, random_state=1) 
    
    data_sets = DataSet(images, labels, img_names, cls)
    
    return data_sets
  

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



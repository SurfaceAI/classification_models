#------------------------------------
# Author: Xinqi Zhu
# Please cite paper https://arxiv.org/abs/1709.09890 if you use this code
#------------------------------------

import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config
from src.utils import preprocessing
from src import constants

import torch
import keras
import numpy as np
import os
from keras.models import Model

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import BatchNormalization
from keras.utils.data_utils import get_file
from keras import backend as K

#-------- surfaceai config  ---------

config = train_config.rateke_flatten_params



# config = dict(
#     project = "road-surface-classification-type",
#     name = "VGG16",
#     save_name = 'VGG16.pt',
#     architecture = "VGG16",
#     dataset = 'V4', #'annotated_images',
#     label_type = 'annotated', #'predicted
#     dataset_class = 'FlattenFolders', #'FlattenFolders', #'PartialImageFolder'
#     batch_size = 32,
#     valid_batch_size = 32,
#     epochs = 2,
#     learning_rate = 0.0001,
#     seed = 42,
#     validation_size = 0.2,
#     image_size_h_w = (256, 256),
#     crop = 'lower_middle_third',
#     normalization = 'from_data', # None, # 'imagenet', 'from_data'
#     # norm_mean = [0.485, 0.456, 0.406],
#     # norm_std = [0.229, 0.224, 0.225],
#     selected_classes = [constants.ASPHALT,
#                         constants.CONCRETE,
#                         constants.SETT,
#                         constants.UNPAVED,
#                         constants.PAVING_STONES,
#     ]

# )




#learning rate scheduler
def scheduler(epoch):
  learning_rate_init = 0.003
  if epoch > 3:
    learning_rate_init = 0.0005
  if epoch > 7:
    learning_rate_init = 0.0001
  return learning_rate_init

#loss weight modifier (defines which block the model puts more focus on during which epoch)
class LossWeightsModifier(keras.callbacks.Callback):
  def __init__(self, alpha, beta):
    self.alpha = alpha
    self.beta = beta
  def on_epoch_end(self, epoch, logs={}):
    if epoch == 8:
      K.set_value(self.alpha, 0.3)
      K.set_value(self.beta, 0.7)
    if epoch == 18:
      K.set_value(self.alpha, 0.1)
      K.set_value(self.beta, 0.9)
    if epoch == 28:
      K.set_value(self.alpha, 0)
      K.set_value(self.beta, 1)


#-------- dimensions ---------
img_rows, img_cols = 256, 256
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
#-----------------------------

train_size = 50

#--- coarse 1 classes ---
num_c = 5

#--- fine classes ---
num_classes  = 18

batch_size   = 128
epochs       = 2

#--- file paths ---
log_filepath = './tb_log_medium_dynamic/'
weights_store_filepath = './medium_dynamic_weights/'
train_id = '1'
model_name = 'weights_medium_dynamic_surfaceai'+train_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)


#-------------------- data loading surfaceai -------------

# general_transform = {
#         'resize': config.get('image_size_h_w'),
#         'crop': config.get('crop'),
#         'normalize': config.get('normalization'),
#     }

# augmentation = dict(
#     random_horizontal_flip = True,
#     random_rotation = 10,
# )

# from PIL import Image
# import numpy as np

# def load_images_from_file_paths(file_paths):
#     images = []
#     for file_path in file_paths:
#         image = Image.open(file_path).convert('RGB')  # Assuming RGB images, adjust if needed
#         image_array = np.array(image)
#         images.append(image_array)
#     return images

# Example usage:
# Assuming train_data.imgs is a list of (image_path, label) tuples
# Replace this with your actual train_data definition

train_data, valid_data = preprocessing.create_train_validation_datasets(data_root=config.get('root_data'),
                                                                        dataset=config.get('dataset'),
                                                                        selected_classes=config.get('selected_classes'),
                                                                        validation_size=config.get('validation_size'),
                                                                        general_transform=config.get('transform'),
                                                                        augmentation=config.get('augment'),
                                                                        random_state=config.get('random_seed'),
                                                                        is_regression=config.get('is_regression'),
                                                                        level=config.get('level'),
                                                                        )



#load all images from train_data
train_images = []
for i, image in enumerate(train_data): # or i, image in enumerate(dataset)
  image = train_data[i][0]
  train_images.append(image)

#convert to numpy array
stacked_train = torch.stack(train_images)
x_train = stacked_train.numpy()
x_train = np.moveaxis(x_train, 1, -1)

valid_images = []
for i, image in enumerate(valid_data): # or i, image in enumerate(dataset)
  image = valid_data[i][0]
  valid_images.append(image)

#convert to numpy array
stacked_valid = torch.stack(valid_images)
x_valid = stacked_valid.numpy()
x_valid = np.moveaxis(x_valid, 1, -1)

y_train = keras.utils.to_categorical(train_data.targets)
y_valid = keras.utils.to_categorical(valid_data.targets)

#here we define the parent classes for each fine grained class 
parent = {
  0:0, 1:0, 2:0, 3:0,
  4:1, 5:1, 6:1, 7:1,
  8:2, 9:2, 10:2, 11:2,
  12:3, 13:3, 14:3,
  15:4, 16:4, 17:4,
}


y_c_train = np.zeros((y_train.shape[0], num_c)).astype("float32")
y_c_valid = np.zeros((y_valid.shape[0], num_c)).astype("float32")


# Transform labels for coarse level
for i in range(y_train.shape[0]):
    y_c_train[i][parent[np.argmax(y_train[i])]] = 1.0

for i in range(y_valid.shape[0]):
    y_c_valid[i][parent[np.argmax(y_valid[i])]] = 1.0


#-------------------- data loading ----------------------
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# #rewrite
# x_train = x_train[:train_size]
# y_train = y_train[:train_size]
# x_test = x_test[:train_size]
# y_test = y_test[:train_size]

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# #---------------- data preprocessiong -------------------
# x_train = (x_train-np.mean(x_train)) / np.std(x_train)
# x_test = (x_test-np.mean(x_test)) / np.std(x_test)

#---------------------- make coarse 2 labels --------------------------
# parent_f = {
#   2:3, 3:5, 5:5,
#   1:2, 7:6, 4:6,
#   0:0, 6:4, 8:1, 9:2
# }
# y_c2_train = np.zeros((y_train.shape[0], num_c_2)).astype("float32")
# y_c2_test = np.zeros((y_test.shape[0], num_c_2)).astype("float32")
# for i in range(y_c2_train.shape[0]):
#   y_c2_train[i][parent_f[np.argmax(y_train[i])]] = 1.0
# for i in range(y_c2_test.shape[0]):
#   y_c2_test[i][parent_f[np.argmax(y_test[i])]] = 1.0

#---------------------- make coarse 1 labels --------------------------
#have to go with integers here as we are assessing classes over indeces
# parent = {
#   0:0, 1:0, 2:0, 3:0,
#   4:1, 5:1, 6:1, 7:1,
#   8:2, 9:2, 10:2, 11:2,
#   12:3, 13:3, 14:3,
#   15:4, 16:4, 17:4,
# }

# # Initialize y_c_train and y_c_test
# y_c_train = np.zeros((y_train.shape[0], num_c)).astype("float32")
# y_c_test = np.zeros((y_test.shape[0], num_c)).astype("float32")



# # Transform labels for one coarse level
# for i in range(y_train.shape[0]):
#     y_c_train[i][parent[np.argmax(y_train[i])]] = 1.0

# for i in range(y_test.shape[0]):
#     y_c_test[i][parent[np.argmax(y_test[i])]] = 1.0

# parent_f = {
#   0:0, 1:0, 2:0,
#   3:1, 4:1, 5:1, 6:1
# }

# #here, we
# y_c_train = np.zeros((y_train.shape[0], num_c)).astype("float32")
# y_c_test = np.zeros((y_test.shape[0], num_c)).astype("float32")
# for i in range(y_train.shape[0]):
#   y_train[i][parent_f[np.argmax(y_c_train[i])]] = 1.0
# for i in range(y_test.shape[0]):
#   y_test[i][parent_f[np.argmax(y_c_test[i])]] = 1.0


#----------------------- model definition ---------------------------
alpha = K.variable(value=0.99, dtype="float32", name="alpha") # A1 in paper
beta = K.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
#gamma = K.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper #out because we only have 2 levels

img_input = Input(shape=input_shape, name='input')

#--- block 1 ---
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#--- block 2 ---
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

#--- coarse branch ---
c_branch = Flatten(name='c_flatten')(x)
c_branch = Dense(256, activation='relu', name='c_fc2_surfaceai_1')(c_branch)
c_branch = BatchNormalization()(c_branch)
c_branch = Dropout(0.5)(c_branch)
c_branch = Dense(256, activation='relu', name='c_fc2')(c_branch)
c_branch = BatchNormalization()(c_branch)
c_branch = Dropout(0.5)(c_branch)
c_pred = Dense(num_c, activation='softmax', name='c_predictions_surfaceai')(c_branch)

#--- block 3 ---
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#--- coarse 2 branch ---
# c_2_bch = Flatten(name='c2_flatten')(x)
# c_2_bch = Dense(512, activation='relu', name='c2_fc_surfaceai_1')(c_2_bch)
# c_2_bch = BatchNormalization()(c_2_bch)
# c_2_bch = Dropout(0.5)(c_2_bch)
# c_2_bch = Dense(512, activation='relu', name='c2_fc2')(c_2_bch)
# c_2_bch = BatchNormalization()(c_2_bch)
# c_2_bch = Dropout(0.5)(c_2_bch)
# c_2_pred = Dense(num_c_2, activation='softmax', name='c2_predictions_surfaceai')(c_2_bch)

#--- block 4 ---
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

#--- fine block ---
x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu', name='fc_surfaceai_1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
fine_pred = Dense(num_classes, activation='softmax', name='predictions_surfaceai')(x)

model = Model(img_input, [c_pred, fine_pred], name='medium_dynamic')

#----------------------- compile and fit ---------------------------
sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              loss_weights=[alpha, beta],
              # optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
change_lw = LossWeightsModifier(alpha, beta)
cbks = [change_lr, tb_cb, change_lw]

model.fit(x_train, [y_c_train, y_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=cbks,
          validation_data=(x_valid, [y_c_valid, y_valid]))

#---------------------------------------------------------------------------------
# The following compile() is just a behavior to make sure this model can be saved.
# We thought it may be a bug of Keras which cannot save a model compiled with loss_weights parameter
#---------------------------------------------------------------------------------
model.compile(loss='categorical_crossentropy',
            #optimizer=keras.optimizers.Adadelta(),
            optimizer=sgd, 
            metrics=['accuracy'])

score = model.evaluate(x_valid, [y_c_valid, y_valid], verbose=0)
model.save(model_path)
print('score is: ', score)

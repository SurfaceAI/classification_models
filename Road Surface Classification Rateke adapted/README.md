# Simple CNN: Road Surface Classification Model Rateke et al. (2019)

You can either use this code for training the model yourself or for using different pre-trained models for prediction: 


## 1. Training the model

The model is trained in 2 steps: First, the Surface Type Model will be trained using the '01Surface Type' folder. All further instructions can be found in the README within this folder. In the second step a quality model is trained for each class of surface type. 


Before you can start training, create the following folders that you populate with your train and test images.

1. In the '01Surface Type' folder: 
'train_data' and 'test_data', the images should be saved in subfolders according to their label. The names of the subfolders (e.g. paved/sett) will be the classes tags later.

2. In the '01Surface Quality' folder: 
'train_data_asphalt_quality', 'train_data_paved_quality', 'test_data_asphalt_quality', 'test_data_paved_quality' and populate them in the same way as in the Type folder. Again, put in subfolders named after the respective quality label. 


3. Now go to the README in '01Surface Type'. 


## 2. Testing only

If you just want to classify images with a pretrained model to test it, you have to create the following folders and populate them with your testing images:

1. In the '01Surface Type' folder: 
'test_data', the images should be saved in subfolders according to their label (if available). The names of the subfolders (e.g. paved/sett) will be the classes tags later.

2. In the '01Surface Quality' folder: 
'test_data_asphalt_quality', 'test_data_paved_quality' and populate them in the same way as in the 01Surface Type folder. Again, put in subfolders named after the respective quality label. 

3. If you want to test only the Surface type, got to README in '01Surface Type', if you want to test Surface quality or both combined, go to the README in '02Surface Quality' folder. 



Reference: 
Github: https://github.com/thiagortk/Road-Surface-Classification
Scientific Article: https://seer.ufrgs.br/rita/article/view/RITA_VOL26_NR3_50


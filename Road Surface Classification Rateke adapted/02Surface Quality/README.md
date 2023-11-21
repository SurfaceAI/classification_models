This folder is used to train the road surface quality model and use the stacked models for classification. 

## Training the model:

For each class of surface type we train a separate model. Here, a model for asphalt and paved will be trained. Before doing this, the follwing steps should be executed: 

1. Create a folder called 'training_data_asphalt_quality' and 'training_data_paved_quality' in the directory you have saved the files in and populate it with your training and test images. The images should again be saved in subfolders according to their quality labels. The names of the subfolders (e.g. good/bad) will be the classes tags later.


3. Set the parameters: 
- Batch_size
- Validation size
- Learning rate
- Image size
- Number of channels
- Number of iterations

3. Run the trainAsphaltQuality.py and the trainPavedQuality.py files.



## Using the models for classification: 

Before you can use the test_combined.py file for classification execute the following steps: 

1. Create a folder called 'test_data_asphalt_quality' and 'test_data_paved_quality' and populate them with your test images. The subfolder names should be identical to the ones in 'train_data_xy'.


Ignore Steps 2. and 3. I still have to add the option to use different models in the quality part.
2. Specify which model you want to use for prediction by setting 'model= '.
3. Also specify the model name in the checkpoint file. (todo: how can this be done automatically? )

4. If you want to run the pretrained model from Rateke et al. (2019), you have to add a third subfolder in your test_data called 'unpaved'.

4. Run the test_combined.py file
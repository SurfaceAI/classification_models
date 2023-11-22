This folder is used to train the road surface quality model and use the stacked models for classification. 

## 1. Training the model:

For each class of surface type we train a separate model. Here, a model for asphalt and paved will be trained. Before doing this, the follwing steps should be executed. 

1. Set the parameters in trainAsphaltQuality.py and trainPavedQuality.py: 
- Batch_size
- Validation size
- Learning rate
- Image size
- Number of channels
- Number of iterations

2. Run the trainAsphaltQuality.py and the trainPavedQuality.py files.



## 2. Test the model:

Before you can use the test_combined.py file for classification execute the following steps: 

Ignore Steps 1. and 2. I still have to add the option to use different models here. 
1. Specify which model you want to use for prediction by setting 'model= '.
2. Also specify the model name in the checkpoint file. (todo: how can this be done automatically? )

4. Run the test_combined.py file
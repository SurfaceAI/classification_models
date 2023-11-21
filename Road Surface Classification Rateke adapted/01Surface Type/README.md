
This folder is used to train the road surface type model and use it for classification. 

## Training the model:

Before running the script the follwing steps should be executed: 

1. Create a folder called 'train_data' in the directory you have saved the files in and populate it with your train images. The images should be saved in subfolders according to their label. The names of the subfolders (e.g. paved/sett) will be the classes tags later.

2. Select how you want your trainig images to be preprocessed: For cropping only, choose 'dataset' for 'dataset= ', if you want them to be augmented, select 'dataset_augmented'.

3. Select how you want your model output to be named by specifying 'model= '

3. Set the parameters: 
- Batch_size
- Validation size
- Learning rate
- Image size
- Number of channels
- Number of iterations

3. Run the train.py file 



## Using the model for classification: 

Before you can use the test.py file for classification execute the following steps: 

1. Create a folder called 'test_data' in the same way you have created 'train_data' and add your test images. The subfolder names should be identical to the ones in 'train_data'.

2. Specify which model you want to use for prediction by setting 'model= '.

3. Also specify the model name in the checkpoint file. (todo: how can this be done automatically?)

4. If you want to run the pretrained model from Rateke et al. (2019), you have to add a third subfolder in your test_data called 'unpaved'.

4. Run the test.py file
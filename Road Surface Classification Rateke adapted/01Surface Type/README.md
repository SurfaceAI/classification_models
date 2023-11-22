
This folder is used to train the road surface type model and use it for classification. 

## 1. Training the model:

Before running the train.py script the follwing steps should be executed: 

1. Select how you want your trainig images to be preprocessed: For cropping only, choose 'dataset=dataset', if you want them to be augmented, select 'dataset_augmented'.

2. Select how you want your model output to be named by specifying 'model= '

3. Set the parameters: 
- Batch_size
- Validation size
- Learning rate
- Image size
- Number of channels
- Number of iterations

4. Run the train.py file, three model files should be saved in your current directory and in the '02Surface Quality' folder. Next, go to the README in '02Surface Quality'.


## 2. Testing: 

Before you can use the test.py file for classification execute the following steps: 

1. Specify which pretrained model you want to use for prediction by setting 'model= '.

2. Also specify the model name in the checkpoint file. (todo: how can this be done automatically?)

3. Check whether the pretrained model had the same number of classes as your test_data folder and adjust if not. If you want to run the pretrained model from Rateke et al. (2019) on our current OSM data, you have to add a third subfolder in your test_data called 'unpaved'.

4. Run the test.py file
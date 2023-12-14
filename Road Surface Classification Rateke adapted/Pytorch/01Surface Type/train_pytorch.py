import cv2
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision
import os
import mlflow
import shutil
import sys
import mlflow.pytorch
from mlflow import MlflowClient
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
import pprint
import glob




# This is the directory in which this .py file is in
execution_directory = os.path.dirname(os.path.abspath(__file__))

#First set which model and which data preprocessing to include, either just cropped or cropped and augmented
#Models: roadsurface-model.meta; roadsurface-model-augmented.meta
#Dataset files: "dataset", "dataset_augmented"

train_path = os.path.join(execution_directory, 'train_data_v2')
save_path = execution_directory
quality_path = os.path.join(os.path.dirname(execution_directory), '02Surface Quality') #our surface quality folder
model = "roadsurface-model"
dataset = "dataset_pytorch"
import dataset_pytorch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set hypterparameters 
batch_size = 64

learning_rate = 1e-4
num_epochs = 100

img_size = 128
num_channels = 3
validation_size = 0.2

#weights and biases experiment tracking, set tags
wandb.login()
wandb.init(
    #set project and tags 
    project = "road-surface-classification-type", 
    name = "run 10", 
    
    #track hyperparameters and run metadata
    config={
    "architecture": "CNN Rateke pytorch",
    "dataset": "v2",
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "seed": 42,
    "augmented": "No"
    }
) 


#w&b sweep hyperparameter optimization
# sweep_config = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "val_loss_avg"},
#     "parameters": {
#         "batch_size": {"values": [8, 16, 32]},
#         "num_epochs": {"values": [10, 20, 50]},
#         "learning_rate": {"values": [0.0001, 0.001]},
#     }
# }

#pprint.pprint(sweep_config)
#for bayes optimization
# metric = {
#     'name': 'loss',
#     'goal': 'minimize'   
#     }
# sweep_id = wandb.sweep(sweep_config, project="pytorch-CNN-sweep")
# config = wandb.config

# wandb.init(config=config)



#Prepare input data
classes = os.listdir(train_path) 
num_classes = len(classes)
max_index = 4 #maximum index we want to read, here we have 5 classes



torch.manual_seed(42)
numpy.random.seed(42)

import dataset_pytorch
# We shall load all the train and validation images and labels into memory using openCV and use that during train
data = dataset_pytorch.read_train_sets(train_path, img_size, classes, validation_size=validation_size, max_index=max_index)



print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

#Here, we print the first image of our train data after preprocessing to check how it looks.
# It should pop up in an image editor outside of this window. 
# cv2.imshow('image view',data.validid.images[0])
# k = cv2.waitKey(0) & 0xFF #without this, the execution would crush the kernel on windows
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
#first images of train and valid datasets are always the same

#Alternatively, we can also save our images in separate folders in our directory
# save_folder = os.path.join(execution_directory, 'preprocessed_images')

# # # Assuming data is an instance of DataSet
# for i in range(data.train.num_examples):
#     image = data.train.images[i].squeeze()  # Remove the batch dimension
#     dataset_pytorch.save_image(image, save_folder, f"image_{i}")


#todo
# all_transforms = transforms.Compose([transforms.Resize((32,32)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                           std=[0.2023, 0.1994, 0.2010])
#                                      ])


# Data loader objects allow us to iterate through our images in batches

train_loader = torch.utils.data.DataLoader(dataset = data.train,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        worker_init_fn=torch.manual_seed(42))


#check that the loader is reproducible with torch.manual_seed
# for image in train_loader:
#     print(image)

valid_loader = torch.utils.data.DataLoader(dataset = data.valid,
                                        batch_size = batch_size,
                                        shuffle = False, #in the validation set we don't need shuffling
                                        worker_init_fn=torch.manual_seed(42)
                                        )




class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU())    #output is (32,64,64) 
        # Weight initilization Layer 1
        # init.xavier_normal_(self.conv1[0].weight)
        # init.constant_(self.conv1[0].bias, 0.05)
        
        # Conv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()) #output is (32,32,32)
    
        # init.xavier_normal_(self.conv2[0].weight)
        # init.constant_(self.conv2[0].bias, 0.05)
        
        # Conv Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()) #output is (64, 16, 16)
        
        # init.xavier_normal_(self.conv3[0].weight)
        # init.constant_(self.conv3[0].bias, 0.05)
        
        self.flat = nn.Flatten() # output (16384)
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128, bias=True), 
            nn.ReLU()) 
        
        # init.xavier_normal_(self.fc1[0].weight) 
        # init.constant_(self.fc1[0].bias, 0.05)
        
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        # init.xavier_normal_(self.fc2.weight)
        # init.constant_(self.fc2.bias, 0.05)
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
   
#printing a picture 

# def imshow(img):
#     plt.imshow(numpy.transpose(img, (1, 2, 0)))
#     plt.show()

# dataiter = iter(train_loader)
# images, labels, _, _ = next(dataiter)
# img = images[20]

# imshow(img)


#initializing model and choosing loss functiona and optimizer
model= ConvNet(num_classes)
#model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss() #this already includes the Argmax
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)



def train(num_epochs):

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for i, (images, labels, _, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            #zero parameter gradients
            optimizer.zero_grad()
            
            #forward pass and loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()
            
            #zbackpropagation and update weights
            loss.backward()
            optimizer.step()
            
        model.eval()
        total = 0
        valid_total = 0
        accuracy_total = 0
        correct = 0
            
        with torch.no_grad():
            for i, (images, labels, _, _) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                valid_total += valid_loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                true_cls = torch.argmax(labels, dim=1)
                correct += (predicted == true_cls).sum().item()
                accuracy = 100 * correct / total
                accuracy_total += accuracy 
        
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(epoch+1, num_epochs, loss.item(), valid_loss, accuracy))
        wandb.log({'epoch': epoch+1, 'train loss': loss, 'validation loss': valid_loss, 'validation accuracy': accuracy})
                
    avg_train_loss = total_train_loss / (len(train_loader.dataset) // batch_size)
    val_loss_avg = valid_total / (len(valid_loader.dataset) // batch_size)
    accuracy_avg = accuracy_total / len(valid_loader)
    wandb.log({"avg train loss": avg_train_loss, "average validation loss": val_loss_avg, "average validation accuracy": accuracy_avg})
    
    return avg_train_loss
        
            
    

train(num_epochs)


#wandb.agent(sweep_id=sweep_id, function=train(config=config), count=5)


torch.save(model.state_dict(), "pytorch_CNN") 
wandb.save('pytorch_CNN.pt')

wandb.unwatch()




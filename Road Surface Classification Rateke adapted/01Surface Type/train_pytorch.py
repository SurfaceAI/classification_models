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
import torch.nn.init as init

mlflow.set_tracking_uri("http://127.0.0.1:5000")

#sys.path.append('./')


# This is the directory in which this .py file is in
execution_directory = os.path.dirname(os.path.abspath(__file__))

#mlflow
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

#First set which model and which data preprocessing to include, either just cropped or cropped and augmented
#Models: roadsurface-model.meta; roadsurface-model-augmented.meta
#Dataset files: "dataset", "dataset_augmented"

train_path = os.path.join(execution_directory, 'train_data')
save_path = execution_directory
quality_path = os.path.join(os.path.dirname(execution_directory), '02Surface Quality') #our surface quality folder
model = "roadsurface-model"
dataset = "dataset_pytorch"
import dataset_pytorch

#defining hypterparameters and input data
batch_size = 32
validation_size = 0.2
learning_rate = 1e-4
img_size = 128
num_channels = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Adding global pytorch seed 
#todo

#os.system('spd-say -t male3 "I will try to learn this, my master."')

#Prepare input data
classes = os.listdir(train_path) 
num_classes = len(classes)
num_epochs = 20
classes

torch.manual_seed(1)

# We shall load all the train and validation images and labels into memory using openCV and use that during train
data = dataset_pytorch.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
data.train.labels

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
                                           shuffle = True)

train_loader
#check that the loader is reproducible with torch.manual_seed
# for image in train_loader:
#     print(image)

valid_loader = torch.utils.data.DataLoader(dataset = data.valid,
                                           batch_size = batch_size,
                                           shuffle = True)

image = data.train.images[0]
image.shape

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU())     
        
        init.xavier_normal_(self.conv1[0].weight)
        init.constant_(self.conv1[0].bias, 0.05)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
    
        init.xavier_normal_(self.conv2[0].weight)
        init.constant_(self.conv2[0].bias, 0.05)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        
        init.xavier_normal_(self.conv3[0].weight)
        init.constant_(self.conv3[0].bias, 0.1)
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128), 
            nn.ReLU()) 
        self.fc2 = nn.Linear(128, num_classes)
        
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flat(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#initializing model and choosing loss functiona and optimizer
model= ConvNet(num_classes)
#model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

mlflow.pytorch.autolog()

def train(num_epochs):
    with mlflow.start_run() as run:
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels, img_names, cls) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels, img_names, cls in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                true_cls = torch.argmax(labels, dim=1)
                correct += (predicted == true_cls).sum().item()

                print('Accuracy of the network on the {} validation images: {} %'.format(222, 100 * correct / total))

    #print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


train(4)


# def train(num_epochs):
#     for epoch in range(num_epochs):
#         model.train()
#         # for images, labels in data.train:
#         #     images, labels = images.to(device), labels.to(device)
        
#         images = data.train.images
#         labels = data.train.labels

#             # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
            
#         model.eval()
#         with torch.no_grad():
#             total, correct = 0, 0
#             for images, labels in data.valid:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#             accuracy = correct / total
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {100 * accuracy:.2f}%')

    
# train(num_epochs=10)

# torch.save(model.state_dict(), os.path.join(save_path, f'{model}.pth'))

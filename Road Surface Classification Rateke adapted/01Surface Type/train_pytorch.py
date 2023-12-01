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



# weights and biases
wandb.login()

# start a new experiment
wandb.init(project="pytorch-CNN")
# capture a dictionary of hyperparameters with config
config = wandb.config 
config.learning_rate = 0.001
config.epochs = 100
config.seed = 42



# optional: save model at the end
# model.to_onnx()
# wandb.save("model.onnx")





### mlflow

# mlflow.pytorch.get_default_conda_env()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
# pytorch_experiment = mlflow.set_experiment("Pytorch_CNN")
# mlflow.end_run()
# #overview of experiments
# all_experiments = client.search_experiments()
# print(all_experiments)
# #sys.path.append('./')


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
#mlflow.log_param("device", device)

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
        # Conv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU())    #output is (32,64,64) 
        # Weight initilization Layer 1
        init.xavier_normal_(self.conv1[0].weight)
        init.constant_(self.conv1[0].bias, 0.05)
        
        # Conv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()) #output is (32,32,32)
    
        init.xavier_normal_(self.conv2[0].weight)
        init.constant_(self.conv2[0].bias, 0.05)
        
        # Conv Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()) #output is (64, 16, 16)
        
        init.xavier_normal_(self.conv3[0].weight)
        init.constant_(self.conv3[0].bias, 0.05)
        
        self.flat = nn.Flatten() # output (16384)
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128, bias=True), 
            nn.ReLU()) 
        
        init.xavier_normal_(self.fc1[0].weight) 
        init.constant_(self.fc1[0].bias, 0.05)
        
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        init.xavier_normal_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0.05)
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    
#checking our model step by step

# dataiter = iter(train_loader)
# images, labels, img_names, cls = next(dataiter)
# image = images[0]
# image = numpy.transpose(image, (1, 2, 0))
# plt.imshow(torchvision.utils.make_grid(image))

# conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2), 
#             nn.ReLU()) 

# conv2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU())
    
# conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU())

# flat = nn.Flatten(0,-1)
# fc1 =  nn.Sequential(
#             nn.Linear(64 * 16 * 16, 128, bias=True), 
#             nn.ReLU())

# fc2 = nn.Linear(128, num_classes, bias=True)

# image = images[0]
# image.shape

# x = conv1(image)
# x = conv2(x)
# x.shape
# x = conv3(x)
# x.shape
# x = flat(x)
# x.shape
# x = fc1(x)
# x.shape
# x = fc2(x)
# x.shape






#initializing model and choosing loss functiona and optimizer
model= ConvNet(num_classes)
#model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)



# optional: track gradients
wandb.watch(model, log="all")
  
#mlflow.pytorch.autolog()




def train(num_epochs):
    #with mlflow.start_run() as run:

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (images, labels, img_names, cls) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            
    avg_train_loss = train_loss / len(train_loader)
    wandb.log({"train_loss": avg_train_loss})
    # mlflow.log_metrics("train_loss", avg_train_loss)
    # mlflow.pytorch.log_model(model, "model")
    
    return avg_train_loss
           
            
def validate():
    model.eval()
    total = 0
    valid_total = 0
    valid_loss = 0
    accuracy_total = 0
    correct = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, img_names, cls in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            valid_loss = criterion(outputs, labels)
            valid_total += valid_loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            true_cls = torch.argmax(labels, dim=1)
            correct += (predicted == true_cls).sum().item()
            accuracy = 100 * correct / total
            accuracy_total += accuracy
            
            msg = "Validation Accuracy: {0:.2f}%,  Validation Loss: {1:.3f}"
            print(msg.format(accuracy, valid_loss))
    
    val_loss_avg = valid_total / len(valid_loader)
    accuracy_avg = accuracy_total / len(valid_loader)
    
    wandb.log({"validation_loss": val_loss_avg, "validation_accuracy": accuracy_avg})
    
    return val_loss_avg, accuracy_avg

    #print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


train(4)
validate()

# wandb.log({
#         "Train Loss": avg_train_loss,
#         "Validation Loss": val_loss_avg,
#         "Validation Accuracy": accuracy_avg})


torch.save(model.state_dict(), "pytorch CNN") # ???
wandb.save('pytorch_CNN.pt')

wandb.unwatch()
# with mlflow.start_run() as run:
#     # Log the parameters used for the model fit
#     mlflow.log_params(params)

#     # Log the error metrics that were calculated during validation
#     mlflow.log_metrics(metrics)

#     # Log an instance of the trained model for later use
#     mlflow.pytorch.log_model(pytorch_model=model, artifact_path="mlartifacts")







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

import cv2
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision
import os
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
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd



#sys.path.append(r'C:\Users\esthe\Documents\GitHub\classification_models')
import preprocessing
import helper
import training

# This is the directory in which this .py file is in
execution_directory = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(execution_directory, 'training_data/annotated_images_downsampled_without_concrete')
save_path = execution_directory
#quality_path = os.path.join(os.path.dirname(execution_directory), '02Surface Quality') #our surface quality folder
#dataset = "dataset_pytorch"
#import dataset_pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# config
data_version = 'annotated_images_downsampled_without_concrete'
batch_size = 16
valid_batch_size = 16
epochs = 20
learning_rate = 0.0001
seed = 42
img_size = 128
validation_size = 0.2

#Edith 
# image_height = 256
# image_width = 256
# norm_mean = [0.485, 0.456, 0.406]
# norm_std = [0.229, 0.224, 0.225]
# horizontal_flip = True
# random_rotation = 10

# wandb.login()
# wandb.init(
#     #set project and tags 
#     project = "road-surface-classification-type", 
#     name = "Simple CNN 3", 
    
#     #track hyperparameters and run metadata
#     config={
#     "architecture": "CNN Rateke pytorch",
#     "dataset": data_version,
#     "learning_rate": learning_rate,
#     "batch_size": batch_size,
#     "seed": seed,
#     "augmented": "No"
#     }
# ) 



# preprocessing Edith

general_transform = {
    'resize': (img_size, img_size),
    #'normalize': (norm_mean, norm_std),
}

# train_augmentation = {
#     'random_horizontal_flip': horizontal_flip,
#     'random_rotation': random_rotation,
# }


train_transform = preprocessing.transform(**general_transform)
valid_transform = preprocessing.transform(**general_transform)



#data_root= os.path.join(train_path, data_version)

train_data, valid_data = preprocessing.train_validation_spilt_datasets(train_path, validation_size, train_transform, valid_transform, random_state=seed)

num_classes = len(train_data.classes)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size)

# for inputs, labels in trainloader:
#     helper.multi_imshow(inputs, labels)
#     inputs, labels = inputs.to(device), labels.to(device)

#Here, we print the first image of our train data after preprocessing to check how it looks.
# It should pop up in an image editor outside of this window. 

# images, _ = next(iter(trainloader))
# img = images[3]
# img = numpy.transpose(img.numpy(), (1, 2, 0))

# img.shape
# #train_np = numpy.array(trainloader.dataset)

# plt.imshow('image view', img)
# k = cv2.waitKey(0) & 0xFF #without this, the execution would crush the kernel on windows
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# #plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))


#images, labels, img_names, true_cls = next(iter(trainloader))
#Prepare input data
classes = os.listdir(train_path) 
num_classes = len(classes)
max_index = num_classes - 1 #maximum index we want to read, here we have 5 classes


classes
torch.manual_seed(42)
numpy.random.seed(42)

#import dataset_pytorch
# We shall load all the train and validation images and labels into memory using openCV and use that during train
#data = dataset_pytorch.read_train_sets(train_path, img_size, classes, validation_size=validation_size, max_index=max_index)


# print("Complete reading input data. Will Now print a snippet of it")
# print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
# print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



# #Alternatively, we can also save our images in separate folders in our directory
# save_folder = os.path.join(execution_directory, 'preprocessed_images')

# # # Assuming data is an instance of DataSet
# for i in range(train_data.):
#     image = data.train.images[i].squeeze()# Remove the batch dimension
#     image = numpy.transpose(image, (1, 2, 0))
#     dataset_pytorch.save_image(image, save_folder, f"image_{i}")


#todo
# all_transforms = transforms.Compose([transforms.Resize((32,32)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                           std=[0.2023, 0.1994, 0.2010])
#                                      ])


# Data loader objects allow us to iterate through our images in batches

# trainloader = torch.utils.data.DataLoader(dataset = train_data,
#                                         batch_size = batch_size,
#                                         shuffle = True,
#                                         worker_init_fn=torch.manual_seed(42))


# #check that the loader is reproducible with torch.manual_seed
# # for image in trainloader:
# #     print(image)

# validloader = torch.utils.data.DataLoader(dataset = valid_data,
#                                         batch_size = batch_size,
#                                         shuffle = False, #in the validation set we don't need shuffling
#                                         worker_init_fn=torch.manual_seed(42)
#                                         )




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

# dataiter = iter(trainloader)
# images, labels, _, _ = next(dataiter)
# img = images[20]

# imshow(img)


##initializing model and choosing loss functiona and optimizer
model= ConvNet(num_classes)
#model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss() #this already includes the Argmax
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(trainloader)

def train(num_epochs):

    
    for epoch in range(num_epochs):
    
        model.train()
        total_train_loss = 0.0
        total_train = 0
        correct_train = 0
        accuracy_train_total = 0
        
        for i, (images, labels) in enumerate(trainloader):
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
           
            
            #prediction on training images
            total_train += labels.size(0)
            predictions_train = torch.argmax(outputs, 1)
            correct_train += (predictions_train == labels).sum().item()
            accuracy_train  = 100 * correct_train / total_train
            accuracy_train_total += accuracy_train
            
            
        model.eval()
        total_valid = 0
        valid_total = 0
        accuracy_valid_total = 0
        correct_valid = 0
    
       
        y_true = []
        y_pred = []
            
        with torch.no_grad():
            for i, (images, labels) in enumerate(validloader):
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.shape[0]
                #y_true.append(labels)
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                valid_total += valid_loss
                total_valid += labels.size(0)

                predictions_valid = torch.argmax(outputs, dim=1)
                #y_pred.append(predictions_valid)
                correct_valid += (predictions_valid == labels).sum().item()
                accuracy_valid = 100 * correct_valid / total_valid
                accuracy_valid_total += accuracy_valid
                
                for j in range(batch_size):
                    y_true.append(labels[j])
                    y_pred.append(predictions_valid[j])
                
                #conf_matrix = confusion_matrix(labels.cpu().numpy(), predictions_valid.cpu().numpy(), labels=numpy.arange(len(classes)))
                    
        avg_train_loss = total_train_loss / len(trainloader)
        avg_train_accuracy = accuracy_train_total / len(trainloader)
        avg_val_loss = valid_total / len(validloader)
        avg_valid_accuracy = accuracy_valid_total / len(validloader)
        
        #if i % 100 == 0:
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(epoch+1, num_epochs, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_valid_accuracy))
        #wandb.log({'epoch': epoch+1, 'train loss': avg_train_loss, 'train accuracy': avg_train_accuracy, 'validation loss': avg_val_loss, 'validation accuracy': avg_valid_accuracy})
    return y_pred, y_true

from sklearn.metrics import confusion_matrix
y_pred, y_true = train(20)
y_pred = torch.stack(y_pred)
y_true = torch.stack(y_true)
y_true
y_pred
confusion_matrix(y_true, y_pred)

from torchmetrics.classification import MulticlassConfusionMatrix

# confmat = MulticlassConfusionMatrix(num_classes=7)
# confmat.update(y_pred, y_true)
# fig, ax = confmat.plot


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".0f")
plt.xticks(rotation=45, ha="right")
plt.title('Confusion Matrix')
plt.show()


#wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        # y_true=y_true, preds=y_pred,
                        # class_names=classes)})



#wandb.agent(sweep_id=sweep_id, function=train(config=config), count=5)


#torch.save(model.state_dict(), "pytorch_CNN") 
#wandb.save('pytorch_CNN.pt')

#wandb.unwatch()




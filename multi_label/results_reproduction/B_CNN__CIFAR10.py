import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config
from src.utils import preprocessing
from src import constants



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import datasets
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import torchvision.utils as vutils
#from torchtnt.framework.callback import Callback

import wandb
import numpy as np
import os

config = train_config.B_CNN_multilabel

device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

print(device)

run = wandb.init(
    project="multi-label",
    tags=["B_CNN", "CIFAR_10"],
    config = config
)


def to_one_hot_tensor(y, num_classes):
    y = torch.tensor(y)
    return F.one_hot(y, num_classes)

#--- coarse classes ---
num_c1 = 2

#--- coarse 2 classes ---
num_c2 = 7

#--- fine classes ---
num_classes  = 10


# other parameters
# batch_size   = 128
# epochs       = 60

#--- file paths ---

weights_store_filepath = './B_CNN_weights/'
train_id = '1'
model_name = 'weights_B_CNN_surfaceai'+train_id+'.h5'
model_path = os.path.join(weights_store_filepath, model_name)



# Define the neural network model
class B_CNN(nn.Module):
    def __init__(self, num_c1, num_c2, num_classes):
        super(B_CNN, self).__init__()


        ### Block 1
        self.block1_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.block1_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        ### Block 2
        self.block2_layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.block2_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        ### Coarse 1 branch
        self.c1_fc = nn.Sequential(
            nn.Linear(8192, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c1_fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c1_fc2 = (
            nn.Linear(256, num_c1)
        )

        ### Block 3
        self.block3_layer1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.block3_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        ### Coarse 2 branch
        self.c2_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c2_fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Dropout(0.5))
        self.c2_fc2 = (
            nn.Linear(512, num_c2)
        )

        ### Block 4
        self.block4_layer1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.block4_layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        ### Fine Block
        self.f_flat = nn.Flatten() 
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            #nn.Linear(512 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            )
        self.fc2 = (
            nn.Linear(1024, num_classes)
        )     

    @ staticmethod
    def get_class_probabilies(x):
         return nn.functional.softmax(x, dim=1)

    def forward(self, x):
        x = self.block1_layer1(x) #[batch_size, 64, 256, 256]
        x = self.block1_layer2(x) #[batch_size, 64, 128, 128]

        x = self.block2_layer1(x)#[batch_size, 64, 128, 128] 
        x = self.block2_layer2(x) #(batch_size, 128, 64, 64)

        #coarse 1
        flat = x.reshape(x.size(0), -1) #(48, 524288)
        coarse1_output = self.c1_fc(flat) #(48,4096)
        coarse1_output = self.c1_fc1(coarse1_output)
        coarse1_output = self.c1_fc2(coarse1_output)

        x = self.block3_layer1(x)
        x = self.block3_layer2(x)

        #coarse2
        flat = x.reshape(x.size(0), -1) 
        coarse2_output = self.c2_fc(flat) 
        coarse2_output = self.c2_fc1(coarse2_output)
        coarse2_output = self.c2_fc2(coarse2_output)


        x = self.block4_layer1(x)
        x = self.block4_layer2(x) # output: [batch_size, 512 #channels, 16, 16 #height&width]

        flat = x.reshape(x.size(0), -1) #([48, 131072])
        fine_output = self.fc(flat) #([48, 4096])
        fine_output = self.fc1(fine_output) #([48, 4096])
        fine_output = self.fc2(fine_output) #[48, 18])

        return coarse1_output, coarse2_output, fine_output



#learning rate scheduler manual, it returns the multiplier for our initial learning rate
def lr_lambda(epoch):
  learning_rate_multi = 1.0
  if epoch > 42:
    learning_rate_multi = (1/6) # 0.003/6 to get lr = 0.0005
  if epoch > 52:
    learning_rate_multi = (1/30) # 0.003/30 to get lr = 0.0001
  return learning_rate_multi

# Loss weights modifier
class LossWeightsModifier():
    def __init__(self, alpha, beta, gamma):
        super(LossWeightsModifier, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def on_epoch_end(self, epoch):
        if epoch >= 10:
            self.alpha = torch.tensor(0.1)
            self.beta = torch.tensor(0.8)
            self.gamma = torch.tensor(0.1)
        elif epoch >= 20:
            self.alpha = torch.tensor(0.1)
            self.beta = torch.tensor(0.2)
            self.gamma = torch.tensor(0.7)
        elif epoch >= 30:
            self.alpha = torch.tensor(0.0)
            self.beta = torch.tensor(0.0)
            self.beta = torch.tensor(1.0)

        return self.alpha, self.beta, self.gamma

# Define the data loaders and transformations

# train_data, valid_data = preprocessing.create_train_validation_datasets(data_root=config.get('root_data'),
#                                                                         dataset=config.get('dataset'),
#                                                                         selected_classes=config.get('selected_classes'),
#                                                                         validation_size=config.get('validation_size'),
#                                                                         general_transform=config.get('transform'),
#                                                                         augmentation=config.get('augment'),
#                                                                         random_state=config.get('random_seed'),
#                                                                         is_regression=config.get('is_regression'),
#                                                                         level=config.get('level'),
#                                                                         )

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='./data', train=True,
                                        download=True, 
                                        transform=transform)

valid_data = datasets.CIFAR10(root='./data', train=False,
                                       download=True, 
                                       transform=transform)


# mapping = {
#     1: 0, 2: 1, 3: 2, 0: 3, 5: 4, 6: 5, 7: 6, 4: 7,
#     9: 8, 10: 9, 11: 10, 8: 11, 13: 12, 14: 13, 12: 14,
#     16: 15, 15: 16, 17: 17
# }


# train_data.targets = [mapping[target] for target in train_data.targets]



#create train and valid loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.get('batch_size'), shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.get('batch_size'), shuffle=False)


#create one-hot encoded tensors with the fine class labels
y_train = to_one_hot_tensor(train_data.targets, num_classes)
y_valid = to_one_hot_tensor(valid_data.targets, num_classes)


#-----Define label tree

#Coarse 2 labels
parent_f = torch.tensor([0, 2, 3, 5, 6, 5, 4, 6, 1, 2])

y_c2_train = torch.zeros(y_train.size(0), num_c2, dtype=torch.float32)
y_c2_valid = torch.zeros(y_train.size(0), num_c2, dtype=torch.float32)

# Transform labels for coarse level
for i in range(y_train.shape[0]):
    y_c2_train[i][parent_f[torch.argmax(y_train[i])]] = 1.0

for j in range(y_valid.shape[0]):
    y_c2_valid[j][parent_f[torch.argmax(y_valid[j])]] = 1.0




#here we define the label tree, left is the fine class (e.g. asphalt-excellent) and right the coarse (e.g.asphalt)
parent_c2 = torch.tensor([0, 0, 0, 1, 1, 1, 1])


y_c1_train = torch.zeros(y_c2_train.size(0), num_c1, dtype=torch.float32)
y_c1_valid = torch.zeros(y_c2_train.size(0), num_c1, dtype=torch.float32)

# y_c_train = torch.tensor((y_train.shape[0], num_c))
# y_c_valid = torch.tensor((y_valid.shape[0], num_c))

#classes = ('plane', 'car', 'bird', 'cat',
           #'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Transform labels for coarse level
for i in range(y_c1_train.shape[0]):
    y_c1_train[i][parent_c2[torch.argmax(y_c2_train[i])]] = 1.0

for j in range(y_c1_valid.shape[0]):
    y_c1_valid[j][parent_c2[torch.argmax(y_c2_valid[j])]] = 1.0


#print some images and check labels


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(train_loader)
# images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join(f'{fine_classes[labels[j]]:
# 


# Initialize the loss weights

alpha = torch.tensor(0.98)
beta = torch.tensor(0.01)
gamma = torch.tensor(0.01)

# Initialize the model, loss function, and optimizer
model = B_CNN(num_c1=2, num_c2=7, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# Set up learning rate scheduler
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_weights_modifier = LossWeightsModifier(alpha, beta, gamma)

# Train the model
writer = SummaryWriter('logs')



for epoch in range(config.get('epochs')):
    model.train()
    running_loss = 0.0
    coarse1_correct = 0
    coarse2_correct = 0
    fine_correct = 0

    for batch_index, data in enumerate(train_loader, 0):

        inputs, fine_labels = data
        #imshow(torchvision.utils.make_grid(inputs))


        # if batch_index == 0:  # Print only the first batch
        #     print("Batch Images:")
        #     images_grid = vutils.make_grid(inputs, nrow=8, padding=2, normalize=True)  # Assuming batch size is 64
        #     plt.figure(figsize=(16, 16))
        #     plt.imshow(np.transpose(images_grid, (1, 2, 0)))
        #     plt.axis('off')
        #     plt.show()


        inputs, fine_labels = inputs.to(device), fine_labels.to(device)

        optimizer.zero_grad()

        coarse2_labels = parent_f[fine_labels]
        coarse1_labels = parent_c2[coarse2_labels]

        coarse1_outputs, coarse2_outputs, fine_outputs = model.forward(inputs)
        coarse1_loss = criterion(coarse1_outputs, coarse1_labels)
        coarse2_loss = criterion(coarse2_outputs, coarse2_labels)
        fine_loss = criterion(fine_outputs, fine_labels)

        loss = alpha * coarse1_loss + beta * coarse2_loss + gamma * fine_loss #weighted loss functions for different levels
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 

        coarse1_probs = model.get_class_probabilies(coarse1_outputs)
        coarse1_predictions = torch.argmax(coarse1_probs, dim=1)
        coarse1_correct += (coarse1_predictions == coarse1_labels).sum().item()

        coarse2_probs = model.get_class_probabilies(coarse2_outputs)
        coarse2_predictions = torch.argmax(coarse2_probs, dim=1)
        coarse2_correct += (coarse2_predictions == coarse2_labels).sum().item()

        fine_probs = model.get_class_probabilies(fine_outputs)
        fine_predictions = torch.argmax(fine_probs, dim=1)
        fine_correct += (fine_predictions == fine_labels).sum().item()

        # if batch_index == 0:
        #     break

    #learning rate step        
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]

    #loss weights step
    alpha, beta, gamma = loss_weights_modifier.on_epoch_end(epoch)

    # epoch_loss = running_loss /  len(inputs) * (batch_index + 1) 
    # epoch_coarse_accuracy = 100 * coarse_correct / (len(inputs) * (batch_index + 1))
    # epoch_fine_accuracy = 100 * fine_correct / (len(inputs) * (batch_index + 1))
    epoch_loss = running_loss /  len(train_loader)
    epoch_coarse1_accuracy = 100 * coarse1_correct / len(train_loader.sampler)
    epoch_coarse2_accuracy = 100 * coarse2_correct / len(train_loader.sampler)
    epoch_fine_accuracy = 100 * fine_correct / len(train_loader.sampler)

    #writer.add_scalar('Training Loss', epoch_loss, epoch)

    # Validation
    model.eval()
    loss = 0.0
    val_running_loss = 0.0
    val_coarse1_correct = 0
    val_coarse2_correct = 0
    val_fine_correct = 0

    with torch.no_grad():
        for batch_index, (inputs, fine_labels) in enumerate(valid_loader):

            inputs, fine_labels = inputs.to(device), fine_labels.to(device)

            coarse2_labels = parent_f[fine_labels]
            coarse1_labels = parent_c2[coarse2_labels]

            coarse1_outputs, coarse2_outputs, fine_outputs = model.forward(inputs)

            coarse1_loss = criterion(coarse1_outputs, coarse1_labels)
            coarse2_loss = criterion(coarse2_outputs, coarse2_labels)
            fine_loss = criterion(fine_outputs, fine_labels)

            loss = (coarse1_loss + coarse2_loss + fine_loss) / 3
            val_running_loss += loss.item() 

            coarse1_probs = model.get_class_probabilies(coarse1_outputs)
            coarse1_predictions = torch.argmax(coarse1_probs, dim=1)
            val_coarse1_correct += (coarse1_predictions == coarse1_labels).sum().item()

            coarse2_probs = model.get_class_probabilies(coarse2_outputs)
            coarse2_predictions = torch.argmax(coarse2_probs, dim=1)
            val_coarse2_correct += (coarse2_predictions == coarse2_labels).sum().item()

            fine_probs = model.get_class_probabilies(fine_outputs)
            fine_predictions = torch.argmax(fine_probs, dim=1)
            val_fine_correct += (fine_predictions == fine_labels).sum().item()

            # if batch_index == 1:
            #     break

    # val_epoch_loss = val_running_loss /  (len(inputs) * (batch_index + 1))
    # val_epoch_coarse_accuracy = 100 * val_coarse_correct / (len(inputs) * (batch_index + 1))
    # val_epoch_fine_accuracy = 100 * val_fine_correct / (len(inputs) * (batch_index + 1))
    val_epoch_loss = val_running_loss /  len(valid_loader)
    val_epoch_coarse1_accuracy = 100 * val_coarse1_correct / len(valid_loader.sampler)
    val_epoch_coarse2_accuracy = 100 * val_coarse2_correct / len(valid_loader.sampler)
    val_epoch_fine_accuracy = 100 * val_fine_correct / len(valid_loader.sampler)

    if config.get('wandb_on'):
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": epoch_loss,
                    "train/accuracy/coarse1": epoch_coarse1_accuracy,
                    "train/accuracy/coarse2": epoch_coarse2_accuracy,
                    "train/accuracy/fine": epoch_fine_accuracy , 
                    "eval/loss": val_epoch_loss,
                    "eval/accuracy/coarse1": val_epoch_coarse1_accuracy,
                    "eval/accuracy/coarse2": val_epoch_coarse2_accuracy,
                    "eval/accuracy/fine": val_epoch_fine_accuracy,
                }
            )


    print(f"""
        Epoch: {epoch+1}: 
        Learning Rate: {before_lr} ->  {after_lr},
        Loss Weights: [alpha, beta] = [{alpha}, {beta}],
        Train loss: {epoch_loss:.3f}, 
        Train coarse 1 accuracy: {epoch_coarse1_accuracy:.3f}%, 
        Train coarse 2 accuracy: {epoch_coarse2_accuracy:.3f}%,
        Train fine accuracy: {epoch_fine_accuracy:.3f}%,
        Validation loss: {val_epoch_loss:.3f}, 
        Validation coarse 1 accuracy: {val_epoch_coarse1_accuracy:.3f}%,
        Validation coarse 2 accuracy: {val_epoch_coarse2_accuracy:.3f}%,  
        Validation fine accuracy: {val_epoch_fine_accuracy:.3f}% """)
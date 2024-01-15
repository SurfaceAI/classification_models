import os

import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("pytorch-lightning-experiment")
mlflow.pytorch.autolog()


# This is the directory in which this .py file is in
execution_directory = os.path.dirname(os.path.abspath(__file__))

# mlflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# First set which model and which data preprocessing to include, either just cropped or cropped and augmented
# Models: roadsurface-model.meta; roadsurface-model-augmented.meta
# Dataset files: "dataset", "dataset_augmented"

train_path = os.path.join(execution_directory, "train_data")
save_path = execution_directory
quality_path = os.path.join(
    os.path.dirname(execution_directory), "02Surface Quality"
)  # our surface quality folder
model = "roadsurface-model"
dataset = "dataset_pytorch"
import dataset_pytorch

# defining hypterparameters and input data
batch_size = 32
validation_size = 0.2
learning_rate = 1e-4
img_size = 128
num_channels = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adding global pytorch seed
# todo

# os.system('spd-say -t male3 "I will try to learn this, my master."')

# Prepare input data
classes = os.listdir(train_path)
num_classes = len(classes)
num_epochs = 20
classes

torch.manual_seed(1)

# We shall load all the train and validation images and labels into memory using openCV and use that during train
data = dataset_pytorch.read_train_sets(
    train_path, img_size, classes, validation_size=validation_size
)
data.train.labels

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

# Here, we print the first image of our train data after preprocessing to check how it looks.
# It should pop up in an image editor outside of this window.
# cv2.imshow('image view',data.validid.images[0])
# k = cv2.waitKey(0) & 0xFF #without this, the execution would crush the kernel on windows
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# first images of train and valid datasets are always the same

# Alternatively, we can also save our images in separate folders in our directory
# save_folder = os.path.join(execution_directory, 'preprocessed_images')

# # # Assuming data is an instance of DataSet
# for i in range(data.train.num_examples):
#     image = data.train.images[i].squeeze()  # Remove the batch dimension
#     dataset_pytorch.save_image(image, save_folder, f"image_{i}")


# todo
# all_transforms = transforms.Compose([transforms.Resize((32,32)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                           std=[0.2023, 0.1994, 0.2010])
#                                      ])


# Data loader objects allow us to iterate through our images in batches
train_loader = torch.utils.data.DataLoader(
    dataset=data.train, batch_size=batch_size, shuffle=True
)

train_loader
# check that the loader is reproducible with torch.manual_seed
# for image in train_loader:
#     print(image)

valid_loader = torch.utils.data.DataLoader(
    dataset=data.valid, batch_size=batch_size, shuffle=True
)

image = data.train.images[0]
image.shape


class ConvNet(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4):
        super(ConvNet, self).__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )  # output is (32,64,64)
        # Weight initilization Layer 1
        init.xavier_normal_(self.conv1[0].weight)
        init.constant_(self.conv1[0].bias, 0.05)

        # Conv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )  # output is (32,32,32)

        init.xavier_normal_(self.conv2[0].weight)
        init.constant_(self.conv2[0].bias, 0.05)

        # Conv Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )  # output is (64, 16, 16)

        init.xavier_normal_(self.conv3[0].weight)
        init.constant_(self.conv3[0].bias, 0.05)

        self.flat = nn.Flatten()  # output (16384)

        self.fc1 = nn.Sequential(nn.Linear(64 * 16 * 16, 128, bias=True), nn.ReLU())

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

    def configure_optimizers(self):
        # Define optimizer and optionally scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels, _, _ = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def on_train_epoch_end(self, outputs):
        # Log the average training loss at the end of each epoch
        avg_train_loss = torch.stack(outputs).mean()
        self.avg_train_loss = avg_train_loss
        self.log("train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        images, labels, _, _ = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        return {"val_loss": loss, "correct": correct, "total": labels.size(0)}

    def on_validation_epoch_end(self, outputs):
        # Calculate and log validation loss and accuracy
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        total_correct = sum([x["correct"] for x in outputs])
        total_samples = sum([x["total"] for x in outputs])
        val_accuracy = total_correct / total_samples * 100.0

        self.log("val_loss", avg_val_loss)
        self.log("val_accuracy", val_accuracy)


def train_lightning_model(num_epochs, train_loader, valid_loader):
    model = ConvNet(num_classes=num_classes)
    trainer = pl.Trainer(max_epochs=num_epochs)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {"num_epochs": num_epochs, "learning_rate": model.learning_rate}
        )

        # Log the model architecture
        mlflow.pytorch.log_model(model, "model")

        trainer.fit(model, train_loader, valid_loader)


# Assuming you have train_loader and valid_loader
train_lightning_model(
    num_epochs=4, train_loader=train_loader, valid_loader=valid_loader
)

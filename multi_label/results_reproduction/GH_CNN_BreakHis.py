import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config
from src.utils import preprocessing, checkpointing
from src.utils.helper import * 
from src import constants
import wandb
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import lr_scheduler

from src.architecture.vgg16_GH_CNN import GH_CNN

from datetime import datetime
import time
import numpy as np
import os
from PIL import Image
from torchvision import transforms


class BreakHisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []

        for mag_dir in ['40X', '100X', '200X', '400X']:
            mag_path = os.path.join(root_dir, mag_dir)
            if not os.path.isdir(mag_path):
                continue

            for file_name in os.listdir(mag_path):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if necessary
                    file_path = os.path.join(mag_path, file_name)
                    label = file_name.split('_')[2]  # Extracting the label (e.g., 'A' from 'SOB_B_A-14-22549AB-40-001.png')
                    self.image_paths.append(file_path)
                    self.labels.append(label)
                    if label not in self.classes:
                        self.classes.append(label)

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label



config = train_config.GH_CNN
torch.manual_seed(config.get("seed"))
np.random.seed(config.get("seed"))


device = torch.device(
        f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
    )

print(device)

root = r"C:\Users\esthe\Documents\GitHub\classification_models\data\standard_datasets\break_his"

if config.get('wandb_on'):
    run = wandb.init(
        project=config.get('project'),
        name=config.get('name'),
        config = config
    )

#--- file paths ---

level = config.get("level").split("/", 1)
type_class = None
if len(level) == 2:
    type_class = level[-1]

start_time = datetime.fromtimestamp(
        time.time() if not config.get('wandb_on') else run.start_time
    ).strftime("%Y%m%d_%H%M%S")
id = "" if not config.get('wandb_on') else "-" + run.id
saving_name = (
    config.get('level') + "-" + config.get("model") + "-" + start_time + id + ".pt"
)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dir = os.path.join(root, 'train')
test_dir = os.path.join(root, 'test')

train_data = BreakHisDataset(root_dir=train_dir, transform=transform)
valid_data = BreakHisDataset(root_dir=test_dir, transform=transform)

#fine classes
num_classes = len(train_data.classes)
#coarse classes
num_c = len(Counter([entry.split('_')[0] for entry in train_data.classes]))

#create train and valid loader
train_loader = DataLoader(train_data, batch_size=config.get('batch_size'), shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config.get('batch_size'), shuffle=False)

#create one-hot encoded tensors with the fine class labels
y_train = to_one_hot_tensor(train_data.labels, num_classes)
y_valid = to_one_hot_tensor(valid_data.labels, num_classes)


#here we define the label tree, left is the fine class (e.g. asphalt-excellent) and right the coarse (e.g.asphalt)
parent = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4])


y_c_train = torch.zeros(y_train.size(0), num_c, dtype=torch.float32)
y_c_valid = torch.zeros(y_train.size(0), num_c, dtype=torch.float32)

# Transform labels for coarse level
for i in range(y_train.shape[0]):
    y_c_train[i][parent[torch.argmax(y_train[i])]] = 1.0

for j in range(y_valid.shape[0]):
    y_c_valid[j][parent[torch.argmax(y_valid[j])]] = 1.0

class LossWeightsModifier_GH():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch):
        if 0.15 * config.get('epochs') <= epoch < 0.25 * config.get('epochs'):
            self.alpha = torch.tensor(0.5)
            self.beta = torch.tensor(0.5)
        elif epoch >= 0.25 * config.get('epochs'):
            self.alpha = torch.tensor(0.0)
            self.beta = torch.tensor(1.0)
            
        return self.alpha, self.beta

# Initialize the loss weights

alpha = torch.tensor(1)
beta = torch.tensor(0)

# Initialize the model, loss function, and optimizer
model = GH_CNN(num_c=num_c, num_classes=num_classes)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=config.get('learning_rate'), momentum=0.9)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# Set up learning rate scheduler
#scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_weights_modifier = LossWeightsModifier_GH(alpha, beta)

# Train the model
checkpointer = checkpointing.CheckpointSaver(
        dirpath=config.get("root_model"),
        saving_name=saving_name,
        decreasing=True,
        config=config,
        dataset=valid_loader.dataset,
        top_n=config.get("checkpoint_top_n", const.CHECKPOINT_DEFAULT_TOP_N),
        early_stop_thresh=config.get("early_stop_thresh", const.EARLY_STOPPING_DEFAULT),
        save_state=config.get("save_state", True),
    )

for epoch in range(config.get('epochs')):
    model.train()
    running_loss = 0.0
    coarse_correct = 0
    fine_correct = 0
    
    for batch_index, (inputs, fine_labels) in enumerate(train_loader):
    
        inputs, labels = inputs.to(device), fine_labels.to(device)
        
        optimizer.zero_grad()
        coarse_labels = parent[fine_labels].to(device)
        coarse_one_hot = to_one_hot_tensor(coarse_labels, num_c).to(device)
        
        #basic model
        raw_coarse, raw_fine = model.forward(inputs)
        
        #3 different training phases
        if epoch < 0.15 * config.get('epochs'):
            coarse_outputs, fine_outputs = raw_coarse, raw_fine
            
        elif 0.15 * config.get('epochs') <= epoch < 0.25 * config.get('epochs'): 
            coarse_outputs, fine_outputs = model.teacher_forcing(raw_coarse, raw_fine, coarse_one_hot)
            
        else:
            coarse_outputs, fine_outputs = model.bayesian_adjustment(raw_coarse, raw_fine)

        coarse_loss = criterion(coarse_outputs, coarse_labels)
        fine_loss = criterion(fine_outputs, fine_labels)
        
        coarse_probs = model.get_class_probabilies(coarse_outputs)
        coarse_predictions = torch.argmax(coarse_probs, dim=1)
        coarse_correct += (coarse_predictions == coarse_labels).sum().item()
        
        fine_probs = model.get_class_probabilies(fine_outputs)
        fine_predictions = torch.argmax(fine_probs, dim=1)
        fine_correct += (fine_predictions == fine_labels).sum().item()
        
        loss_h = torch.sum(alpha * coarse_loss + beta * fine_loss)
        
        #coarse only, weights should be (1,0)
        if epoch < 0.15 * config.get('epochs'):
            loss = loss_h   
           
        #teacher forcing 
        elif 0.15 * config.get('epochs') <= epoch < 0.25 * config.get('epochs'):
            loss = loss_h 
            
        #added calculating the loss_v (greatest error on a prediction where coarse and subclass prediction dont match)
        else:
            mismatched_indices = (coarse_predictions != parent[fine_predictions])
            max_mismatched_coarse_loss = max(coarse_loss[mismatched_indices])
            max_mismatched_fine_loss = max(fine_loss[mismatched_indices])
            loss_v = max(max_mismatched_coarse_loss, max_mismatched_fine_loss)
            loss = loss_h + loss_v
            
        #backward step
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 
        
        # if batch_index == 0:
        #     break

from torch import nn
from torchvision import models
from collections import OrderedDict

architecture = "Simple CNN not pretrained"
   
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU())    #output is (32,64,64) 
        
        # Conv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()) #output is (32,32,32)
        
        # Conv Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()) #output is (64, 16, 16)
        
        self.flat = nn.Flatten() # output (16384 when 128 image size, 65536 when 256 image size)
        
        self.fc1 = nn.Sequential(
            nn.Linear(65536, 256, bias=True), 
            nn.ReLU()) 
        
        self.fc2 = nn.Linear(256, num_classes, bias=True)
      
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

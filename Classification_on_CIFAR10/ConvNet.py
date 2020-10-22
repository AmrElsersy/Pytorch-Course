import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        # general, used in many layers
        self.max_pool = nn.MaxPool2d(2,2)
        
        # Size = Image_Size - Kernel Size + 1 ...... only if stride = 1 & padding = 0

        # size = 32 * 32 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)
        self.relu1 = nn.ReLU()
        # size = 28 * 28

        # size = 14 * 14 (because of max pooling)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        # size = 10 * 10 

        # size = 5 * 5 (because of max pooling) with 16 channel
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = nn.ReLU()
        # size = 3 * 3 with 32 channel
        # no max pooling

        # after flatten
        self.fc1 = nn.Linear(32 * 3 * 3, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10) # 10 classes

        # don't apply softmax for multi classification ... as it is imported already in the CrossEntropyLoss function

    def forward(self, x):

        x = self.max_pool(self.relu1(self.conv1(x)))
        x = self.max_pool(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        # Flatten
        x = x.view(-1, 32*3*3)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
        


## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torchvision.models as models

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #self.input = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5,
         #                      stride=1, padding=2)
        #self.vgg = models.resnet18(pretrained=True)
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Sequential( # (1,224,224)
                     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                               stride=1, padding=2), # (16,224,224)  
                     nn.BatchNorm2d(32),
                     nn.ReLU()
                     #nn.MaxPool2d(kernel_size=2) # (16,112,112)  
                     )  
        self.conv2 = nn.Sequential( # (16,112,112)  
                     nn.Conv2d(32, 32, 5, 1, 2), # (32,112,112)  
                     nn.ReLU(),  
                     nn.MaxPool2d(2) # (32,56,56)  
                     )
        self.conv3 = nn.Sequential( # (32,56,56)  
                     nn.Conv2d(32, 128, 5, 1, 2), # (64,56,56)  
                     nn.ReLU(),  
                     nn.MaxPool2d(2) # (64,28,28)  
                     )
        #self.dense = nn.Linear(64*56*56,1024)
        
        self.out = nn.Linear(128*56*56,136)
       
        
        #self.out = nn.Linear(1000,136)
        #self.out = nn.Linear(136,2)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)  
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) # batch，32,56,56）flatten（batch，32*56*56）  
        #x = self.f1(x) 
        '''
        x = self.input(x)
        x = self.vgg(x)
        
        '''
        x = self.out(x)
        # final output
        # return x
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

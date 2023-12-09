import torch
from torch import nn

class OriginalDepthwiseLayer(nn.Module):
    def __init__(self, inputChannel, outputChannel, filterHeight, stride):
        super(OriginalDepthwiseLayer, self).__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.filterHeight = filterHeight
        self.stride = stride
        self.padding = 0
        if(self.filterHeight == 3):
            self.padding = 1
        elif(self.filterHeight == 5):
            self.padding = 2
        self.conv1 = nn.Conv2d(in_channels = self.inputChannel, out_channels = self.outputChannel, kernel_size = self.filterHeight, stride = self.stride, padding = self.padding, groups = self.inputChannel, bias = False)
        
    def forward(self, x):
        return self.conv1(x)

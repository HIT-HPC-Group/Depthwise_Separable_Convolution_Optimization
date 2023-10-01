import torch
from torch import nn

class OriginalPointwiseLayer(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(OriginalPointwiseLayer, self).__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel

        self.conv1 = nn.Conv2d(in_channels = self.inputChannel, out_channels = self.outputChannel, kernel_size = 1, stride = 1, padding = 0, bias = False)
        
    def forward(self, x):
        return self.conv1(x)

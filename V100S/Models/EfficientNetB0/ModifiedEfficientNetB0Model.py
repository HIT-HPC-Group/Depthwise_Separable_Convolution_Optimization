import torch
from torch import nn
from torchvision.models import efficientnet_b0
from DepthwiseLayer import OptimizedDepthwiseLayer
from PointwiseLayer import OptimizedPointwiseLayer

class ModifiedEfficientNetB0(nn.Module):
    def __init__(self):
        super(ModifiedEfficientNetB0, self).__init__()
        self.modifiedModel = efficientnet_b0()
        self.modifiedModel.features[1][0].block[0][0] = OptimizedDepthwiseLayer(inputChannel = 32, outputChannel = 32, filterHeight = 3, stride = 1)
        self.modifiedModel.features[1][0].block[2][0] = OptimizedPointwiseLayer(inputChannel = 32, outputChannel = 16)

        self.modifiedModel.features[2][0].block[0][0] = OptimizedPointwiseLayer(inputChannel = 16, outputChannel = 96)
        self.modifiedModel.features[2][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 96, outputChannel = 96, filterHeight = 3, stride = 2)
        self.modifiedModel.features[2][0].block[3][0] = OptimizedPointwiseLayer(inputChannel = 96, outputChannel = 24)

        self.modifiedModel.features[2][1].block[0][0] = OptimizedPointwiseLayer(inputChannel = 24, outputChannel = 144)
        self.modifiedModel.features[2][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 144, outputChannel = 144, filterHeight = 3, stride = 1)
        self.modifiedModel.features[2][1].block[3][0] = OptimizedPointwiseLayer(inputChannel = 144, outputChannel = 24)

        self.modifiedModel.features[3][0].block[0][0] = OptimizedPointwiseLayer(inputChannel = 24, outputChannel = 144)
        self.modifiedModel.features[3][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 144, outputChannel = 144, filterHeight = 5, stride = 2)
        self.modifiedModel.features[3][0].block[3][0] = OptimizedPointwiseLayer(inputChannel = 144, outputChannel= 40)

        self.modifiedModel.features[3][1].block[0][0] = OptimizedPointwiseLayer(inputChannel = 40, outputChannel = 240)
        self.modifiedModel.features[3][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 240, outputChannel = 240, filterHeight = 5, stride = 1)
        self.modifiedModel.features[3][1].block[3][0] = OptimizedPointwiseLayer(inputChannel = 240, outputChannel = 40)

        self.modifiedModel.features[4][0].block[0][0] = OptimizedPointwiseLayer(inputChannel = 40, outputChannel = 240)
        self.modifiedModel.features[4][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 240, outputChannel = 240, filterHeight = 3, stride = 2)
        self.modifiedModel.features[4][0].block[3][0] = OptimizedPointwiseLayer(inputChannel = 240, outputChannel = 80)

        self.modifiedModel.features[4][1].block[0][0] = OptimizedPointwiseLayer(inputChannel = 80, outputChannel = 480)
        self.modifiedModel.features[4][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 480, outputChannel = 480, filterHeight = 3, stride = 1)
        self.modifiedModel.features[4][1].block[3][0] = OptimizedPointwiseLayer(inputChannel = 480, outputChannel = 80)

        self.modifiedModel.features[4][2].block[0][0] = OptimizedPointwiseLayer(inputChannel = 80, outputChannel = 480)
        self.modifiedModel.features[4][2].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 480, outputChannel = 480, filterHeight = 3, stride = 1)
        self.modifiedModel.features[4][2].block[3][0] = OptimizedPointwiseLayer(inputChannel = 480, outputChannel = 80)

        self.modifiedModel.features[5][0].block[0][0] = OptimizedPointwiseLayer(inputChannel = 80, outputChannel = 480)
        self.modifiedModel.features[5][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 480, outputChannel = 480, filterHeight = 5, stride = 1)
        self.modifiedModel.features[5][0].block[3][0] = OptimizedPointwiseLayer(inputChannel = 480, outputChannel = 112)

        self.modifiedModel.features[5][1].block[0][0] = OptimizedPointwiseLayer(inputChannel = 112, outputChannel = 672)
        self.modifiedModel.features[5][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 672, outputChannel = 672, filterHeight = 5, stride = 1)
        self.modifiedModel.features[5][1].block[3][0] = OptimizedPointwiseLayer(inputChannel = 672, outputChannel = 112)

        self.modifiedModel.features[5][2].block[0][0] = OptimizedPointwiseLayer(inputChannel = 112, outputChannel = 672)
        self.modifiedModel.features[5][2].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 672, outputChannel = 672, filterHeight = 5, stride = 1)
        self.modifiedModel.features[5][2].block[3][0] = OptimizedPointwiseLayer(inputChannel = 672, outputChannel = 112)

        self.modifiedModel.features[6][0].block[0][0] = OptimizedPointwiseLayer(inputChannel = 112, outputChannel = 672)
        self.modifiedModel.features[6][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 672, outputChannel = 672, filterHeight = 5, stride = 2)
        self.modifiedModel.features[6][0].block[3][0] = OptimizedPointwiseLayer(inputChannel = 672, outputChannel = 192)

        self.modifiedModel.features[6][1].block[0][0] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 1152)
        self.modifiedModel.features[6][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 5, stride = 1)
        self.modifiedModel.features[6][1].block[3][0] = OptimizedPointwiseLayer(inputChannel = 1152, outputChannel = 192)

        self.modifiedModel.features[6][2].block[0][0] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 1152)
        self.modifiedModel.features[6][2].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 5, stride = 1)
        self.modifiedModel.features[6][2].block[3][0] = OptimizedPointwiseLayer(inputChannel = 1152, outputChannel = 192)

        self.modifiedModel.features[6][3].block[0][0] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 1152)
        self.modifiedModel.features[6][3].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 5, stride = 1)
        self.modifiedModel.features[6][3].block[3][0] = OptimizedPointwiseLayer(inputChannel = 1152, outputChannel = 192)

        self.modifiedModel.features[7][0].block[0][0] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 1152)
        self.modifiedModel.features[7][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 3, stride = 1)
        self.modifiedModel.features[7][0].block[3][0] = OptimizedPointwiseLayer(inputChannel = 1152, outputChannel = 320)

        self.modifiedModel.features[8][0] = OptimizedPointwiseLayer(inputChannel = 320, outputChannel = 1280)

        self.modifiedModel.classifier[0] = nn.Linear(in_features=1280, out_features=10, bias=True)
            
    def forward(self, x):
        x = self.modifiedModel(x)
        return x

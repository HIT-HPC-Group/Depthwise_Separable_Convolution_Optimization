import torch
from torch import nn
from torchvision.models import mobilenet_v2
from DepthwiseLayer import OptimizedDepthwiseLayer
from PointwiseLayer import OptimizedPointwiseLayer

class ModifiedMobileNetV2(nn.Module):
    def __init__(self):
        super(ModifiedMobileNetV2, self).__init__()
        self.modifiedModel = mobilenet_v2()
        self.modifiedModel.features[1].conv[0][0] = OptimizedDepthwiseLayer(inputChannel = 32, outputChannel = 32, filterHeight = 3, stride = 1)
        self.modifiedModel.features[1].conv[1] = OptimizedPointwiseLayer(inputChannel = 32, outputChannel = 16)

        self.modifiedModel.features[2].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 16, outputChannel = 96)
        self.modifiedModel.features[2].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 96, outputChannel = 96, filterHeight = 3, stride = 2)
        self.modifiedModel.features[2].conv[2] = OptimizedPointwiseLayer(inputChannel = 96, outputChannel = 24)

        self.modifiedModel.features[3].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 24, outputChannel = 144)
        self.modifiedModel.features[3].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 144, outputChannel = 144, filterHeight = 3, stride = 1)
        self.modifiedModel.features[3].conv[2] = OptimizedPointwiseLayer(inputChannel = 144, outputChannel = 24)

        self.modifiedModel.features[4].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 24, outputChannel = 144)
        self.modifiedModel.features[4].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 144, outputChannel = 144, filterHeight = 3, stride = 2)
        self.modifiedModel.features[4].conv[2] = OptimizedPointwiseLayer(inputChannel = 144, outputChannel = 32)

        self.modifiedModel.features[5].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 32, outputChannel = 192)
        self.modifiedModel.features[5].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 192, outputChannel = 192, filterHeight = 3, stride = 1)
        self.modifiedModel.features[5].conv[2] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 32)

        self.modifiedModel.features[6].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 32, outputChannel = 192)
        self.modifiedModel.features[6].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 192, outputChannel = 192, filterHeight = 3, stride = 1)
        self.modifiedModel.features[6].conv[2] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 32)

        self.modifiedModel.features[7].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 32, outputChannel = 192)
        self.modifiedModel.features[7].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 192, outputChannel = 192, filterHeight = 3, stride = 2)
        self.modifiedModel.features[7].conv[2] = OptimizedPointwiseLayer(inputChannel = 192, outputChannel = 64)

        self.modifiedModel.features[8].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 64, outputChannel = 384)
        self.modifiedModel.features[8].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 384, outputChannel = 384, filterHeight = 3, stride = 1)
        self.modifiedModel.features[8].conv[2] = OptimizedPointwiseLayer(inputChannel = 384, outputChannel = 64)

        self.modifiedModel.features[9].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 64, outputChannel = 384)
        self.modifiedModel.features[9].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 384, outputChannel = 384, filterHeight = 3, stride = 1)
        self.modifiedModel.features[9].conv[2] = OptimizedPointwiseLayer(inputChannel = 384, outputChannel = 64)

        self.modifiedModel.features[10].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 64, outputChannel = 384)
        self.modifiedModel.features[10].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 384, outputChannel = 384, filterHeight = 3, stride = 1)
        self.modifiedModel.features[10].conv[2] = OptimizedPointwiseLayer(inputChannel = 384, outputChannel = 64)

        self.modifiedModel.features[11].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 64, outputChannel = 384)
        self.modifiedModel.features[11].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 384, outputChannel = 384, filterHeight = 3, stride = 1)
        self.modifiedModel.features[11].conv[2] = OptimizedPointwiseLayer(inputChannel = 384, outputChannel = 96)

        self.modifiedModel.features[12].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 96, outputChannel = 576)
        self.modifiedModel.features[12].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 576, outputChannel = 576, filterHeight = 3, stride = 1)
        self.modifiedModel.features[12].conv[2] = OptimizedPointwiseLayer(inputChannel = 576, outputChannel = 96)

        self.modifiedModel.features[13].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 96, outputChannel = 576)
        self.modifiedModel.features[13].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 576, outputChannel = 576, filterHeight = 3, stride = 1)
        self.modifiedModel.features[13].conv[2] = OptimizedPointwiseLayer(inputChannel = 576, outputChannel = 96)

        self.modifiedModel.features[14].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 96, outputChannel = 576)
        self.modifiedModel.features[14].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 576, outputChannel = 576, filterHeight = 3, stride = 2)
        self.modifiedModel.features[14].conv[2] = OptimizedPointwiseLayer(inputChannel = 576, outputChannel = 160)

        self.modifiedModel.features[15].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 160, outputChannel = 960)
        self.modifiedModel.features[15].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 960, outputChannel = 960, filterHeight = 3, stride = 1)
        self.modifiedModel.features[15].conv[2] = OptimizedPointwiseLayer(inputChannel = 960, outputChannel = 160)

        self.modifiedModel.features[16].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 160, outputChannel = 960)
        self.modifiedModel.features[16].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 960, outputChannel = 960, filterHeight = 3, stride = 1)
        self.modifiedModel.features[16].conv[2] = OptimizedPointwiseLayer(inputChannel = 960, outputChannel = 160)

        self.modifiedModel.features[17].conv[0][0] = OptimizedPointwiseLayer(inputChannel = 160, outputChannel = 960)
        self.modifiedModel.features[17].conv[1][0] = OptimizedDepthwiseLayer(inputChannel = 960, outputChannel = 960, filterHeight = 3, stride = 1)
        self.modifiedModel.features[17].conv[2] = OptimizedPointwiseLayer(inputChannel = 960, outputChannel = 320)

        self.modifiedModel.features[18][0] = OptimizedPointwiseLayer(inputChannel = 320, outputChannel = 1280)
        
        self.modifiedModel.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=10, bias=True)
        )
        
    def forward(self, x):
        x = self.modifiedModel(x)
        return x

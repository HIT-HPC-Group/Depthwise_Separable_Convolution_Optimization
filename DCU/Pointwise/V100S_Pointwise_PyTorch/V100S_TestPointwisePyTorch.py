import torch
from torch import nn
from OriginalLayer import OriginalPointwiseLayer
import pandas as pd
import numpy as np

def test(inputBatchNumber, inputChannel, inputHeight, inputWidth, outputChannel, loopTime, doPrint = False):
    # Randomly create input data and output data
    inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth, dtype = torch.float).to(cuda_device)

    original = OriginalPointwiseLayer(inputChannel, outputChannel).to(cuda_device)

    # Measure performane
    forwardTimeOriginal = 0

    with torch.no_grad():
        for _ in range(loopTime):
            starter.record()
            original(inputData)
            ender.record()
            torch.cuda.synchronize()
            forwardTimeOriginal += starter.elapsed_time(ender)
            
    if doPrint == True:
        print(f'InputBatchNumber: {inputBatchNumber}, InputChannel: {inputChannel}, InputHeight/Width: {inputHeight}, Output Channel: {outputChannel}')
        print('    Forward original: {:.3f} us'.format(forwardTimeOriginal * 1e3 / loopTime))

        return forwardTimeOriginal * 1e3 / loopTime

# start from here
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
loop = 10
starter = torch.cuda.Event(enable_timing = True)
ender = torch.cuda.Event(enable_timing = True)

# All possible batch numbers
batchNumberOptions = [1, 8, 16, 32, 64, 128]

# All layer structure parameters
# Input Channel, Input Height(Width), OutputChannel
layerConfigs = [[32, 112, 16],
                 [16, 112, 96],
                 [96, 56, 24],
                 [24, 56, 144],
                 [144, 56, 24],
                 [144, 28, 32],
                 [32, 28, 192],
                 [192, 28, 32],
                 [144, 28, 40],
                 [40, 28, 240],
                 [240, 28, 40],
                 [192, 14, 64],
                 [64, 14, 384],
                 [384, 14, 64],
                 [384, 14, 96],
                 [96, 14, 576],
                 [576, 14, 96],
                 [240, 14, 80],
                 [80, 14, 480],
                 [480, 14, 80],
                 [480, 14, 112],
                 [112, 14, 672],
                 [672, 14, 112],
                 [576, 7, 160],
                 [160, 7, 960],
                 [960, 7, 160],
                 [960, 7, 320],
                 [320, 7, 1280],
                 [672, 7, 192],
                 [192, 7, 1152],
                 [1152, 7, 192],
                 [1152, 7, 320]]

print("Start warm up.")
#warm up, no print info
for layerConfig in layerConfigs:
    for batchNumber in batchNumberOptions:
        test(batchNumber, layerConfig[0], layerConfig[1], layerConfig[1], layerConfig[2], 1, False)
print("Finish warm up.")

# Test
columns = [
    "Input Channel", "Input Height/Width", "Output Channel", 
    "Input Batch = 1 - V100S PyTorch (us)",
    "Input Batch = 8 - V100S PyTorch (us)",
    "Input Batch = 16 - V100S PyTorch (us)",
    "Input Batch = 32 - V100S PyTorch (us)",
    "Input Batch = 64 - V100S PyTorch (us)",
    "Input Batch = 128 - V100S PyTorch (us)"
]

resultTable = pd.DataFrame(columns = columns)
for layerConfig in layerConfigs:
    result = []
    for batchNumber in batchNumberOptions:
        currResult = test(batchNumber, layerConfig[0], layerConfig[1], layerConfig[1], layerConfig[2], loop, True)
        result.append("%.3f" % currResult)
    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[layerConfig[0], layerConfig[1], layerConfig[2],
        result[0], result[1], result[2], 
        result[3], result[4], result[5]], axis = 0), 
        columns = columns)

resultTable.to_csv("V100S_Pointwise_PyTorch_Result.csv")
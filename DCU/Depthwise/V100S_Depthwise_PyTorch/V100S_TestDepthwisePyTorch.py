import torch
from torch import nn
from OriginalLayer import OriginalDepthwiseLayer
import pandas as pd
import numpy as np

def test(inputBatchNumber, inputChannel, inputHeight, inputWidth, filterHeight, stride, loopTime, doPrint = False):
    outputChannel = inputChannel
    inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth).to(cuda_device)
    original = OriginalDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)

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
        print(f'InputBatchNumber: {inputBatchNumber}, InputChannel: {inputChannel}, InputHeight: {inputHeight}, InputWidth: {inputWidth}, FilterHeight: {filterHeight}, Stride: {stride}')
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
# Input Channel, Input Height/Width, Input Width, Fitler Height/Width, Stride
parameterList = [(32, 112, 112, 3, 1),
                 (144, 56, 56, 3, 1),
                 (192, 28, 28, 3, 1),
                 (240, 28, 28, 5, 1),
                 (384, 14, 14, 3, 1),
                 (480, 14, 14, 3, 1),
                 (480, 14, 14, 5, 1),
                 (576, 14, 14, 3, 1),
                 (672, 14, 14, 5, 1),
                 (960, 7, 7, 3, 1),
                 (1152, 7, 7, 3, 1),
                 (1152, 7, 7, 5, 1),

                 (96, 112, 112, 3, 2),
                 (144, 56, 56, 3, 2),
                 (144, 56, 56, 5, 2),
                 (192, 28, 28, 3, 2),
                 (240, 28, 28, 3, 2),
                 (576, 14, 14, 3, 2),
                 (672, 14, 14, 5, 2)
                 ]

print("Start warm up.")
#warm up, no print info
for parameters in parameterList:
    for batchNumber in batchNumberOptions:
        test(batchNumber, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], 1, False)
print("Finish warm up.")

# Test
columns = [
    "Input Channel", "Input Height/Width", "Filter Height/Width", "Stride", 
    "Input Batch = 1 - V100S PyTorch (us)",
    "Input Batch = 8 - V100S PyTorch (us)",
    "Input Batch = 16 - V100S PyTorch (us)",
    "Input Batch = 32 - V100S PyTorch (us)",
    "Input Batch = 64 - V100S PyTorch (us)",
    "Input Batch = 128 - V100S PyTorch (us)"
]

resultTable = pd.DataFrame(columns = columns)
for parameters in parameterList:
    result = []
    for batchNumber in batchNumberOptions:
        currResult = test(batchNumber, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], loop, True)
        result.append("%.3f" % currResult)
    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[parameters[0], parameters[1], parameters[3], parameters[4], 
        result[0], result[1], result[2], 
        result[3], result[4], result[5]], axis = 0), 
        columns = columns)

resultTable.to_csv("V100S_Depthwise_PyTorch_Result.csv")
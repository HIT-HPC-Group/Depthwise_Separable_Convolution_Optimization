import torch
from torch import nn
from DepthwiseLayer import OptimizedDepthwiseLayer
from OriginalLayer import OriginalDepthwiseLayer
import pandas as pd
import numpy as np

def test(inputBatchNumber, inputChannel, inputHeight, inputWidth, filterHeight, stride, loopTime, doPrint = False):
    if(filterHeight == 3):
        paddingHeight = paddingWidth = 1
    elif(filterHeight == 5):
        paddingHeight = paddingWidth = 2
    else:
        paddingHeight = paddingWidth = 0

    # Determine the output size
    outputBatchNumber = inputBatchNumber
    outputChannel = inputChannel
    outputHeight = int((inputHeight + paddingHeight * 2 - filterHeight) / stride + 1)
    outputWidth = int((inputWidth + paddingWidth * 2 - filterHeight) / stride + 1)

    # Randomly create input data and output data
    inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth).to(cuda_device)
    # outputData = torch.randn(outputBatchNumber, outputChannel, outputHeight, outputWidth).to(cuda_device)

    optimized = OptimizedDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)
    original = OriginalDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)

    # Measure performane
    forwardTimeOptimized = 0
    forwardTimeOriginal = 0

    # backwardTimeOptimized = 0
    # backwardTimeOriginal = 0
    with torch.no_grad():
        for _ in range(loopTime):
            starter.record()
            original(inputData)
            ender.record()
            torch.cuda.synchronize()
            forwardTimeOriginal += starter.elapsed_time(ender)
            
            """
            lossOriginal = loss_fn(output2, outputData)
            torch.cuda.synchronize()
            start = time.time()
            lossOriginal.backward()
            torch.cuda.synchronize()
            backwardTimeOriginal += time.time() - start
            """

            starter.record()
            optimized(inputData)
            ender.record()
            torch.cuda.synchronize()
            forwardTimeOptimized += starter.elapsed_time(ender)

            """
            lossOptimized = loss_fn(output1, outputData)
            torch.cuda.synchronize()
            start = time.time()
            lossOptimized.backward()
            torch.cuda.synchronize()
            backwardTimeOptimized += time.time() - start
            """

    if doPrint == True:
        print(f'InputBatchNumber: {inputBatchNumber}, InputChannel: {inputChannel}, InputHeight: {inputHeight}, InputWidth: {inputWidth}, FilterHeight: {filterHeight}, Stride: {stride}')
        print('    Forward optimized: {:.3f} us'.format(forwardTimeOptimized * 1e3 / loopTime))
        print('    Forward original: {:.3f} us'.format(forwardTimeOriginal * 1e3 / loopTime))

        #print('    Backward optimized: {:.3f} us'.format(backwardTimeOptimized * 1e6 / loopTime))
        #print('    Backward original: {:.3f} us'.format(backwardTimeOriginal * 1e6 / loopTime))

        return [forwardTimeOptimized * 1e3 / loopTime, forwardTimeOriginal * 1e3 / loopTime]
        
# start from here
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
# loss_fn = nn.CrossEntropyLoss()
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
    "Input Batch = 1 - Optimized (us)", "Input Batch = 1 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 8 - Optimized (us)", "Input Batch = 8 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 16 - Optimized (us)", "Input Batch = 16 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 32 - Optimized (us)", "Input Batch = 32 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 64 - Optimized (us)", "Input Batch = 64 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 128 - Optimized (us)", "Input Batch = 128 - PyTorch (us)", "Speed Up (%)"
]

resultTable = pd.DataFrame(columns = columns)
for parameters in parameterList:
    result = []
    for batchNumber in batchNumberOptions:
        currResult = test(batchNumber, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], loop, True)
        result.append("%.3f" % currResult[0])
        result.append("%.3f" % currResult[1])
        speedup = 100 * (currResult[1] - currResult[0]) / currResult[1]
        result.append("%.3f" % speedup)
    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[parameters[0], parameters[1], parameters[3], parameters[4], 
        result[0], result[1], result[2], 
        result[3], result[4], result[5], 
        result[6], result[7], result[8], 
        result[9], result[10], result[11],  
        result[12], result[13], result[14], 
        result[15], result[16], result[17]], axis = 0), 
        columns = columns)

resultTable.to_csv("DCU_Depthwise_Extension_Result.csv")
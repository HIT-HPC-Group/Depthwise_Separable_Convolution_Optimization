import torch
from torch import nn
from PointwiseLayer import OptimizedPointwiseLayer
from OriginalLayer import OriginalPointwiseLayer
import pandas as pd
import numpy as np

def test(inputBatchNumber, inputChannel, inputHeight, inputWidth, outputChannel, loopTime, doPrint = False):
    # Determine the output size
    outputBatchNumber = inputBatchNumber
    outputHeight = inputHeight
    outputWidth = inputWidth

    # Randomly create input data and output data
    inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth, dtype = torch.float).to(cuda_device)
    # outputData = torch.randn(outputBatchNumber, outputChannel, outputHeight, outputWidth, dtype = torch.float).to(cuda_device)

    optimized = OptimizedPointwiseLayer(inputChannel, outputChannel).to(cuda_device)
    original = OriginalPointwiseLayer(inputChannel, outputChannel).to(cuda_device)

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
        print(f'InputBatchNumber: {inputBatchNumber}, InputChannel: {inputChannel}, InputHeight/Width: {inputHeight}, Output Channel: {outputChannel}')
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
    "Input Batch = 1 - Optimized (us)", "Input Batch = 1 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 8 - Optimized (us)", "Input Batch = 8 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 16 - Optimized (us)", "Input Batch = 16 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 32 - Optimized (us)", "Input Batch = 32 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 64 - Optimized (us)", "Input Batch = 64 - PyTorch (us)", "Speed Up (%)",
    "Input Batch = 128 - Optimized (us)", "Input Batch = 128 - PyTorch (us)", "Speed Up (%)"
]

resultTable = pd.DataFrame(columns = columns)
for layerConfig in layerConfigs:
    result = []
    for batchNumber in batchNumberOptions:
        currResult = test(batchNumber, layerConfig[0], layerConfig[1], layerConfig[1], layerConfig[2], loop, True)
        result.append("%.3f" % currResult[0])
        result.append("%.3f" % currResult[1])
        speedup = 100 * (currResult[1] - currResult[0]) / currResult[1]
        result.append("%.3f" % speedup)
    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[layerConfig[0], layerConfig[1], layerConfig[2],
        result[0], result[1], result[2], 
        result[3], result[4], result[5], 
        result[6], result[7], result[8], 
        result[9], result[10], result[11],  
        result[12], result[13], result[14], 
        result[15], result[16], result[17]], axis = 0), 
        columns = columns)

resultTable.to_csv("DCU_Pointwise_Extension_Result.csv")
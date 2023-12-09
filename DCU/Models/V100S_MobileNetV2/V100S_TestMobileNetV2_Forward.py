import torch
from torch import nn
from torchvision.models import mobilenet_v2
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================================================
# Turn on the GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ===================================================================================================
# Create original mobilenet v2
originalModel = mobilenet_v2()
originalModel.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=10, bias=True)
        )
originalModel.to(device)

# ===================================================================================================
# Create data and measure forward pass time
starter = torch.cuda.Event(enable_timing = True)
ender = torch.cuda.Event(enable_timing = True)

def test(inputBatchNumber, loopTime, doPrint = False):
    # Randomly create input data and output data
    inputData = torch.randn(inputBatchNumber, 3, 224, 224, dtype = torch.float).to(device)

    # Measure performane
    forwardTimeOriginal = 0
    
    with torch.no_grad():
        for _ in range(loopTime):
            starter.record()
            originalModel(inputData)
            ender.record()
            torch.cuda.synchronize()
            forwardTimeOriginal += starter.elapsed_time(ender)
    
    if doPrint == True:
        print(f'InputBatchNumber: {inputBatchNumber}')
        print('    Forward Original: {:.3f} ms'.format(forwardTimeOriginal / loopTime))
        
        return forwardTimeOriginal / loopTime

# ===================================================================================================
loopTime = 100
batchSizeOptions = [1, 8, 16, 32, 64, 128]
modelNames = ["MobileNet V2"]

print("Start warm up.")
# warm up
for batchSize in batchSizeOptions:
    test(batchSize, 10, False)
print("Finish warm up.")

# Test
columns = [
    "Model Name",
    "Input Batch = 1 - V100S PyTorch (ms)",
    "Input Batch = 8 - V100S PyTorch (ms)",
    "Input Batch = 16 - V100S PyTorch (ms)",
    "Input Batch = 32 - V100S PyTorch (ms)",
    "Input Batch = 64 - V100S PyTorch (ms)",
    "Input Batch = 128 - V100S PyTorch (ms)"
]

resultTable = pd.DataFrame(columns = columns)
for modelName in modelNames:
    result = []
    for batchSize in batchSizeOptions:
        currResult = test(batchSize, loopTime, True)
        result.append("%.3f" % currResult)
    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[modelName,
        result[0], result[1], result[2], 
        result[3], result[4], result[5]], axis = 0), 
        columns = columns)

resultTable.to_csv("V100S_MobileNetV2_Forward_Result.csv")
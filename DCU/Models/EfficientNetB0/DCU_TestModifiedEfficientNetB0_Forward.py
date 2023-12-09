import torch
from torch import nn
from torchvision.models import efficientnet_b0
from ModifiedEfficientNetB0Model import ModifiedEfficientNetB0
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================================================
# Turn on the GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ===================================================================================================
# Create original efficientnet b0
originalModel = efficientnet_b0()
originalModel.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=10, bias=True)
        )
originalModel.to(device)

# ===================================================================================================
# Create modified efficientnet v2
modifiedModel = ModifiedEfficientNetB0()
modifiedModel.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=10, bias=True)
        )
modifiedModel.to(device)

# ===================================================================================================
# Create data and measure forward pass time
starter = torch.cuda.Event(enable_timing = True)
ender = torch.cuda.Event(enable_timing = True)

def test(inputBatchNumber, loopTime, doPrint = False):
    # Randomly create input data and output data
    inputData = torch.randn(inputBatchNumber, 3, 224, 224, dtype = torch.float).to(device)

    # Measure performane
    forwardTimeOptimized = 0
    forwardTimeOriginal = 0
    
    with torch.no_grad():
        for _ in range(loopTime):
            
            starter.record()
            originalModel(inputData)
            ender.record()
            torch.cuda.synchronize()
            forwardTimeOriginal += starter.elapsed_time(ender)

            starter.record()
            modifiedModel(inputData)
            ender.record()
            torch.cuda.synchronize()
            forwardTimeOptimized += starter.elapsed_time(ender)
    
    if doPrint == True:
        print(f'InputBatchNumber: {inputBatchNumber}')
        print('    Forward Modified: {:.3f} ms'.format(forwardTimeOptimized / loopTime))
        print('    Forward Original: {:.3f} ms'.format(forwardTimeOriginal / loopTime))
        
        return [forwardTimeOptimized / loopTime, forwardTimeOriginal / loopTime]

# ===================================================================================================
loopTime = 100
batchSizeOptions = [1, 8, 16, 32, 64, 128]
modelNames = ["EfficientNet B0"]

print("Start warm up.")
# warm up
for batchSize in batchSizeOptions:
    test(batchSize, 10, False)
print("Finish warm up.")

# Test
columns = [
    "Model Name",
    "Input Batch = 1 - Modified (ms)", "Input Batch = 1 - PyTorch (ms)", "Speed Up (%)",
    "Input Batch = 8 - Modified (ms)", "Input Batch = 8 - PyTorch (ms)", "Speed Up (%)",
    "Input Batch = 16 - Modified (ms)", "Input Batch = 16 - PyTorch (ms)", "Speed Up (%)",
    "Input Batch = 32 - Modified (ms)", "Input Batch = 32 - PyTorch (ms)", "Speed Up (%)",
    "Input Batch = 64 - Modified (ms)", "Input Batch = 64 - PyTorch (ms)", "Speed Up (%)",
    "Input Batch = 128 - Modified (ms)", "Input Batch = 128 - PyTorch (ms)", "Speed Up (%)"
]

resultTable = pd.DataFrame(columns = columns)
for modelName in modelNames:
    result = []
    for batchSize in batchSizeOptions:
        currResult = test(batchSize, loopTime, True)
        result.append("%.3f" % currResult[0])
        result.append("%.3f" % currResult[1])
        speedup = 100 * (currResult[1] - currResult[0]) / currResult[1]
        result.append("%.3f" % speedup)
    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[modelName,
        result[0], result[1], result[2], 
        result[3], result[4], result[5], 
        result[6], result[7], result[8], 
        result[9], result[10], result[11],  
        result[12], result[13], result[14], 
        result[15], result[16], result[17]], axis = 0), 
        columns = columns)

resultTable.to_csv("DCU_ModifiedEfficientNetB0_Forward_Result.csv")
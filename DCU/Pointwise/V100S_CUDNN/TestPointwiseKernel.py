import os
import time
import pandas as pd
import numpy as np

# All batch size
batchSizeList = [1, 8, 16, 32, 64, 128]

# All layer configurations in MobileNet V2 and EfficientNet B0
paramList = [
    [32, 112, 16], 
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
    [1152, 7, 320]
    ]

loopTime = 2

# Create table
columns = [
    "Input Channel", "Input Height/Width", "Output Channel", 
    "Input Batch = 1 - cuDNN (us)",
    "Input Batch = 8 - cuDNN (us)",
    "Input Batch = 16 - cuDNN (us)",
    "Input Batch = 32 - cuDNN (us)",
    "Input Batch = 64 - cuDNN (us)",
    "Input Batch = 128 - cuDNN (us)"
]

resultTable = pd.DataFrame(columns = columns)

# Run kernels
for param in paramList:
    result = []
    for batchSize in batchSizeList:
        os.system("rm -rf result.txt")
        for i in range(loopTime):
            print("Calculating Input Batch: " + str(batchSize) + ", " + 
            "Input Channel: " + str(param[0]) + ", " + 
            "Input Height: " + str(param[1]) + ", " +
            "Ouput Channel: " + str(param[2]) + " for " + str(i + 1) + " time.")
            cli = "./kernel" + " " + str(batchSize) + " " + str(param[0]) + " " + str(param[1]) + " " + str(param[2]) + " >> result.txt"
            os.system(cli)

        cudnnTime = 0
        with open("result.txt", "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if lines[i] == "cuDNN Calculation Finished.\n":
                    print("cuDNN Calculation Finished.")
                    cudnnTime += float(lines[i + 1].replace("cuDNN time : ", "").replace(" ms.\n", ""))
        cudnnTime = 1000 * cudnnTime / loopTime
        result.append("%.3f" % cudnnTime)

    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[param[0], param[1], param[2],
        result[0], result[1], result[2],
        result[3], result[4], result[5]], axis = 0),
        columns = columns)

# Output table
resultTable.to_csv("V100S_Pointwise_CUDNN_Result.csv")
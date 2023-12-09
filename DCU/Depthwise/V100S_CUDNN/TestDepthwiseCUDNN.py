import os
import time
import pandas as pd
import numpy as np
# All batch size
batchSizeList = [1, 8, 16, 32, 64, 128]

# All layer configurations in MobileNet V2 and EfficientNet B0
paramList = [
    [32, 112, 112, 3, 1],
    [144, 56, 56, 3, 1],
    [192, 28, 28, 3, 1],
    [240, 28, 28, 5, 1],
    [384, 14, 14, 3, 1],
    [480, 14, 14, 3, 1],
    [480, 14, 14, 5, 1],
    [576, 14, 14, 3, 1],
    [672, 14, 14, 5, 1],
    [960, 7, 7, 3, 1],
    [1152, 7, 7, 3, 1],
    [1152, 7, 7, 5, 1],
    [96, 112, 112, 3, 2],
    [144, 56, 56, 3, 2],
    [144, 56, 56, 5, 2],
    [192, 28, 28, 3, 2],
    [240, 28, 28, 3, 2],
    [576, 14, 14, 3, 2],
    [672, 14, 14, 5, 2]
    ]

loopTime = 3

# Create table
columns = [
    "Input Channel", "Input Height/Width", "Filter Height/Width", "Stride", 
    "Input Batch = 1 - cuDNN (us)",
    "Input Batch = 8 - cuDNN (us)",
    "Input Batch = 16 - cuDNN (us)",
    "Input Batch = 32 - cuDNN (us)",
    "Input Batch = 64 - cuDNN (us)",
    "Input Batch = 128 - cuDNN (us)",
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
        "Filter Height: " + str(param[3]) + ", " +
        "Stride: " + str(param[4]) + " " + "for " + str(i + 1) + " time.")
            cli = "./kernel" + " " + str(batchSize) + " " + str(param[0]) + " " + str(param[1]) + " " + str(param[3]) + " " + str(param[4]) + " >> result.txt"
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
        values=[param[0], param[1], param[3], param[4], 
        result[0], 
        result[1], 
        result[2],
        result[3],
        result[4],
        result[5]], axis = 0), 
        columns = columns)

# Output table
resultTable.to_csv("V100S_Depthwise_CUDNN_Result.csv")
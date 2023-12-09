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

loopTime = 1

# Create table
columns = [
    "Input Channel", "Input Height/Width", "Output Channel", 
    "Input Batch = 1 - Kernel (us)", "Input Batch = 1 - MIOpen (us)", "Speed Up (%)",
    "Input Batch = 8 - Kernel (us)", "Input Batch = 8 - MIOpen (us)", "Speed Up (%)",
    "Input Batch = 16 - Kernel (us)", "Input Batch = 16 - MIOpen (us)", "Speed Up (%)",
    "Input Batch = 32 - Kernel (us)", "Input Batch = 32 - MIOpen (us)", "Speed Up (%)",
    "Input Batch = 64 - Kernel (us)", "Input Batch = 64 - MIOpen (us)", "Speed Up (%)",
    "Input Batch = 128 - Kernel (us)", "Input Batch = 128 - MIOpen (us)", "Speed Up (%)"
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
            cli = "./build/kernel" + " " + str(batchSize) + " " + str(param[0]) + " " + str(param[1]) + " " + str(param[2]) + " >> result.txt"
            os.system(cli)

        miopenTime = 0
        kernelTime = 0
        with open("result.txt", "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                if lines[i] == "Kernel Calculation Correct.\n":
                    print(lines[i])
                    miopenTime += float(lines[i + 1].replace("MIOpen time : ", "").replace(" ms.\n", ""))
                    kernelTime += float(lines[i + 2].replace("Kernel time : ", "").replace(" ms.\n", ""))
        miopenTime = 1000 * miopenTime / loopTime
        kernelTime = 1000 * kernelTime / loopTime
        result.append("%.3f" % kernelTime)
        result.append("%.3f" % miopenTime)
        speedup = 100 * (miopenTime - kernelTime) / miopenTime
        result.append("%.3f" % speedup)

    resultTable = pd.DataFrame(
        np.insert(resultTable.values, len(resultTable.index), 
        values=[param[0], param[1], param[2],
        result[0], result[1], result[2],
        result[3], result[4], result[5],
        result[6], result[7], result[8],
        result[9], result[10], result[11],
        result[12], result[13], result[14],
        result[15], result[16], result[17]], axis = 0),
        columns = columns)

# Output table
resultTable.to_csv("DCU_Pointwise_Kernel_Result.csv")
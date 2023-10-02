import math

# Define constants of hardware resources
# Warp Size
GPU_WARP_SIZE = 32
DCU_WARP_SIZE = 64

# Registers per Streaming Multiprocessor
GPU_REGISTER_PER_SM = 65536
DCU_REGISTER_PER_SM = 65536

# Shared memory per Streaming Multiprocessor
GPU_SHARED_MEM_PER_SM = 98304 # (bytes)
DCU_SHARED_MEM_PER_SM = 65536 # (bytes)

# Total number of Streaming Multiprocessor
GPU_TOTAL_SM_NUM = 80
DCU_TOTAL_SM_NUM = 64

# ==============================================================================
# batch size 
batchSizeOptions = [1, 8, 16, 32, 64, 128]

# All layer structure parameters
# [Input Channel, Input Height(= Input Width), OutputChannel]
layerConfigs = [[32, 112, 16], [16, 112, 96],
                [96, 56, 24], [24, 56, 144], [144, 56, 24],
                [144, 28, 32], [32, 28, 192], [192, 28, 32], [144, 28, 40], [40, 28, 240], [240, 28, 40], 
                [192, 14, 64], [64, 14, 384], [384, 14, 64], [384, 14, 96], [96, 14, 576], [576, 14, 96],
                [240, 14, 80], [80, 14, 480], [480, 14, 80], [480, 14, 112], [112, 14, 672], [672, 14, 112],
                [576, 7, 160], [160, 7, 960], [960, 7, 160], [960, 7, 320], [320, 7, 1280],
                [672, 7, 192], [192, 7, 1152], [1152, 7, 192], [1152, 7, 320]]

"""
Return list of: 
[blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, horizontalRepeat, verticalRepeat]
"""
def getCombinationOptions(device, batchSize, layerConfig):
    combinationCandidates = []

    inputChannel = layerConfig[0]
    inputWidth = layerConfig[1]
    outputChannel = layerConfig[2]

    blockNumPerSMOptions = [2, 3, 4, 5, 6, 7, 8]
    if inputChannel == 32 and inputWidth == 112 and outputChannel == 16:
        a = 0
    elif inputChannel == 16 and inputWidth == 112 and outputChannel == 96:
        a = 0

    elif inputChannel == 96 and inputWidth == 56 and outputChannel == 24:
        a = 0
    elif inputChannel == 24 and inputWidth == 56 and outputChannel == 144:
        a = 0
    elif inputChannel == 144 and inputWidth == 56 and outputChannel == 24:
        a = 0

    elif inputChannel == 144 and inputWidth == 28 and outputChannel == 32:
        a = 0
    elif inputChannel == 32 and inputWidth == 28 and outputChannel == 192:
        a = 0
    elif inputChannel == 192 and inputWidth == 28 and outputChannel == 32:
        a = 0
    elif inputChannel == 144 and inputWidth == 28 and outputChannel == 40:
        a = 0
    elif inputChannel == 40 and inputWidth == 28 and outputChannel == 240:
        a = 0
    elif inputChannel == 240 and inputWidth == 28 and outputChannel == 40:
        a = 0
    
    elif inputChannel == 192 and inputWidth == 14 and outputChannel == 64:
        a = 0
    elif inputChannel == 64 and inputWidth == 14 and outputChannel == 384:
        a = 0
    elif inputChannel == 384 and inputWidth == 14 and outputChannel == 64:
        a = 0
    elif inputChannel == 384 and inputWidth == 14 and outputChannel == 96:
        a = 0
    elif inputChannel == 96 and inputWidth == 14 and outputChannel == 576:
        a = 0
    elif inputChannel == 576 and inputWidth == 14 and outputChannel == 96:
        a = 0
    elif inputChannel == 240 and inputWidth == 14 and outputChannel == 80:
        a = 0
    elif inputChannel == 80 and inputWidth == 14 and outputChannel == 480:
        a = 0
    elif inputChannel == 480 and inputWidth == 14 and outputChannel == 80:
        a = 0
    elif inputChannel == 480 and inputWidth == 14 and outputChannel == 112:
        a = 0
    elif inputChannel == 112 and inputWidth == 14 and outputChannel == 672:
        a = 0
    elif inputChannel == 672 and inputWidth == 14 and outputChannel == 112:
        a = 0

    elif inputChannel == 576 and inputWidth == 7 and outputChannel == 160:
        a = 0
    elif inputChannel == 160 and inputWidth == 7 and outputChannel == 960:
        a = 0
    elif inputChannel == 960 and inputWidth == 7 and outputChannel == 160:
        a = 0
    elif inputChannel == 960 and inputWidth == 7 and outputChannel == 320:
        a = 0
    elif inputChannel == 320 and inputWidth == 7 and outputChannel == 1280:
        a = 0
    elif inputChannel == 672 and inputWidth == 7 and outputChannel == 192:
        a = 0
    elif inputChannel == 192 and inputWidth == 7 and outputChannel == 1152:
        a = 0
    elif inputChannel == 1152 and inputWidth == 7 and outputChannel == 192:
        a = 0
    elif inputChannel == 1152 and inputWidth == 7 and outputChannel == 320:
        a = 0

    # horizontal, vertical
    repeatOptions = [horizontalRepeat, verticalRepeat]

    for blockNumPerSM in blockNumPerSMOptions:
        for warpNumPerBlock in warpNumPerBlockOptions:
            for outputChannelPerWarp in outputChannelPerWarpOptions:
                for outputWidthPerWarp in outputWidthPerWarpOptions:
                    for channelGroupSize in channelGroupSizeOptions:
                        for repeat in repeatOptions:
                                combinationCandidates.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])
                        
    return combinationCandidates

"""
For each possible hardware resource parameter combination,
calculate Streaming Multiprocessor Utilization and Arithmetic Intensity 
"""
def calculateUtilizationandAI(device, blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, outputBatchNumber, outputChannel, outputHeight, outputWidth, horizontalRepeat, verticalRepeat):
    # Get hardware resources limit of each thread block
    if device == "GPU":
        registerLimitPerThread = GPU_REGISTER_PER_SM / (blockNumPerSM * warpNumPerBlock * GPU_WARP_SIZE)
        sharedMemLimitPerBlock = GPU_SHARED_MEM_PER_SM / blockNumPerSM
        channelGroupPerWarp = GPU_WARP_SIZE / channelGroupSize
    else:
        registerLimitPerThread = DCU_REGISTER_PER_SM / (blockNumPerSM * warpNumPerBlock * DCU_WARP_SIZE)
        sharedMemLimitPerBlock = DCU_SHARED_MEM_PER_SM / blockNumPerSM
        channelGroupPerWarp = DCU_WARP_SIZE / channelGroupSize

    # Calculate hardware resources used by each thread block
    outputChannelPerThread = outputChannelPerWarp / channelGroupPerWarp
    resultRegisterNumPerThread = outputWidthPerWarp * outputChannelPerThread
    operandRegisterNumPerThread = outputWidthPerWarp + outputChannelPerThread

    if device == "GPU":
        tempRegisterNumPerThread= math.ceil((horizontalRepeat * outputWidthPerWarp * channelGroupSize) / (warpNumPerBlock * GPU_WARP_SIZE)) + math.ceil((verticalRepeat * outputChannelPerWarp * channelGroupSize) / (warpNumPerBlock * GPU_WARP_SIZE))
    else:
        tempRegisterNumPerThread = math.ceil((horizontalRepeat * outputWidthPerWarp * channelGroupSize) / (warpNumPerBlock * DCU_WARP_SIZE)) + math.ceil((verticalRepeat * outputChannelPerWarp * channelGroupSize) / (warpNumPerBlock * DCU_WARP_SIZE))

    totalRegisterPerBlock = resultRegisterNumPerThread + operandRegisterNumPerThread + tempRegisterNumPerThread + 40 # not sure if 40 is accurate
    totalSharedMemPerBlock = (horizontalRepeat * outputWidthPerWarp * channelGroupSize + verticalRepeat * outputChannelPerWarp * channelGroupSize) * 4 * 2 # total bytes
    
    # if block configuration exceeds the limit, then this config is invalid
    if totalRegisterPerBlock > registerLimitPerThread and totalSharedMemPerBlock > sharedMemLimitPerBlock:
        return None
    
    # if config is valid, then calculate SM utilization and arithmetic intensity
    totalBlockNum = outputBatchNumber * outputChannel * outputHeight * outputWidth / (outputChannelPerWarp * outputWidthPerWarp * warpNumPerBlock)
    if device == "GPU":
        SMUtilization = totalBlockNum / (blockNumPerSM * GPU_TOTAL_SM_NUM)
    else:
        SMUtilization = totalBlockNum / (blockNumPerSM * GPU_TOTAL_SM_NUM)

    ArithmeticIntensity = outputWidthPerWarp * outputChannelPerThread / (outputWidthPerWarp + outputChannelPerThread)
    return [SMUtilization, ArithmeticIntensity]

"""
Given a device, calculate the hardware resource parameters that can better utilize sm and get higher AI
"""
def getBestHardwareResourceParameters(device):
    for batchSize in batchSizeOptions:
        for layerConfig in layerConfigs:
            # 1. For each layer config, get all possible hardware resource parameters
            # [blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, horizontalRepeat, verticalRepeat]
            combinationOptions = getCombinationOptions(device, batchSize, layerConfig)

            # 2. From all possible parameter combinations, find those valid ones and 
            #    calculate hardware utilization and arithmetic intensity of them
            validCombinationUtilizationAI = []
            for combinationOption in combinationOptions:
                if calculateUtilizationandAI(device, combinationOptions[0], combinationOptions[1], combinationOptions[2], combinationOptions[3], combinationOptions[4], batchSize, layerConfig[2], layerConfig[1], layerConfig[1], combinationOptions[5], combinationOptions[6]) != None:
                    validCombinationUtilizationAI.append([])

def main():
    # GPU
    print("For GPU: ")
    getBestHardwareResourceParameters("GPU")
    print("GPU Finished!")
    print("====================================================")

    # DCU
    print("For DCU: ")
    getBestHardwareResourceParameters("DCU")
    print("DCU Finished!")

if __name__=="__main__":
    main()

# 1. If all combinations have > 1 utilization, then select the one with smallest utilization
# 2. If some combination < 1, then select those with utilization close to 1
# 3. Find the best one with largest AI
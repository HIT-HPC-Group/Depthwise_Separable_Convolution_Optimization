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
"""
Return list of: 
[blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, horizontalRepeat, verticalRepeat]
"""
def getCombinationOptions(device, batchSize, layerConfig):
    combinationOptions = []

    inputChannel = layerConfig[0]
    inputWidth = layerConfig[1]
    outputChannel = layerConfig[2]

    blockNumPerSMOptions = [2, 3, 4, 5, 6, 7, 8]

    if inputChannel == 32 and inputWidth == 112 and outputChannel == 16:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4], [4, 2], [8, 1]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 8:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 16 and inputWidth == 112 and outputChannel == 96:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4], [4, 2], [8, 1]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 8:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28, 56]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    """
    input size 56 x 56
    """
    elif inputChannel == 96 and inputWidth == 56 and outputChannel == 24:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 12, 24]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 12]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 24 and inputWidth == 56 and outputChannel == 144:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [8, 16, 24, 48, 72, 144]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [8, 24, 72]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [8, 4, 2]
                                elif outputChannelPerWarp == 72:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 144:
                                    channelGroupSizeOptions = [8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 72:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 144:
                                    channelGroupSizeOptions = [8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 144 and inputWidth == 56 and outputChannel == 24:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 8, 14, 28]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 12, 24]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 12]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [16, 8, 4]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [16, 8]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    """
    input size 28 x 28
    """
    elif inputChannel == 144 and inputWidth == 28 and outputChannel == 32:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4], [4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 16, 32]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 8, 16]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [4, 8]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 32 and inputWidth == 28 and outputChannel == 192:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4], [4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 96]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 48]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 24]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 192:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 192:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 192 and inputWidth == 28 and outputChannel == 32:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4], [4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 32]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 16]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [64, 32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 144 and inputWidth == 28 and outputChannel == 40:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 20, 40]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 20]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8, 4]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 40 and inputWidth == 28 and outputChannel == 240:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 1:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 14:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [8, 16, 24, 40, 48, 80, 120, 240]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [8, 24, 40, 120]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [8, 4, 2]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [8, 4, 2]
                                elif outputChannelPerWarp == 120:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 240:
                                    channelGroupSizeOptions = [8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [8, 4]
                                elif outputChannelPerWarp == 120:
                                    channelGroupSizeOptions = [8]
                                elif outputChannelPerWarp == 240:
                                    channelGroupSizeOptions = [8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 240 and inputWidth == 28 and outputChannel == 40:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[2, 2], [4, 1]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[4, 2]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    if repeat[0] == 2:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 4:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    elif repeat[0] == 7:
                        outputWidthPerWarpOptions = [4, 7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 10, 20, 40]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 10, 20]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8, 4]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    """
    input size 14 x 14
    """
    elif inputChannel == 192 and inputWidth == 14 and outputChannel == 64:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 32, 64]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 32]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 16]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [64, 32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 64 and inputWidth == 14 and outputChannel == 384:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 96]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 48]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 192:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 384:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            else:
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 192:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 384:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 384 and inputWidth == 14 and outputChannel == 64:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 32, 64]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 32]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 16]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [64, 32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 384 and inputWidth == 14 and outputChannel == 96:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 96]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8, 12]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [64, 32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])
                                
    elif inputChannel == 96 and inputWidth == 14 and outputChannel == 576:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 64, 72, 96, 144, 192, 288, 576]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 72, 96, 144, 288]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 48, 72, 144]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 24, 72]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 72:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 144:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 192:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 288:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 576:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 64:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 72:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 144:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 192:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 288:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 576:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])
        
    elif inputChannel == 576 and inputWidth == 14 and outputChannel == 96:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 96]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [2, 4, 8, 12]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [64, 32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [64, 32, 16]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [64, 32, 16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 240 and inputWidth == 14 and outputChannel == 80:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 16, 20, 40, 80]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 8, 20, 40]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [4, 20]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [16, 8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 80 and inputWidth == 14 and outputChannel == 480:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 12, 16, 20, 24, 32, 40, 48, 60, 80, 96, 120, 160, 240, 480]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 8, 12, 16, 20, 24, 40, 48, 60, 80, 120, 240]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [4, 8, 12, 20, 24, 40, 60, 120]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [4, 12, 20, 60]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 60:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 120:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 160:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 240:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 480:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 12:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 60:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 120:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 160:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 240:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 480:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 480 and inputWidth == 14 and outputChannel == 80:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 16, 20, 40, 80]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 8, 20, 40]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [4, 20]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 20:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 40:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 80:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 480 and inputWidth == 14 and outputChannel == 112:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 28, 56, 112]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 28, 56]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 14, 28]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 28:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 56:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 112:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 28:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 56:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 112:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 112 and inputWidth == 14 and outputChannel == 672:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[1, 8], [2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [4, 8, 12, 14, 16, 24, 28, 32, 42, 48, 56, 84, 96, 112, 168, 224, 336, 672]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [4, 8, 12, 14, 16, 24, 28, 42, 48, 56, 84, 112, 168, 336]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [4, 8, 12, 14, 24, 28, 42, 56, 84, 168]
                        elif repeat[1] == 8:
                            outputChannelPerWarpOptions = [4, 12, 14, 28, 42, 84]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 28:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 42:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 56:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 84:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 112:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 168:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 224:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 336:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 672:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                            elif device == "DCU":
                                if outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 24:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 28:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 32:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 42:
                                    channelGroupSizeOptions = [16]
                                elif outputChannelPerWarp == 48:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 56:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 84:
                                    channelGroupSizeOptions = [16, 8]
                                elif outputChannelPerWarp == 96:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 112:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 168:
                                    channelGroupSizeOptions = [16, 8, 4]
                                elif outputChannelPerWarp == 224:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                                elif outputChannelPerWarp == 336:
                                    channelGroupSizeOptions = [16, 8, 4, 2]
                                elif outputChannelPerWarp == 672:
                                    channelGroupSizeOptions = [16, 8, 4, 2, 1]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 672 and inputWidth == 14 and outputChannel == 112:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [4, 7, 8, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                # horizontal repeat, vertical repeat
                if warpNumPerBlock == 4:
                    repeatOptions = [[1, 4], [2, 2]]
                elif warpNumPerBlock == 7:
                    repeatOptions = [[7, 1]]
                elif warpNumPerBlock == 8:
                    repeatOptions = [[2, 4]]
                elif warpNumPerBlock == 14:
                    repeatOptions = [[7, 2], [14, 1]]
                for repeat in repeatOptions:
                    outputWidthPerWarpOptions = [7, 14]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        if repeat[1] == 1:
                            outputChannelPerWarpOptions = [2, 4, 8, 16, 28, 56, 112]
                        elif repeat[1] == 2:
                            outputChannelPerWarpOptions = [2, 4, 8, 28, 56]
                        elif repeat[1] == 4:
                            outputChannelPerWarpOptions = [2, 4, 14, 28]
                        for outputChannelPerWarp in outputChannelPerWarpOptions:
                            if device == "GPU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                                elif outputChannelPerWarp == 28:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 56:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 112:
                                    channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif device == "DCU":
                                if outputChannelPerWarp == 2:
                                    channelGroupSizeOptions = [32]
                                elif outputChannelPerWarp == 4:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 8:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 16:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                                elif outputChannelPerWarp == 28:
                                    channelGroupSizeOptions = [32, 16]
                                elif outputChannelPerWarp == 56:
                                    channelGroupSizeOptions = [32, 16, 8]
                                elif outputChannelPerWarp == 112:
                                    channelGroupSizeOptions = [32, 16, 8, 4]
                            for channelGroupSize in channelGroupSizeOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    """
    input size 7 x 7
    """
    elif inputChannel == 576 and inputWidth == 7 and outputChannel == 160:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 80, 160]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 40, 80]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [32]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 160 and inputWidth == 7 and outputChannel == 960:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [4, 8, 12, 16, 20, 24, 32, 40, 48, 60, 64, 80, 96, 120, 160, 192, 240, 320, 480, 960]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [4, 8, 12, 16, 20, 24, 32, 40, 48, 60, 80, 96, 120, 160, 240, 480]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32,16, 8, 4]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [32,16, 8]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32,16, 8, 4, 2]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32,16, 8]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [32,16, 8, 4]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32,16, 8, 4]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32,16, 8, 4, 2]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 240:
                                channelGroupSizeOptions = [32,16, 8, 4, 2]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 480:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 960:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32,16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32,16, 8]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [32,16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32,16, 8, 4]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32,16]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [32,16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32,16, 8, 4, 2]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32,16, 8]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32,16, 8, 4]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32,16, 8, 4, 2]
                            elif outputChannelPerWarp == 240:
                                channelGroupSizeOptions = [32,16, 8, 4]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 480:
                                channelGroupSizeOptions = [32,16, 8, 4, 2]
                            elif outputChannelPerWarp == 960:
                                channelGroupSizeOptions = [32,16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 960 and inputWidth == 7 and outputChannel == 160:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 80, 160]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 40, 80]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 960 and inputWidth == 7 and outputChannel == 320:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 80, 160]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 320 and inputWidth == 7 and outputChannel == 1280:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 256, 320, 640, 1280]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 128:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 256:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 640:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 1280:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 128:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 256:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 640:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 1280:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 672 and inputWidth == 7 and outputChannel == 192:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 96]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 48:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 96:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 192:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 48:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 96:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 192:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 192 and inputWidth == 7 and outputChannel == 1152:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 36, 48, 64, 72, 96, 128, 144, 192, 288, 384, 576, 1152]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 36, 48, 64, 72, 96, 144, 192, 288, 576]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 36:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 48:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 72:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 96:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 128:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 144:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 192:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 288:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 384:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 576:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 1152:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 36:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 48:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 72:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 96:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 128:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 144:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 192:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 288:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 384:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 576:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 1152:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 1152 and inputWidth == 7 and outputChannel == 192:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 12, 16, 24, 32, 48, 96]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 48:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 96:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 192:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 12:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 24:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 48:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 96:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 192:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    elif inputChannel == 1152 and inputWidth == 7 and outputChannel == 320:
        for blockNumPerSM in blockNumPerSMOptions:
            warpNumPerBlockOptions = [7, 14]
            for warpNumPerBlock in warpNumPerBlockOptions:
                if warpNumPerBlock == 7:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320]
                elif warpNumPerBlock == 14:
                    outputChannelPerWarpOptions = [2, 4, 8, 10, 16, 20, 32, 40, 80, 160]
                for outputChannelPerWarp in outputChannelPerWarpOptions:
                    outputWidthPerWarpOptions = [7]
                    for outputWidthPerWarp in outputWidthPerWarpOptions:
                        # if (inputChannel / (2 * channelGroupSize)) is end with 0.5, then in the kernel, after the for loop, repeat one more time
                        if device == "GPU":
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [32, 16]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [32, 16, 8]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [32, 16, 8, 4]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [32, 16, 8, 4, 2, 1]
                        else:
                            if outputChannelPerWarp == 2:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 4:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 8:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 10:
                                channelGroupSizeOptions = [64, 32]
                            elif outputChannelPerWarp == 16:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 20:
                                channelGroupSizeOptions = [64, 32, 16]
                            elif outputChannelPerWarp == 32:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 40:
                                channelGroupSizeOptions = [64, 32, 16, 8]
                            elif outputChannelPerWarp == 64:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                            elif outputChannelPerWarp == 80:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4]
                            elif outputChannelPerWarp == 160:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2]
                            elif outputChannelPerWarp == 320:
                                channelGroupSizeOptions = [64, 32, 16, 8, 4, 2, 1]
                        for channelGroupSize in channelGroupSizeOptions:
                            # horizontal repeat, vertical repeat
                            if warpNumPerBlock == 7:
                                repeatOptions = [[7, 1]]
                            elif warpNumPerBlock == 14:
                                repeatOptions = [[7, 2]]
                            for repeat in repeatOptions:
                                combinationOptions.append([blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, repeat[0], repeat[1]])

    return combinationOptions

"""
For each possible hardware resource parameter combination,
calculate Streaming Multiprocessor Utilization and Arithmetic Intensity
If return None, then this combination is invalid
If return [SMUtilization, ArithmeticIntensity], then this is valid one
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
def getBestHardwareResourceParameters(device, batchSize, layerConfig):
    # 1. For each layer config, get all possible hardware resource parameters
    # list of [blockNumPerSM, warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, horizontalRepeat, verticalRepeat]
    combinationOptions = getCombinationOptions(device, batchSize, layerConfig)

    # 2. From all possible parameter combinations (include invalid ones) , find those valid ones and 
    #    calculate hardware utilization and arithmetic intensity of them
    validCombinationUtilizationAI = []
    for combinationOption in combinationOptions:
        resultUtilizationAI = calculateUtilizationandAI(device, combinationOption[0], combinationOption[1], combinationOption[2], combinationOption[3], combinationOption[4], batchSize, layerConfig[2], layerConfig[1], layerConfig[1], combinationOption[5], combinationOption[6])
        if resultUtilizationAI != None:
            validCombinationUtilizationAI.append([device, combinationOption[0], combinationOption[1], combinationOption[2], combinationOption[3], combinationOption[4], batchSize, layerConfig[2], layerConfig[1], layerConfig[1], combinationOption[5], combinationOption[6], resultUtilizationAI[0], resultUtilizationAI[1]])

    # 3. From all valid parameter combinations
    utilizationGreterThan1 = []
    utilizationSmallerThan1 = []
    for validComb in validCombinationUtilizationAI:
        if(validComb[12] > 1):
            utilizationGreterThan1.append(validComb)
        else:
            utilizationSmallerThan1.append(validComb)
    #   3a. If all combinations have > 1 utilization, then select the one with smallest utilization
    if len(utilizationSmallerThan1) == 0:
        return sorted(utilizationGreterThan1, key = lambda validComb: validComb[12])[0]
    #   b. If some combination < 1, then select those with utilization close to 1
    #   c. Find the best one with largest AI
    else:
        sortedUtilizationSmallerThan1 = sorted(utilizationSmallerThan1, key = lambda validComb: validComb[12], reverse = True)
        possibleResult = sortedUtilizationSmallerThan1[0 : ]
    
    
def main():
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
    # GPU
    print("For GPU: ")
    for batchSize in batchSizeOptions:
        for layerConfig in layerConfigs:
            bestParameters = getBestHardwareResourceParameters("GPU", batchSize, layerConfig)
            print()
    print("GPU Finished!")
    print("====================================================")

    # DCU
    print("For DCU: ")
    for batchSize in batchSizeOptions:
        for layerConfig in layerConfigs:
            bestParameters = getBestHardwareResourceParameters("DCU", batchSize, layerConfig)
            print()
    print("DCU Finished!")

if __name__=="__main__":
    main()

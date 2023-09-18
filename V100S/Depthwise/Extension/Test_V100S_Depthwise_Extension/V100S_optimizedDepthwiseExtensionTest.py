import torch
from torch import nn
from DepthwiseLayer import OptimizedDepthwiseLayer
from OriginalLayer import OriginalDepthwiseLayer
from torch.profiler import profile, record_function, ProfilerActivity
import time


# To Test one possible parameter combination
def test(inputBatchNumber, inputChannel, inputHeight, inputWidth, filterHeight, stride, loopTime):
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
    outputData = torch.randn(outputBatchNumber, outputChannel, outputHeight, outputWidth).to(cuda_device)

    optimized = OptimizedDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)
    original = OriginalDepthwiseLayer(inputChannel, outputChannel, filterHeight, stride).to(cuda_device)
    
    # Test if the output is correct
    original.conv1.weight.data = optimized.filter.data.clone()
    
    output1 = optimized(inputData)
    output2 = original(inputData)
    
    if torch.allclose(output1, output2, atol=0.0001, rtol=0) is False:
        print("False")

    # Measure performane
    forwardTimeOptimized = 0
    forwardTimeOriginal = 0

    backwardTimeOptimized = 0
    backwardTimeOriginal = 0
    
    for _ in range(loopTime):
        start = time.time()
        output1 = optimized(inputData)
        torch.cuda.synchronize()
        forwardTimeOptimized += time.time() - start
        
        lossOptimized = loss_fn(output1, outputData)
        
        start = time.time()
        lossOptimized.backward()
        torch.cuda.synchronize()
        backwardTimeOptimized += time.time() - start
        
        start = time.time()
        output2 = original(inputData)
        torch.cuda.synchronize()
        forwardTimeOriginal += time.time() - start
        
        lossOriginal = loss_fn(output2, outputData)
        
        start = time.time()
        lossOriginal.backward()
        torch.cuda.synchronize()
        backwardTimeOriginal += time.time() - start
        
    print(f'InputBatchNumber: {inputBatchNumber}, InputChannel: {inputChannel}, InputHeight: {inputHeight}, InputWidth: {inputWidth}, FilterHeight: {filterHeight}, Stride: {stride}')
    print('    Forward optimized: {:.3f} us'.format(forwardTimeOptimized * 1e6 / loopTime))
    print('    Forward original: {:.3f} us'.format(forwardTimeOriginal * 1e6 / loopTime))

    print('    Backward optimized: {:.3f} us'.format(backwardTimeOptimized * 1e6 / loopTime))
    print('    Backward original: {:.3f} us'.format(backwardTimeOriginal * 1e6 / loopTime))

        
# start from here
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")
loss_fn = nn.CrossEntropyLoss()
loop = 1000

# All possible batch numbers
batchNumberOptions = [1, 8, 16, 32, 64, 128]

# All layer structure parameters
# Input Channel, Input Height, Input Width, filterHeight(Width), stride
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
                 
# Test
for parameters in parameterList:
    for batchNumber in batchNumberOptions:
        test(batchNumber, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], loop)

"""
# warm up
# This method is used in the pytorch official tutorial for profiling
optimizedLayer(inputData)
originalLayer(inputData)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_0:
    output0 = optimizedLayer(inputData)
    loss0 = loss_fn(output0, outputRandom)
#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_1:
    loss0.backward()
    
# Test
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof1:
    output1 = optimizedLayer(inputData)
    lossOptimized = loss_fn(output1, outputRandom)
#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof2:
    lossOptimized.backward()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof3:
    output2 = originalLayer(inputData)
    lossOriginal = loss_fn(output2, outputRandom)
#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof4:
    lossOriginal.backward()

print(prof1.key_averages().table(sort_by="self_cpu_time_total"))
#print(prof2.key_averages().table(sort_by="self_cpu_time_total"))
print(prof3.key_averages().table(sort_by="self_cpu_time_total"))
#print(prof4.key_averages().table(sort_by="self_cpu_time_total"))
"""

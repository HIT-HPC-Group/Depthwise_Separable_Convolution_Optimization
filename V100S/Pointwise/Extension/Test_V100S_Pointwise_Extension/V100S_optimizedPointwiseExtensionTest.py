import torch
from torch import nn
from PointwiseLayer import OptimizedPointwiseLayer
from OriginalLayer import OriginalPointwiseLayer
from torch.profiler import profile, record_function, ProfilerActivity
import time

# To Test one possible parameter combination
def test(inputBatchNumber, inputChannel, inputHeight, inputWidth, outputChannel, loopTime):

    # Determine the output size
    outputBatchNumber = inputBatchNumber
    outputHeight = inputHeight
    outputWidth = inputWidth

    # Randomly create input data and output data
    inputData = torch.randn(inputBatchNumber, inputChannel, inputHeight, inputWidth).to(cuda_device)
    outputData = torch.randn(outputBatchNumber, outputChannel, outputHeight, outputWidth).to(cuda_device)

    optimized = OptimizedPointwiseLayer(inputChannel, outputChannel).to(cuda_device)
    original = OriginalPointwiseLayer(inputChannel, outputChannel).to(cuda_device)
    
    # Test if the output is correct
    original.conv1.weight.data = optimized.filter.data.clone()
    
    output1 = optimized(inputData)
    output2 = original(inputData)
    
    if torch.allclose(output1, output2, atol = 0.0001, rtol = 0) is False:
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
        
    print(f'InputBatchNumber: {inputBatchNumber}, InputChannel: {inputChannel}, InputHeight: {inputHeight}, InputWidth: {inputWidth}, OutputChannel: {outputChannel}')
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
                 
# Test
for layerConfig in layerConfigs:
    for batchNumber in batchNumberOptions:
        test(batchNumber, layerConfig[0], layerConfig[1], layerConfig[1], layerConfig[2], loop)

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

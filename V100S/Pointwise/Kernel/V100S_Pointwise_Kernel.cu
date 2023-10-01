#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cudnn.h>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <random>
#include <vector>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

//CSV header and path
#ifdef AMD_PLATFORM
#define CSVPATH "DCU_Pointwise_result.csv"
// vector<string> csvHeader={"Input Batch","Input Channel","Height","Filter Size","Stride","DCU Kernel(ms)","HipDNN(ms)"};
#else
#define CSVPATH "V100S_Pointwise_result.csv"
// vector<string> csvHeader={"Input Batch","Input Channel","Height","Filter Size","Stride","V100S Kernel(ms)","Cudnn(ms)"};
#endif

/*
CUDA and CUDNN Error Handling

checkCuda(err)  - to check if an CUDA API call returned some error.
checkKernel()   - to check if the kernel invocation is failed.
checkCudnn(err) - to check if an CUDNN API call returned some error.
*/
#define checkCuda(err) __checkCuda(err, __FILE__, __LINE__)
#define checkKernel() __checkKernel(__FILE__, __LINE__)
#define checkCudnn(err) __checkCudnn(err, __FILE__, __LINE__)

inline void __checkCuda(cudaError_t err, const char* file, const int line) {
    if (cudaSuccess != err) {
        printf("checkCuda() failed at %s : %i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __checkKernel(const char* file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf("checkKernel() failed at %s : %i : %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __checkCudnn(cudnnStatus_t err, const char* file, const int line) {
    if (CUDNN_STATUS_SUCCESS != err) {
        printf("checkCudnn() failed at %s : %i : %s\n", file, line, cudnnGetErrorString(err));
        exit(-1);
    }
}

/*
Write to csv
*/
void writeCsv(int batchnumber, int channel, int height, int filterheight, int stride, float kerneltime, float cudnntime) {
    fstream fs;
    fs.open(CSVPATH, ios::app);

    if (!fs)
    {
        //create file
        ofstream fout(CSVPATH, ios::app);
        if (fout)
        {
            // fout << csvHeader[0] << ','
            // << csvHeader[1] << ','
            // << csvHeader[2] << ','
            // << csvHeader[3] << ','
            // << csvHeader[4] << ','
            // << csvHeader[5] << ','
            // << csvHeader[6] << std::endl;
            fout << batchnumber << ','
                << channel << ','
                << height << ','
                << filterheight << ','
                << stride << ','
                << kerneltime << ','
                << cudnntime << endl;
            fout.close();
        }
    }
    else
    {
        fs << batchnumber << ','
            << channel << ','
            << height << ','
            << filterheight << ','
            << stride << ','
            << kerneltime << ','
            << cudnntime << endl;
        fs.close();
    }
}

/*
Compare the result calculated by our kernel and that by the cuDNN library.
Use cuDNN library as a reference.
*/
int compareOutput(int n, int c, int h, int w, const float* kernelOutput, const float* cudnnOutput, float delta) {

    // Loop over each element, and compare the value.
    // If the difference is small, then accept, or, reject and return.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < h; k++) {
                for (int l = 0; l < w; l++) {
                    if (abs(kernelOutput[i * c * h * w + j * h * w + k * w + l] - cudnnOutput[i * c * h * w + j * h * w + k * w + l]) > delta) {
                        printf("%f, %f\n", kernelOutput[i * c * h * w + j * h * w + k * w + l], cudnnOutput[i * c * h * w + j * h * w + k * w + l]);
                        printf("Wrong! Output Batch Idx: %d, Channel Idx: %d, Row Idx: %d, Col Idx: %d\n", i, j, k, l);
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

/*
* To get GPU initialization ready
*/
__global__ void warmup() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

/*
******** Super Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel;
Block:
    blockDim.x = inputChannel;

All blocks with the same blockIdx.x are used to handle the same input data and collaborate together to generate a complete output data.
Each block loads a filter that would be used to generate one corresponding output channel. The output channel is decided by the blockIdx.y.
Loop over every (0 - inputHeight, 0 - inputWidth) position with inputChannel channels. Do the pointwise convolution and parallel reduction to get the result of the output.

V100S: 128 576 7 160
Kernel: 36.369358ms
cuDNN: 0.229647ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    // on GPU, use this one:
    int warpSize = 32;

    // on DCU, use this one:
    // int warpSize = 64;

    __shared__ float filterData[576];
    __shared__ float partialResultShared;

    if (threadIdx.x == 0) {
        partialResultShared = 0.0f;
    }

    int inputSize = inputHeight * inputWidth * inputChannel;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputHeight * outputWidth * outputChannel;
    int outputChannelSize = outputHeight * outputWidth;

    // Load filter
    filterData[threadIdx.x] = filter[blockIdx.y * filterInChannel + threadIdx.x];
    __syncthreads();

    int inputStartIdx = blockIdx.x * inputSize + threadIdx.x * inputChannelSize;
    int outputStartIdx = blockIdx.x * outputSize + blockIdx.y * outputChannelSize;
    int loopTime = inputChannelSize;
    for (int i = 0; i < loopTime; i++) {
        // Pointwise convolution
        float partialResult = input[inputStartIdx + i] * filterData[threadIdx.x];
        __syncthreads();

        // Use parallel reduction to get the intermediate result of a warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            // On GPU, use this one:
            partialResult += __shfl_down_sync(0xFFFFFFFF, partialResult, offset, warpSize);

            // On DCU, use this one:
            // partialResult += __shfl_down(partialResult, offset, warpSize);
        }

        // Use atomic add to sum all intermediate result in warps together to get the partial result for the block
        if (threadIdx.x % warpSize == 0) atomicAdd(&partialResultShared, partialResult);
        __syncthreads();

        // Store output
        if (threadIdx.x == 0) {
            output[outputStartIdx + i] = partialResultShared;
            partialResultShared = 0.0f;
        }
    }
}
*/

/*
******** Super Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = inputBatchNumber;
Block:
    blockDim.x = 7 * 7 * 16;

Each block handles one input data. (128 blocks in total.)
Each time, a block loads 16 channels of its input data, for this chunk of data:
    the block loops over the filter data (160 * 576). In each filter out channel (1 * 576), find the corresponding filter in channel (16).
    Each thread calculates the partial result and stores the partial result in a 7 * 7 * 16 output buffer. Then do the parallel reduction on the buffer to get the result for 1 output channel.
    Use the first 49 (outputHeight * outputWidth) threads to write one channel of the output data.

V100S: 128 576 7 160
Kernel: 6.475264 ms.
cuDNN: 0.198624 ms.
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputData[7 * 7 * 16];
    __shared__ float filterData[16];
    __shared__ float outputData[7 * 7 * 16];

    int channelGroup = 16;
    int inputSize = inputHeight * inputWidth * inputChannel;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputHeight * outputWidth * outputChannel;
    int outputChannelSize = outputHeight * outputWidth;

    for (int i = 0; i < inputChannel / 16; i++) {
        // Load input
        inputData[threadIdx.x] = input[blockIdx.x * inputSize + i * channelGroup * inputChannelSize + threadIdx.x];
        __syncthreads();

        // Pointwise convolution
        for (int j = 0; j < filterOutChannel; j++) {
            // load filter
            if (threadIdx.x < channelGroup) {
                filterData[threadIdx.x] = filter[i * channelGroup + j * filterInChannel + threadIdx.x];
            }
            __syncthreads();

            // Calculate the intermediate result
            int filterAccessIdx = threadIdx.x / inputChannelSize;
            outputData[threadIdx.x] = inputData[threadIdx.x] * filterData[filterAccessIdx];
            __syncthreads();

            // Parallel reduction to get one channel of output
            for (int offset = channelGroup / 2; offset > 0; offset /= 2) {
                if (threadIdx.x < outputChannelSize * offset) {
                    outputData[threadIdx.x] += outputData[threadIdx.x + offset * outputChannelSize];
                }
                __syncthreads();
            }

            // Store output
            if (threadIdx.x < outputChannelSize) {
                int outputAccessIdx = blockIdx.x * outputSize + j * outputChannelSize + (threadIdx.x / outputHeight) * outputWidth + threadIdx.x % outputHeight;
                output[outputAccessIdx] += outputData[threadIdx.x];
            }
        }
    }
}
*/

/*
******** Super Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = inputBatchNumber;
    gridDim.y = inputChannel / 16;
Block:
    blockDim.x = 7 * 7 * 16;

Each block handles 7 * 7 * 16 input elements. For a input with 7 * 7 * 576 elements, it needs 36 blocks.
For each output channel, each block also loads 16 corresponding filter elements and generate a partial result and store it in an intermediate output buffer.
Then do the parallel reduction on the buffer to generate partial result of one output channel.
All of the 36 blocks need to add their partial output result together to get a complete result for 1 output channel. This is done by the AtomicAdd operation.

V100S: 128 576 7 160
Kernel: 4.828096 ms.
cuDNN: 0.196608 ms.
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputData[7 * 7 * 16];
    __shared__ float filterData[16];
    __shared__ float outputData[7 * 7 * 16];

    int channelGroup = 16;
    int inputSize = inputHeight * inputWidth * inputChannel;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputHeight * outputWidth * outputChannel;
    int outputChannelSize = outputHeight * outputWidth;

    // Load input
    inputData[threadIdx.x] = input[blockIdx.x * inputSize + blockIdx.y * channelGroup * inputChannelSize + threadIdx.x];
    __syncthreads();

    // Pointwise convolution
    for (int i = 0; i < filterOutChannel; i++) {
        // Load filter
        if (threadIdx.x < channelGroup) {
            filterData[threadIdx.x] = filter[i * filterInChannel + blockIdx.y * channelGroup + threadIdx.x];
        }
        __syncthreads();

        // Calculate the intermediate result
        int filterAccessIdx = threadIdx.x / inputChannelSize;
        outputData[threadIdx.x] = inputData[threadIdx.x] * filterData[filterAccessIdx];
        __syncthreads();

        // Parallel reduction to get one channel of output
        for (int offset = channelGroup / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < outputChannelSize * offset) {
                outputData[threadIdx.x] += outputData[threadIdx.x + outputChannelSize * offset];
            }
            __syncthreads();
        }

        // Store result
        if (threadIdx.x < outputChannelSize) {
            int outputAccessIdx = blockIdx.x * outputSize + i * outputChannelSize + (threadIdx.x / outputHeight) * outputWidth + threadIdx.x % outputHeight;
            atomicAdd(&output[outputAccessIdx], outputData[threadIdx.x]);
        }
    }
}
*/

/*
******** Naive Implementation 1. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 4;

Each block produces 7 x 7 x 16 output data in 4 cycles.
In each time, all threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 128 576 7 160
Kernel: 0.964261ms
cuDNN: 0.163683ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputGroupSize = channelGroup * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    // Each block needs to repeat channelGroup / blockDim.z times to generate a complete output
    int loopTime = channelGroup / blockDim.z;
    for (int i = 0; i < loopTime; i++) {
        int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + blockSize * i + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;
        float partialResult = 0.0f;
        // Pointwise convolution
        for (int j = 0; j < filterInChannel; j++) {
            int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
            int filterAccessIdx = (blockIdx.y * channelGroup + i * blockDim.z + threadIdx.z) * filterInChannel + j;
            partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
        }
        // Store output
        output[outputIdx] = partialResult;
    }
}
*/

/*
******** Naive Implementation 2 ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 128 576 7 160
Kernel: 0.765216ms
cuDNN: 0.227296ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}
*/

/*
******** Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

All blocks with the same blockIdx.x handles the same input data. An input data is separated to blocks containing 7 * 7 * 16 elements.
At each time, blocks with different blockIdx.y load different blocks of the input data and use different parts of the filter to do the pointwise convolution.
After looping through all blocks of the input data, store output back.

V100S: 128 576 7 160
Kernel: 1.072372 ms
cuDNN: 0.222593 ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputData[7 * 7 * 16];

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int inputGroupSize = channelGroup * inputHeight * inputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;
    int inputLoadBaseIdx = blockIdx.x * inputSize + blockIdx.y * inputGroupSize + threadIdx.z * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
    int inputLimit = (blockIdx.x + 1) * inputSize - 1;

    // each thread needs to find its filter start position
    int outputFilterStartIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel;

    // Pointwise Convolution
    int loadLoopTime = filterInChannel / channelGroup;
    float partialResult = 0.0;
    for (int i = 0; i < loadLoopTime; i++) {
        // different thread blocks process different input blocks at each time.
        // load input
        int inputLoadIdx = inputLoadBaseIdx + i * channelGroup * inputChannelSize;
        if (inputLoadIdx > inputLimit) {
            inputLoadIdx -= inputSize;
        }
        inputData[threadIdx.z * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x] = input[inputLoadIdx];
        __syncthreads();

        int offset = blockIdx.y * channelGroup + i * channelGroup;
        if (offset > filterInChannel - 1) {
            offset -= filterInChannel;
        }

        // Something might be wrong here
        for (int j = 0; j < channelGroup; j++) {
            int inputAccessIdx = j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
            partialResult += inputData[inputAccessIdx] * filter[outputFilterStartIdx + offset + j];
        }
        __syncthreads();
    }
    // Store output
    output[outputIdx] = partialResult;
}
*/

/*
******** Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7 * 7 * 16;

All blocks with the same blockIdx.x handles the same input data. An input data is separated to blocks containing 7 * 7 * 16 elements.
At the beginning, all blocks load the corresponding filter channels into shared memory.
Then loop through all blocks of the input data.
At each time, blocks with different blockIdx.y load different blocks of the input data and use different parts of the filter to do the pointwise convolution.
After looping through all blocks of the input data, store output back.

V100S: 128 576 7 160
Kernel: 1.007104 ms
cuDNN: 0.199712 ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputData[7 * 7 * 16];
    __shared__ float filterData[16 * 576];

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int inputGroupSize = channelGroup * inputHeight * inputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    // Load filter
    int filterLoadBaseIdx = blockIdx.y * channelGroup * filterInChannel;
    for (int i = threadIdx.z * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x; i < 16 * 576; i += 7 * 7 * 16) {
        filterData[i] = filter[filterLoadBaseIdx + i];
    }
    __syncthreads();

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;
    int inputLoadBaseIdx = blockIdx.x * inputSize + blockIdx.y * inputGroupSize + threadIdx.z * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
    int inputLimit = (blockIdx.x + 1) * inputSize - 1;

    int outputFilterStartIdx = threadIdx.z * filterInChannel;

    int loadLoopTime = filterInChannel / channelGroup;
    float partialResult = 0.0f;
    for (int i = 0; i < loadLoopTime; i++) {
        // Different thread blocks process different input blocks at each time.
        // Load input
        int inputLoadIdx = inputLoadBaseIdx + i * channelGroup * inputChannelSize;
        if(inputLoadIdx > inputLimit) {
            inputLoadIdx -= inputSize;
        }
        inputData[threadIdx.z * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x] = input[inputLoadIdx];
        __syncthreads();

        // To find the correct filter index
        int offset = (blockIdx.y + i) * channelGroup;
        if(offset > filterInChannel - 1){
            offset -= filterInChannel;
        }

        // Pointwise convolution
        for (int j = 0; j < channelGroup; j++) {
            int inputAccessIdx = j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
            partialResult += inputData[inputAccessIdx] * filterData[outputFilterStartIdx + offset + j];
        }
        __syncthreads();
    }

    // Store output
    output[outputIdx] = partialResult;
}
*/

/*
******** Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputHeight * outputWidth;
Block:
    blockDim.x = 32 * 5;

Use registers to cache input data for the warps.
Each block is responsible for one column of the output data. 49 blocks, each block has 160 threads (5 warps)
For each column of the input data (576 elements), distribute this column onto warps. Each threads contains 576 / 32 = 18 elements.
Then use shuffle sync to communicate and exchange data between threads in the same warp. Each thread also needs to find its own filter value index.
In the end, write the result back to output.

V100S: 128 576 7 160
Kernel: 6.809956 ms
cuDNN: 0.197632 ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int warpSize = 32;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    float rc[18];
    // Id of thread in the warp.
    int localId = threadIdx.x % warpSize;
    int outputIdx = blockIdx.x * outputSize + (blockIdx.y / outputWidth) * outputWidth + blockIdx.y % outputWidth + threadIdx.x * outputChannelSize;
    int filterAccessStart = threadIdx.x * filterInChannel;

    // Fetching into register cache.
    int inputLoadStartIdx = blockIdx.x * inputSize + (blockIdx.y / inputWidth) * inputWidth + blockIdx.y % inputWidth + localId * inputChannelSize;
    for (int i = 0; i < inputChannel / warpSize; i++) {
        rc[i] = input[inputLoadStartIdx + i * inputChannelSize * warpSize];
    }
    __syncthreads();
    // Pointwise convolution
    float partialResult = 0.0f;
    for (int i = 0; i < filterInChannel; i++) {
        float currInputVal = __shfl_sync(0xFFFFFFFF, rc[i / warpSize], i % warpSize);
        float currFilterVal = filter[filterAccessStart + i];
        partialResult += currInputVal * currFilterVal;
    }

    // Store output
    output[outputIdx] = partialResult;
}
*/

/*
******** Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputHeight * outputWidth;
Block:
    blockDim.x = 32 * 5;

Use registers to cache input data for the warps.
Each block is responsible for one column of the output data. 49 blocks, each block has 160 threads (5 warps)
For each column of the input data (576 elements), separate to 3 segments, and in each time, distribute a segment onto warps. Each threads contains 576/ 3 / 32 = 6 elements.
Then use shuffle sync to communicate and exchange data between threads in the same warp. Each thread also needs to find its own filter value index.
In the end, write the result back to output.

V100S: 128 576 7 160
Kernel: 6.706784 ms
cuDNN: 0.224256 ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int warpSize = 32;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    float rc[6];
    // Id of thread in the warp.
    int localId = threadIdx.x % warpSize;
    int outputIdx = blockIdx.x * outputSize + (blockIdx.y / outputWidth) * outputWidth + blockIdx.y % outputWidth + threadIdx.x * outputChannelSize;
    int filterAccessStart = threadIdx.x * filterInChannel;
    float partialResult = 0.0f;
    for (int i = 0; i < 3; i++) {
        // Fetching into register cache.
        int inputLoadStartIdx = blockIdx.x * inputSize + (blockIdx.y / inputWidth) * inputWidth + blockIdx.y % inputWidth + localId * inputChannelSize + i * (inputChannel / 3) * inputChannelSize;
        for (int j = 0; j < inputChannel / 3 / warpSize; j++) {
            rc[j] = input[inputLoadStartIdx + j * inputChannelSize * warpSize];
        }
        __syncthreads();
        // Pointwise convolution
        for (int j = 0; j < filterInChannel / 3; j++) {
            float currInputVal = __shfl_sync(0xFFFFFFFF, rc[j / warpSize], j % warpSize);
            float currFilterVal = filter[filterAccessStart + i * inputChannel / 3 + j];
            partialResult += currInputVal * currFilterVal;
        }
    }

    // Store output
    output[outputIdx] = partialResult;
}
*/

/*
******** Slow. DO NOT USE THIS KERNEL FOR FINAL SUBMISSION ********

Pointwise Convolution Kernel
Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputHeight;
Block:
    blockDim.x = 32 * 5;

Use registers to cache input data for the warps.
Each block is responsible for one column of the output data. 49 blocks, each block has 160 threads (5 warps)
For each column of the input data (576 elements), separate to 3 segments, and in each time, distribute a segment onto warps. Each threads contains 576/ 3 / 32 = 6 elements.
Then use shuffle sync to communicate and exchange data between threads in the same warp. Each thread also needs to find its own filter value index.
In the end, write the result back to output.

V100S: 128 576 7 160
Kernel: 7.013440 ms
cuDNN: 0.196608 ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void Input_7x7_InChannel_576_OutChannel_160
__global__ void Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int warpSize = 32;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    float rc[18];
    // Id of thread in the warp.
    int localId = threadIdx.x % warpSize;
    int filterAccessStart = threadIdx.x * filterInChannel;
    int outputStartIdx = blockIdx.x * outputSize + blockIdx.y * outputWidth + threadIdx.x * outputChannelSize;
    float partialResult = 0.0f;
    for (int i = 0; i < 7; i++) {
        // Fetching into register cache.
        int inputLoadStartIdx = blockIdx.x * inputSize + blockIdx.y * inputWidth + i + localId * inputChannelSize;
        for (int j = 0; j < inputChannel / warpSize; j++) {
            rc[j] = input[inputLoadStartIdx + j * inputChannelSize * warpSize];
        }
        __syncthreads();
        // Pointwise convolution
        for (int j = 0; j < filterInChannel; j++) {
            float currInputVal = __shfl_sync(0xFFFFFFFF, rc[j / warpSize], j % warpSize);
            float currFilterVal = filter[filterAccessStart + j];
            partialResult += currInputVal * currFilterVal;
        }

        // Store output
        output[outputStartIdx + i] = partialResult;
        partialResult = 0.0f;
    }
}
*/

// ===========================================================================
// Input Size 7 x 7, Input Channel 576, Output Channel 160
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_576_OutChannel_160

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 576 7 160
Kernel: 0.056416 ms
cuDNN: 0.090784 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_576_OutChannel_160
__global__ void InputBatch_1_Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_576_OutChannel_160

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 576 7 160
Kernel: 0.074176 ms
cuDNN:  0.110368 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_576_OutChannel_160
__global__ void InputBatch_8_Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_16_Input_7x7_InChannel_576_OutChannel_160

Grid:
    gridDim.x = ;
    gridDim.y = ;
Block:
    blockDim.x = ;
    blockDim.y = ;
    blockDim.z = ;

V100S: 16 576 7 160
Kernel:  ms
cuDNN:   ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_16_Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_16_Input_7x7_InChannel_576_OutChannel_160
__global__ void InputBatch_16_Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

}

/*
Pointwise Convolution Kernel
InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v1.

First Correct Large Kernel Based on The Paper
Grid:
    gridDim.x = (128 * 160 * 7 * 7) / (7 * 7 * 80);
Block:
    blockDim.x = 32 * 7;

WarpH = 7
WarpW = 80
Cnum = 8

One thread block contains 7 warps, 7 * 32 = 224 threads. 
Each thread block is responsible for generating 7 * 7 * 80 output data.
Each warp is responsible for generating 7 * 80 output data.

V100S: 128 576 7 160
Kernel: 0.737856 ms
cuDNN: 0.306400 ms
*/

__global__ void InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v1(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {
    
    // warpNum(7) warps in total, each warp uses warpH (7) * Cnum (8) input data each time
    __shared__ float inputSharedBuffer1[7 * 7 * 8];
    __shared__ float inputSharedBuffer2[7 * 7 * 8];

    // each block generates WarpW (80) output channels. every time, a block uses Cnum (8) channels in a filter
    __shared__ float filterSharedBuffer1[8 * 80];
    __shared__ float filterSharedBuffer2[8 * 80];

    // to hold loaded operands temp.
    // number of input temp = warpH (7)
    // number of filter temp = WarpW / (warpSize / Cnum) = 80 / (32 / 8)
    float inputTemp1 = 0, inputTemp2 = 0, inputTemp3 = 0, inputTemp4 = 0, inputTemp5 = 0, inputTemp6 = 0, inputTemp7 = 0;
    float filterTemp1 = 0, filterTemp2 = 0, filterTemp3 = 0, filterTemp4 = 0, filterTemp5 = 0;
    float filterTemp6 = 0, filterTemp7 = 0, filterTemp8 = 0, filterTemp9 = 0, filterTemp10 = 0;
    float filterTemp11 = 0, filterTemp12 = 0, filterTemp13 = 0, filterTemp14 = 0, filterTemp15 = 0;
    float filterTemp16 = 0, filterTemp17 = 0, filterTemp18 = 0, filterTemp19 = 0, filterTemp20 = 0;

    // to hold operands
    // same number as temp registers
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0, inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0, filterOperand7 = 0, filterOperand8 = 0, filterOperand9 = 0, filterOperand10 = 0;
    float filterOperand11 = 0, filterOperand12 = 0, filterOperand13 = 0, filterOperand14 = 0, filterOperand15 = 0;
    float filterOperand16 = 0, filterOperand17 = 0, filterOperand18 = 0, filterOperand19 = 0, filterOperand20 = 0;

    // to hold intermediate result
    // number of input temp * number of filter temp = 7 * 20
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0; 
    float input1filter6 = 0, input1filter7 = 0, input1filter8 = 0, input1filter9 = 0, input1filter10 = 0;
    float input1filter11 = 0, input1filter12 = 0, input1filter13 = 0, input1filter14 = 0, input1filter15 = 0;
    float input1filter16 = 0, input1filter17 = 0, input1filter18 = 0, input1filter19 = 0, input1filter20 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0, input2filter7 = 0, input2filter8 = 0, input2filter9 = 0, input2filter10 = 0;
    float input2filter11 = 0, input2filter12 = 0, input2filter13 = 0, input2filter14 = 0, input2filter15 = 0;
    float input2filter16 = 0, input2filter17 = 0, input2filter18 = 0, input2filter19 = 0, input2filter20 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0, input3filter7 = 0, input3filter8 = 0, input3filter9 = 0, input3filter10 = 0;
    float input3filter11 = 0, input3filter12 = 0, input3filter13 = 0, input3filter14 = 0, input3filter15 = 0;
    float input3filter16 = 0, input3filter17 = 0, input3filter18 = 0, input3filter19 = 0, input3filter20 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0, input4filter7 = 0, input4filter8 = 0, input4filter9 = 0, input4filter10 = 0;
    float input4filter11 = 0, input4filter12 = 0, input4filter13 = 0, input4filter14 = 0, input4filter15 = 0;
    float input4filter16 = 0, input4filter17 = 0, input4filter18 = 0, input4filter19 = 0, input4filter20 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0, input5filter4 = 0, input5filter5 = 0;
    float input5filter6 = 0, input5filter7 = 0, input5filter8 = 0, input5filter9 = 0, input5filter10 = 0;
    float input5filter11 = 0, input5filter12 = 0, input5filter13 = 0, input5filter14 = 0, input5filter15 = 0;
    float input5filter16 = 0, input5filter17 = 0, input5filter18 = 0, input5filter19 = 0, input5filter20 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0, input6filter4 = 0, input6filter5 = 0;
    float input6filter6 = 0, input6filter7 = 0, input6filter8 = 0, input6filter9 = 0, input6filter10 = 0;
    float input6filter11 = 0, input6filter12 = 0, input6filter13 = 0, input6filter14 = 0, input6filter15 = 0;
    float input6filter16 = 0, input6filter17 = 0, input6filter18 = 0, input6filter19 = 0, input6filter20 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0, input7filter4 = 0, input7filter5 = 0;
    float input7filter6 = 0, input7filter7 = 0, input7filter8 = 0, input7filter9 = 0, input7filter10 = 0;
    float input7filter11 = 0, input7filter12 = 0, input7filter13 = 0, input7filter14 = 0, input7filter15 = 0;
    float input7filter16 = 0, input7filter17 = 0, input7filter18 = 0, input7filter19 = 0, input7filter20 = 0;

    int warpID = threadIdx.x / 32;    // each block contains 7 warps, warpID 0 - 6
    int laneID = threadIdx.x % 32;    // each warp contains 32 threads, laneID 0 - 31

    // load Cnum (8) channels of data from input (7 * 7 * 8) and filter (80 * 8), and store into shared buffer 1
    // input
    int blockLoadInputStartIdx = (blockIdx.x / 2) * inputChannel * inputHeight * inputWidth;
    inputSharedBuffer1[threadIdx.x + 32 * 7 * 0] = input[blockLoadInputStartIdx + threadIdx.x + 32 * 7 * 0];
    if(threadIdx.x < (7 * 7 * 8 - 32 * 7)) {
        inputSharedBuffer1[threadIdx.x + 32 * 7 * 1] = input[blockLoadInputStartIdx + threadIdx.x + 32 * 7 * 1];
    }
    
    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 2) * inputChannel * (outputChannel / 2);
    filterSharedBuffer1[threadIdx.x + 32 * 7 * 0] = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * 7 * inputChannel * 0];    // 28 output channels
    filterSharedBuffer1[threadIdx.x + 32 * 7 * 1] = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * 7 * inputChannel * 1];    // 56 output channels
    if(threadIdx.x < (8 * 80 - 32 * 7 * 2)) {
        filterSharedBuffer1[threadIdx.x + 32 * 7 * 2] = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * 7 * inputChannel * 2]; // last 24 output channels
    }

    __syncthreads();
    
    // For loop begins
    for(int i = 0; i < inputChannel / (2 * 8); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 8;
        inputTemp1 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 0];
        inputTemp2 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 1];
        inputTemp3 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 2];
        inputTemp4 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 3];
        inputTemp5 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 4];
        inputTemp6 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 5];
        inputTemp7 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 6];
        
        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 0];
        filterTemp2 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 1];
        filterTemp3 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 2];
        filterTemp4 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 3];
        filterTemp5 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 4];
        filterTemp6 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 5];
        filterTemp7 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 6];
        filterTemp8 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 7];
        filterTemp9 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 8];
        filterTemp10 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 9];

        filterTemp11 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 10];
        filterTemp12 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 11];
        filterTemp13 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 12];
        filterTemp14 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 13];
        filterTemp15 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 14];
        filterTemp16 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 15];
        filterTemp17 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 16];
        filterTemp18 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 17];
        filterTemp19 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 18];
        filterTemp20 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 19];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 4];
        inputOperand6 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 0];
        filterOperand2 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 1];
        filterOperand3 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 2];
        filterOperand4 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 3];
        filterOperand5 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 4];

        filterOperand6 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 5];
        filterOperand7 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 6];
        filterOperand8 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 7];
        filterOperand9 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 8];
        filterOperand10 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 9];

        filterOperand11 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 10];
        filterOperand12 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 11];
        filterOperand13 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 12];
        filterOperand14 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 13];
        filterOperand15 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 14];

        filterOperand16 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 15];
        filterOperand17 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 16];
        filterOperand18 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 17];
        filterOperand19 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 18];
        filterOperand20 = filterSharedBuffer1[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 19];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;
        input1filter10 += inputOperand1 * filterOperand10;

        input1filter11 += inputOperand1 * filterOperand11; 
        input1filter12 += inputOperand1 * filterOperand12;
        input1filter13 += inputOperand1 * filterOperand13;
        input1filter14 += inputOperand1 * filterOperand14;
        input1filter15 += inputOperand1 * filterOperand15;

        input1filter16 += inputOperand1 * filterOperand16;
        input1filter17 += inputOperand1 * filterOperand17;
        input1filter18 += inputOperand1 * filterOperand18;
        input1filter19 += inputOperand1 * filterOperand19;
        input1filter20 += inputOperand1 * filterOperand20;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input2filter11 += inputOperand2 * filterOperand11; 
        input2filter12 += inputOperand2 * filterOperand12;
        input2filter13 += inputOperand2 * filterOperand13;
        input2filter14 += inputOperand2 * filterOperand14;
        input2filter15 += inputOperand2 * filterOperand15;

        input2filter16 += inputOperand2 * filterOperand16;
        input2filter17 += inputOperand2 * filterOperand17;
        input2filter18 += inputOperand2 * filterOperand18;
        input2filter19 += inputOperand2 * filterOperand19;
        input2filter20 += inputOperand2 * filterOperand20;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input3filter11 += inputOperand3 * filterOperand11; 
        input3filter12 += inputOperand3 * filterOperand12;
        input3filter13 += inputOperand3 * filterOperand13;
        input3filter14 += inputOperand3 * filterOperand14;
        input3filter15 += inputOperand3 * filterOperand15;

        input3filter16 += inputOperand3 * filterOperand16;
        input3filter17 += inputOperand3 * filterOperand17;
        input3filter18 += inputOperand3 * filterOperand18;
        input3filter19 += inputOperand3 * filterOperand19;
        input3filter20 += inputOperand3 * filterOperand20;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input4filter11 += inputOperand4 * filterOperand11; 
        input4filter12 += inputOperand4 * filterOperand12;
        input4filter13 += inputOperand4 * filterOperand13;
        input4filter14 += inputOperand4 * filterOperand14;
        input4filter15 += inputOperand4 * filterOperand15;

        input4filter16 += inputOperand4 * filterOperand16;
        input4filter17 += inputOperand4 * filterOperand17;
        input4filter18 += inputOperand4 * filterOperand18;
        input4filter19 += inputOperand4 * filterOperand19;
        input4filter20 += inputOperand4 * filterOperand20;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input5filter11 += inputOperand5 * filterOperand11; 
        input5filter12 += inputOperand5 * filterOperand12;
        input5filter13 += inputOperand5 * filterOperand13;
        input5filter14 += inputOperand5 * filterOperand14;
        input5filter15 += inputOperand5 * filterOperand15;

        input5filter16 += inputOperand5 * filterOperand16;
        input5filter17 += inputOperand5 * filterOperand17;
        input5filter18 += inputOperand5 * filterOperand18;
        input5filter19 += inputOperand5 * filterOperand19;
        input5filter20 += inputOperand5 * filterOperand20;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input6filter11 += inputOperand6 * filterOperand11; 
        input6filter12 += inputOperand6 * filterOperand12;
        input6filter13 += inputOperand6 * filterOperand13;
        input6filter14 += inputOperand6 * filterOperand14;
        input6filter15 += inputOperand6 * filterOperand15;

        input6filter16 += inputOperand6 * filterOperand16;
        input6filter17 += inputOperand6 * filterOperand17;
        input6filter18 += inputOperand6 * filterOperand18;
        input6filter19 += inputOperand6 * filterOperand19;
        input6filter20 += inputOperand6 * filterOperand20;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        input7filter11 += inputOperand7 * filterOperand11; 
        input7filter12 += inputOperand7 * filterOperand12;
        input7filter13 += inputOperand7 * filterOperand13;
        input7filter14 += inputOperand7 * filterOperand14;
        input7filter15 += inputOperand7 * filterOperand15;

        input7filter16 += inputOperand7 * filterOperand16;
        input7filter17 += inputOperand7 * filterOperand17;
        input7filter18 += inputOperand7 * filterOperand18;
        input7filter19 += inputOperand7 * filterOperand19;
        input7filter20 += inputOperand7 * filterOperand20;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x % 8 < 8){
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 0] = inputTemp1;
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 1] = inputTemp2;
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 2] = inputTemp3;
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 3] = inputTemp4;
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 4] = inputTemp5;
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 5] = inputTemp6;
            inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 6] = inputTemp7;
        }

        if(threadIdx.x < 32) {
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 0] = filterTemp1;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 1] = filterTemp2;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 2] = filterTemp3;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 3] = filterTemp4;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 4] = filterTemp5;

            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 5] = filterTemp6;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 6] = filterTemp7;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 7] = filterTemp8;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 8] = filterTemp9;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 9] = filterTemp10;

            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 10] = filterTemp11;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 11] = filterTemp12;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 12] = filterTemp13;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 13] = filterTemp14;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 14] = filterTemp15;

            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 15] = filterTemp16;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 16] = filterTemp17;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 17] = filterTemp18;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 18] = filterTemp19;
            filterSharedBuffer2[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 19] = filterTemp20;
        }

        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 8;
        inputTemp1 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 0];
        inputTemp2 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 1];
        inputTemp3 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 2];
        inputTemp4 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 3];
        inputTemp5 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 4];
        inputTemp6 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 5];
        inputTemp7 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 8) * 7 * 7 + 6];
        
        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 0];
        filterTemp2 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 1];
        filterTemp3 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 2];
        filterTemp4 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 3];
        filterTemp5 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 4];
        filterTemp6 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 5];
        filterTemp7 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 6];
        filterTemp8 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 7];
        filterTemp9 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 8];
        filterTemp10 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 9];

        filterTemp11 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 10];
        filterTemp12 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 11];
        filterTemp13 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 12];
        filterTemp14 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 13];
        filterTemp15 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 14];
        filterTemp16 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 15];
        filterTemp17 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 16];
        filterTemp18 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 17];
        filterTemp19 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 18];
        filterTemp20 = filter[blockLoadFilterStartIdx + (threadIdx.x / 8) * inputChannel + (threadIdx.x % 8) + (32 / 8) * inputChannel * 19];

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 4];
        inputOperand6 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer2[warpID * 7 + (laneID % 8) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 0];
        filterOperand2 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 1];
        filterOperand3 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 2];
        filterOperand4 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 3];
        filterOperand5 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 4];

        filterOperand6 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 5];
        filterOperand7 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 6];
        filterOperand8 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 7];
        filterOperand9 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 8];
        filterOperand10 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 9];

        filterOperand11 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 10];
        filterOperand12 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 11];
        filterOperand13 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 12];
        filterOperand14 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 13];
        filterOperand15 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 14];

        filterOperand16 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 15];
        filterOperand17 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 16];
        filterOperand18 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 17];
        filterOperand19 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 18];
        filterOperand20 = filterSharedBuffer2[(laneID / 8) * 8 + laneID % 8 + 4 * 8 * 19];  

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;
        input1filter10 += inputOperand1 * filterOperand10;

        input1filter11 += inputOperand1 * filterOperand11; 
        input1filter12 += inputOperand1 * filterOperand12;
        input1filter13 += inputOperand1 * filterOperand13;
        input1filter14 += inputOperand1 * filterOperand14;
        input1filter15 += inputOperand1 * filterOperand15;

        input1filter16 += inputOperand1 * filterOperand16;
        input1filter17 += inputOperand1 * filterOperand17;
        input1filter18 += inputOperand1 * filterOperand18;
        input1filter19 += inputOperand1 * filterOperand19;
        input1filter20 += inputOperand1 * filterOperand20;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input2filter11 += inputOperand2 * filterOperand11; 
        input2filter12 += inputOperand2 * filterOperand12;
        input2filter13 += inputOperand2 * filterOperand13;
        input2filter14 += inputOperand2 * filterOperand14;
        input2filter15 += inputOperand2 * filterOperand15;

        input2filter16 += inputOperand2 * filterOperand16;
        input2filter17 += inputOperand2 * filterOperand17;
        input2filter18 += inputOperand2 * filterOperand18;
        input2filter19 += inputOperand2 * filterOperand19;
        input2filter20 += inputOperand2 * filterOperand20;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input3filter11 += inputOperand3 * filterOperand11; 
        input3filter12 += inputOperand3 * filterOperand12;
        input3filter13 += inputOperand3 * filterOperand13;
        input3filter14 += inputOperand3 * filterOperand14;
        input3filter15 += inputOperand3 * filterOperand15;

        input3filter16 += inputOperand3 * filterOperand16;
        input3filter17 += inputOperand3 * filterOperand17;
        input3filter18 += inputOperand3 * filterOperand18;
        input3filter19 += inputOperand3 * filterOperand19;
        input3filter20 += inputOperand3 * filterOperand20;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input4filter11 += inputOperand4 * filterOperand11; 
        input4filter12 += inputOperand4 * filterOperand12;
        input4filter13 += inputOperand4 * filterOperand13;
        input4filter14 += inputOperand4 * filterOperand14;
        input4filter15 += inputOperand4 * filterOperand15;

        input4filter16 += inputOperand4 * filterOperand16;
        input4filter17 += inputOperand4 * filterOperand17;
        input4filter18 += inputOperand4 * filterOperand18;
        input4filter19 += inputOperand4 * filterOperand19;
        input4filter20 += inputOperand4 * filterOperand20;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input5filter11 += inputOperand5 * filterOperand11; 
        input5filter12 += inputOperand5 * filterOperand12;
        input5filter13 += inputOperand5 * filterOperand13;
        input5filter14 += inputOperand5 * filterOperand14;
        input5filter15 += inputOperand5 * filterOperand15;

        input5filter16 += inputOperand5 * filterOperand16;
        input5filter17 += inputOperand5 * filterOperand17;
        input5filter18 += inputOperand5 * filterOperand18;
        input5filter19 += inputOperand5 * filterOperand19;
        input5filter20 += inputOperand5 * filterOperand20;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input6filter11 += inputOperand6 * filterOperand11; 
        input6filter12 += inputOperand6 * filterOperand12;
        input6filter13 += inputOperand6 * filterOperand13;
        input6filter14 += inputOperand6 * filterOperand14;
        input6filter15 += inputOperand6 * filterOperand15;

        input6filter16 += inputOperand6 * filterOperand16;
        input6filter17 += inputOperand6 * filterOperand17;
        input6filter18 += inputOperand6 * filterOperand18;
        input6filter19 += inputOperand6 * filterOperand19;
        input6filter20 += inputOperand6 * filterOperand20;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        input7filter11 += inputOperand7 * filterOperand11; 
        input7filter12 += inputOperand7 * filterOperand12;
        input7filter13 += inputOperand7 * filterOperand13;
        input7filter14 += inputOperand7 * filterOperand14;
        input7filter15 += inputOperand7 * filterOperand15;

        input7filter16 += inputOperand7 * filterOperand16;
        input7filter17 += inputOperand7 * filterOperand17;
        input7filter18 += inputOperand7 * filterOperand18;
        input7filter19 += inputOperand7 * filterOperand19;
        input7filter20 += inputOperand7 * filterOperand20;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x % 8 < 8){
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 0] = inputTemp1;
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 1] = inputTemp2;
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 2] = inputTemp3;
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 3] = inputTemp4;
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 4] = inputTemp5;
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 5] = inputTemp6;
            inputSharedBuffer1[warpID * 7 + (laneID % 8) * 7 * 7 + 6] = inputTemp7;
        }
        if(threadIdx.x < 32) {
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 0] = filterTemp1;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 1] = filterTemp2;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 2] = filterTemp3;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 3] = filterTemp4;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 4] = filterTemp5;

            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 5] = filterTemp6;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 6] = filterTemp7;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 7] = filterTemp8;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 8] = filterTemp9;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 9] = filterTemp10;

            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 10] = filterTemp11;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 11] = filterTemp12;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 12] = filterTemp13;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 13] = filterTemp14;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 14] = filterTemp15;

            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 15] = filterTemp16;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 16] = filterTemp17;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 17] = filterTemp18;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 18] = filterTemp19;
            filterSharedBuffer1[(threadIdx.x / 8) * 8 + threadIdx.x % 8 + 4 * 8 * 19] = filterTemp20;
        }

        __syncthreads();
    }
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads form one group
    #pragma unroll
    for (int offset = (8 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down_sync(0xffffffff, input1filter1, offset, 8);
        input1filter2 += __shfl_down_sync(0xffffffff, input1filter2, offset, 8);
        input1filter3 += __shfl_down_sync(0xffffffff, input1filter3, offset, 8);
        input1filter4 += __shfl_down_sync(0xffffffff, input1filter4, offset, 8);
        input1filter5 += __shfl_down_sync(0xffffffff, input1filter5, offset, 8);

        input1filter6 += __shfl_down_sync(0xffffffff, input1filter6, offset, 8);
        input1filter7 += __shfl_down_sync(0xffffffff, input1filter7, offset, 8);
        input1filter8 += __shfl_down_sync(0xffffffff, input1filter8, offset, 8);
        input1filter9 += __shfl_down_sync(0xffffffff, input1filter9, offset, 8);
        input1filter10 += __shfl_down_sync(0xffffffff, input1filter10, offset, 8);

        input1filter11 += __shfl_down_sync(0xffffffff, input1filter11, offset, 8);
        input1filter12 += __shfl_down_sync(0xffffffff, input1filter12, offset, 8);
        input1filter13 += __shfl_down_sync(0xffffffff, input1filter13, offset, 8);
        input1filter14 += __shfl_down_sync(0xffffffff, input1filter14, offset, 8);
        input1filter15 += __shfl_down_sync(0xffffffff, input1filter15, offset, 8);

        input1filter16 += __shfl_down_sync(0xffffffff, input1filter16, offset, 8);
        input1filter17 += __shfl_down_sync(0xffffffff, input1filter17, offset, 8);
        input1filter18 += __shfl_down_sync(0xffffffff, input1filter18, offset, 8);
        input1filter19 += __shfl_down_sync(0xffffffff, input1filter19, offset, 8);
        input1filter20 += __shfl_down_sync(0xffffffff, input1filter20, offset, 8);

        input2filter1 += __shfl_down_sync(0xffffffff, input2filter1, offset, 8);
        input2filter2 += __shfl_down_sync(0xffffffff, input2filter2, offset, 8);
        input2filter3 += __shfl_down_sync(0xffffffff, input2filter3, offset, 8);
        input2filter4 += __shfl_down_sync(0xffffffff, input2filter4, offset, 8);
        input2filter5 += __shfl_down_sync(0xffffffff, input2filter5, offset, 8);

        input2filter6 += __shfl_down_sync(0xffffffff, input2filter6, offset, 8);
        input2filter7 += __shfl_down_sync(0xffffffff, input2filter7, offset, 8);
        input2filter8 += __shfl_down_sync(0xffffffff, input2filter8, offset, 8);
        input2filter9 += __shfl_down_sync(0xffffffff, input2filter9, offset, 8);
        input2filter10 += __shfl_down_sync(0xffffffff, input2filter10, offset, 8);

        input2filter11 += __shfl_down_sync(0xffffffff, input2filter11, offset, 8);
        input2filter12 += __shfl_down_sync(0xffffffff, input2filter12, offset, 8);
        input2filter13 += __shfl_down_sync(0xffffffff, input2filter13, offset, 8);
        input2filter14 += __shfl_down_sync(0xffffffff, input2filter14, offset, 8);
        input2filter15 += __shfl_down_sync(0xffffffff, input2filter15, offset, 8);

        input2filter16 += __shfl_down_sync(0xffffffff, input2filter16, offset, 8);
        input2filter17 += __shfl_down_sync(0xffffffff, input2filter17, offset, 8);
        input2filter18 += __shfl_down_sync(0xffffffff, input2filter18, offset, 8);
        input2filter19 += __shfl_down_sync(0xffffffff, input2filter19, offset, 8);
        input2filter20 += __shfl_down_sync(0xffffffff, input2filter20, offset, 8);

        input3filter1 += __shfl_down_sync(0xffffffff, input3filter1, offset, 8);
        input3filter2 += __shfl_down_sync(0xffffffff, input3filter2, offset, 8);
        input3filter3 += __shfl_down_sync(0xffffffff, input3filter3, offset, 8);
        input3filter4 += __shfl_down_sync(0xffffffff, input3filter4, offset, 8);
        input3filter5 += __shfl_down_sync(0xffffffff, input3filter5, offset, 8);

        input3filter6 += __shfl_down_sync(0xffffffff, input3filter6, offset, 8);
        input3filter7 += __shfl_down_sync(0xffffffff, input3filter7, offset, 8);
        input3filter8 += __shfl_down_sync(0xffffffff, input3filter8, offset, 8);
        input3filter9 += __shfl_down_sync(0xffffffff, input3filter9, offset, 8);
        input3filter10 += __shfl_down_sync(0xffffffff, input3filter10, offset, 8);

        input3filter11 += __shfl_down_sync(0xffffffff, input3filter11, offset, 8);
        input3filter12 += __shfl_down_sync(0xffffffff, input3filter12, offset, 8);
        input3filter13 += __shfl_down_sync(0xffffffff, input3filter13, offset, 8);
        input3filter14 += __shfl_down_sync(0xffffffff, input3filter14, offset, 8);
        input3filter15 += __shfl_down_sync(0xffffffff, input3filter15, offset, 8);

        input3filter16 += __shfl_down_sync(0xffffffff, input3filter16, offset, 8);
        input3filter17 += __shfl_down_sync(0xffffffff, input3filter17, offset, 8);
        input3filter18 += __shfl_down_sync(0xffffffff, input3filter18, offset, 8);
        input3filter19 += __shfl_down_sync(0xffffffff, input3filter19, offset, 8);
        input3filter20 += __shfl_down_sync(0xffffffff, input3filter20, offset, 8);

        input4filter1 += __shfl_down_sync(0xffffffff, input4filter1, offset, 8);
        input4filter2 += __shfl_down_sync(0xffffffff, input4filter2, offset, 8);
        input4filter3 += __shfl_down_sync(0xffffffff, input4filter3, offset, 8);
        input4filter4 += __shfl_down_sync(0xffffffff, input4filter4, offset, 8);
        input4filter5 += __shfl_down_sync(0xffffffff, input4filter5, offset, 8);

        input4filter6 += __shfl_down_sync(0xffffffff, input4filter6, offset, 8);
        input4filter7 += __shfl_down_sync(0xffffffff, input4filter7, offset, 8);
        input4filter8 += __shfl_down_sync(0xffffffff, input4filter8, offset, 8);
        input4filter9 += __shfl_down_sync(0xffffffff, input4filter9, offset, 8);
        input4filter10 += __shfl_down_sync(0xffffffff, input4filter10, offset, 8);

        input4filter11 += __shfl_down_sync(0xffffffff, input4filter11, offset, 8);
        input4filter12 += __shfl_down_sync(0xffffffff, input4filter12, offset, 8);
        input4filter13 += __shfl_down_sync(0xffffffff, input4filter13, offset, 8);
        input4filter14 += __shfl_down_sync(0xffffffff, input4filter14, offset, 8);
        input4filter15 += __shfl_down_sync(0xffffffff, input4filter15, offset, 8);

        input4filter16 += __shfl_down_sync(0xffffffff, input4filter16, offset, 8);
        input4filter17 += __shfl_down_sync(0xffffffff, input4filter17, offset, 8);
        input4filter18 += __shfl_down_sync(0xffffffff, input4filter18, offset, 8);
        input4filter19 += __shfl_down_sync(0xffffffff, input4filter19, offset, 8);
        input4filter20 += __shfl_down_sync(0xffffffff, input4filter20, offset, 8);

        input5filter1 += __shfl_down_sync(0xffffffff, input5filter1, offset, 8);
        input5filter2 += __shfl_down_sync(0xffffffff, input5filter2, offset, 8);
        input5filter3 += __shfl_down_sync(0xffffffff, input5filter3, offset, 8);
        input5filter4 += __shfl_down_sync(0xffffffff, input5filter4, offset, 8);
        input5filter5 += __shfl_down_sync(0xffffffff, input5filter5, offset, 8);

        input5filter6 += __shfl_down_sync(0xffffffff, input5filter6, offset, 8);
        input5filter7 += __shfl_down_sync(0xffffffff, input5filter7, offset, 8);
        input5filter8 += __shfl_down_sync(0xffffffff, input5filter8, offset, 8);
        input5filter9 += __shfl_down_sync(0xffffffff, input5filter9, offset, 8);
        input5filter10 += __shfl_down_sync(0xffffffff, input5filter10, offset, 8);

        input5filter11 += __shfl_down_sync(0xffffffff, input5filter11, offset, 8);
        input5filter12 += __shfl_down_sync(0xffffffff, input5filter12, offset, 8);
        input5filter13 += __shfl_down_sync(0xffffffff, input5filter13, offset, 8);
        input5filter14 += __shfl_down_sync(0xffffffff, input5filter14, offset, 8);
        input5filter15 += __shfl_down_sync(0xffffffff, input5filter15, offset, 8);

        input5filter16 += __shfl_down_sync(0xffffffff, input5filter16, offset, 8);
        input5filter17 += __shfl_down_sync(0xffffffff, input5filter17, offset, 8);
        input5filter18 += __shfl_down_sync(0xffffffff, input5filter18, offset, 8);
        input5filter19 += __shfl_down_sync(0xffffffff, input5filter19, offset, 8);
        input5filter20 += __shfl_down_sync(0xffffffff, input5filter20, offset, 8);

        input6filter1 += __shfl_down_sync(0xffffffff, input6filter1, offset, 8);
        input6filter2 += __shfl_down_sync(0xffffffff, input6filter2, offset, 8);
        input6filter3 += __shfl_down_sync(0xffffffff, input6filter3, offset, 8);
        input6filter4 += __shfl_down_sync(0xffffffff, input6filter4, offset, 8);
        input6filter5 += __shfl_down_sync(0xffffffff, input6filter5, offset, 8);

        input6filter6 += __shfl_down_sync(0xffffffff, input6filter6, offset, 8);
        input6filter7 += __shfl_down_sync(0xffffffff, input6filter7, offset, 8);
        input6filter8 += __shfl_down_sync(0xffffffff, input6filter8, offset, 8);
        input6filter9 += __shfl_down_sync(0xffffffff, input6filter9, offset, 8);
        input6filter10 += __shfl_down_sync(0xffffffff, input6filter10, offset, 8);

        input6filter11 += __shfl_down_sync(0xffffffff, input6filter11, offset, 8);
        input6filter12 += __shfl_down_sync(0xffffffff, input6filter12, offset, 8);
        input6filter13 += __shfl_down_sync(0xffffffff, input6filter13, offset, 8);
        input6filter14 += __shfl_down_sync(0xffffffff, input6filter14, offset, 8);
        input6filter15 += __shfl_down_sync(0xffffffff, input6filter15, offset, 8);

        input6filter16 += __shfl_down_sync(0xffffffff, input6filter16, offset, 8);
        input6filter17 += __shfl_down_sync(0xffffffff, input6filter17, offset, 8);
        input6filter18 += __shfl_down_sync(0xffffffff, input6filter18, offset, 8);
        input6filter19 += __shfl_down_sync(0xffffffff, input6filter19, offset, 8);
        input6filter20 += __shfl_down_sync(0xffffffff, input6filter20, offset, 8);

        input7filter1 += __shfl_down_sync(0xffffffff, input7filter1, offset, 8);
        input7filter2 += __shfl_down_sync(0xffffffff, input7filter2, offset, 8);
        input7filter3 += __shfl_down_sync(0xffffffff, input7filter3, offset, 8);
        input7filter4 += __shfl_down_sync(0xffffffff, input7filter4, offset, 8);
        input7filter5 += __shfl_down_sync(0xffffffff, input7filter5, offset, 8);

        input7filter6 += __shfl_down_sync(0xffffffff, input7filter6, offset, 8);
        input7filter7 += __shfl_down_sync(0xffffffff, input7filter7, offset, 8);
        input7filter8 += __shfl_down_sync(0xffffffff, input7filter8, offset, 8);
        input7filter9 += __shfl_down_sync(0xffffffff, input7filter9, offset, 8);
        input7filter10 += __shfl_down_sync(0xffffffff, input7filter10, offset, 8);

        input7filter11 += __shfl_down_sync(0xffffffff, input7filter11, offset, 8);
        input7filter12 += __shfl_down_sync(0xffffffff, input7filter12, offset, 8);
        input7filter13 += __shfl_down_sync(0xffffffff, input7filter13, offset, 8);
        input7filter14 += __shfl_down_sync(0xffffffff, input7filter14, offset, 8);
        input7filter15 += __shfl_down_sync(0xffffffff, input7filter15, offset, 8);

        input7filter16 += __shfl_down_sync(0xffffffff, input7filter16, offset, 8);
        input7filter17 += __shfl_down_sync(0xffffffff, input7filter17, offset, 8);
        input7filter18 += __shfl_down_sync(0xffffffff, input7filter18, offset, 8);
        input7filter19 += __shfl_down_sync(0xffffffff, input7filter19, offset, 8);
        input7filter20 += __shfl_down_sync(0xffffffff, input7filter20, offset, 8);
    }

    // Store output
    int blockWriteOutputStartIdx = (blockIdx.x / 2) * outputWidth * outputHeight * outputChannel + (blockIdx.x % 2) * outputWidth * outputHeight * (outputChannel / 2);

    if(laneID % 8 == 0) {
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter2;

        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter3;

        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter4;

        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter5;

        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter6;

        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter7;
        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter7;
        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter7;
        output[blockWriteOutputStartIdx + 6 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter7;

        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter8;
        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter8;
        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter8;
        output[blockWriteOutputStartIdx + 7 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter8;

        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter9;
        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter9;
        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter9;
        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter9;
        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter9;
        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter9;
        output[blockWriteOutputStartIdx + 8 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter9;

        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter10;
        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter10;
        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter10;
        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter10;
        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter10;
        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter10;
        output[blockWriteOutputStartIdx + 9 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter10;

        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter11;
        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter11;
        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter11;
        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter11;
        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter11;
        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter11;
        output[blockWriteOutputStartIdx + 10 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter11;

        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter12;
        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter12;
        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter12;
        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter12;
        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter12;
        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter12;
        output[blockWriteOutputStartIdx + 11 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter12;

        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter13;
        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter13;
        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter13;
        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter13;
        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter13;
        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter13;
        output[blockWriteOutputStartIdx + 12 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter13;

        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter14;
        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter14;
        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter14;
        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter14;
        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter14;
        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter14;
        output[blockWriteOutputStartIdx + 13 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter14;

        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter15;
        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter15;
        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter15;
        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter15;
        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter15;
        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter15;
        output[blockWriteOutputStartIdx + 14 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter15;

        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter16;
        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter16;
        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter16;
        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter16;
        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter16;
        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter16;
        output[blockWriteOutputStartIdx + 15 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter16;

        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter17;
        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter17;
        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter17;
        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter17;
        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter17;
        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter17;
        output[blockWriteOutputStartIdx + 16 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter17;

        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter18;
        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter18;
        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter18;
        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter18;
        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter18;
        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter18;
        output[blockWriteOutputStartIdx + 17 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter18;

        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter19;
        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter19;
        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter19;
        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter19;
        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter19;
        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter19;
        output[blockWriteOutputStartIdx + 18 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter19;

        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter20;
        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter20;
        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter20;
        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter20;
        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter20;
        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter20;
        output[blockWriteOutputStartIdx + 19 * 4 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter20;
    }
}

/*
Pointwise Convolution Kernel
InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v2

Grid:
    gridDim.x = (128 * 160 * 7 * 7) / (7 * 7 * 80);
Block:
    blockDim.x = 32 * 7;

warpNum = 7
WarpH = 7
WarpW = 80
Cnum = 4

One thread block contains 7 warps, 7 * 32 = 224 threads. 
Each thread block is responsible for generating 7 * 7 * 80 output data.
Each warp is responsible for generating 7 * 80 output data.

V100S: 128 576 7 160
Kernel: 0.578496 ms
cuDNN:  0.307392 ms
*/

__global__ void InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v2(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {
    
    // warpNum(7) warps in total, each warp uses warpH (7) * Cnum (8) input data each time
    __shared__ float inputSharedBuffer1[7 * 7 * 4];
    __shared__ float inputSharedBuffer2[7 * 7 * 4];

    // each block generates WarpW (80) output channels. every time, a block uses Cnum (4) channels in a filter
    __shared__ float filterSharedBuffer1[4 * 80];
    __shared__ float filterSharedBuffer2[4 * 80];

    // to hold loaded operands temp.
    // number of input temp = warpH (7)
    // number of filter temp = WarpW / (warpSize / Cnum) = 80 / (32 / 4)
    float inputTemp1 = 0, inputTemp2 = 0, inputTemp3 = 0, inputTemp4 = 0, inputTemp5 = 0, inputTemp6 = 0, inputTemp7 = 0;
    float filterTemp1 = 0, filterTemp2 = 0, filterTemp3 = 0, filterTemp4 = 0, filterTemp5 = 0;
    float filterTemp6 = 0, filterTemp7 = 0, filterTemp8 = 0, filterTemp9 = 0, filterTemp10 = 0;

    // to hold operands
    // same number as temp registers
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0, inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0, filterOperand7 = 0, filterOperand8 = 0, filterOperand9 = 0, filterOperand10 = 0;
    
    // to hold intermediate result
    // number of input temp * number of filter temp = 7 * 20
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0; 
    float input1filter6 = 0, input1filter7 = 0, input1filter8 = 0, input1filter9 = 0, input1filter10 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0, input2filter7 = 0, input2filter8 = 0, input2filter9 = 0, input2filter10 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0, input3filter7 = 0, input3filter8 = 0, input3filter9 = 0, input3filter10 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0, input4filter7 = 0, input4filter8 = 0, input4filter9 = 0, input4filter10 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0, input5filter4 = 0, input5filter5 = 0;
    float input5filter6 = 0, input5filter7 = 0, input5filter8 = 0, input5filter9 = 0, input5filter10 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0, input6filter4 = 0, input6filter5 = 0;
    float input6filter6 = 0, input6filter7 = 0, input6filter8 = 0, input6filter9 = 0, input6filter10 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0, input7filter4 = 0, input7filter5 = 0;
    float input7filter6 = 0, input7filter7 = 0, input7filter8 = 0, input7filter9 = 0, input7filter10 = 0;

    int warpID = threadIdx.x / 32;    // each block contains 7 warps, warpID 0 - 6
    int laneID = threadIdx.x % 32;    // each warp contains 32 threads, laneID 0 - 31

    // load Cnum (4) channels of data from input (7 * 7 * 4) and filter (80 * 4), and store into shared buffer 1
    // input
    int blockLoadInputStartIdx = (blockIdx.x / 2) * inputChannel * inputHeight * inputWidth;
    if(threadIdx.x < 7 * 7 * 4) {
        inputSharedBuffer1[threadIdx.x] = input[blockLoadInputStartIdx + threadIdx.x];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 2) * inputChannel * (outputChannel / 2);
    filterSharedBuffer1[threadIdx.x + 32 * 7 * 0] = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 0];       // 56 channels
    if(threadIdx.x < (4 * 80 - 32 * 7)){
        filterSharedBuffer1[threadIdx.x + 32 * 7 * 1] = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 1];   // 24 channels
    }
    __syncthreads();
    
    // For loop begins
    for(int i = 0; i < inputChannel / (2 * 4); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 4;
        inputTemp1 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 0];
        inputTemp2 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 1];
        inputTemp3 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 2];
        inputTemp4 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 3];
        inputTemp5 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 4];
        inputTemp6 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 5];
        inputTemp7 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 6];
        
        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 0];
        filterTemp2 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 1];
        filterTemp3 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 2];
        filterTemp4 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 3];
        filterTemp5 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 4];
        filterTemp6 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 5];
        filterTemp7 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 6];
        filterTemp8 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 7];
        filterTemp9 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 8];
        filterTemp10 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 9];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 4];
        inputOperand6 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 0];
        filterOperand2 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 1];
        filterOperand3 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 2];
        filterOperand4 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 3];
        filterOperand5 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 4];

        filterOperand6 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 5];
        filterOperand7 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 6];
        filterOperand8 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 7];
        filterOperand9 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 8];
        filterOperand10 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 9];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;
        input1filter10 += inputOperand1 * filterOperand10;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x % 4 < 4){
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 0] = inputTemp1;
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 1] = inputTemp2;
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 2] = inputTemp3;
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 3] = inputTemp4;
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 4] = inputTemp5;
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 5] = inputTemp6;
            inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 6] = inputTemp7;
        }

        if(threadIdx.x < 32) {
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 0] = filterTemp1;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 1] = filterTemp2;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 2] = filterTemp3;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 3] = filterTemp4;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 4] = filterTemp5;

            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 5] = filterTemp6;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 6] = filterTemp7;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 7] = filterTemp8;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 8] = filterTemp9;
            filterSharedBuffer2[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 9] = filterTemp10;
        }

        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
                // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 4;
        inputTemp1 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 0];
        inputTemp2 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 1];
        inputTemp3 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 2];
        inputTemp4 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 3];
        inputTemp5 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 4];
        inputTemp6 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 5];
        inputTemp7 = input[blockLoadInputStartIdx + warpID * 7 + (laneID % 4) * 7 * 7 + 6];
        
        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 0];
        filterTemp2 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 1];
        filterTemp3 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 2];
        filterTemp4 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 3];
        filterTemp5 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 4];
        filterTemp6 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 5];
        filterTemp7 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 6];
        filterTemp8 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 7];
        filterTemp9 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 8];
        filterTemp10 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * inputChannel * 9];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 4];
        inputOperand6 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 0];
        filterOperand2 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 1];
        filterOperand3 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 2];
        filterOperand4 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 3];
        filterOperand5 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 4];

        filterOperand6 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 5];
        filterOperand7 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 6];
        filterOperand8 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 7];
        filterOperand9 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 8];
        filterOperand10 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 9];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;
        input1filter10 += inputOperand1 * filterOperand10;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x % 4 < 4){
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 0] = inputTemp1;
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 1] = inputTemp2;
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 2] = inputTemp3;
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 3] = inputTemp4;
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 4] = inputTemp5;
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 5] = inputTemp6;
            inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 6] = inputTemp7;
        }

        if(threadIdx.x < 32) {
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 0] = filterTemp1;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 1] = filterTemp2;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 2] = filterTemp3;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 3] = filterTemp4;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 4] = filterTemp5;

            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 5] = filterTemp6;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 6] = filterTemp7;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 7] = filterTemp8;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 8] = filterTemp9;
            filterSharedBuffer1[(threadIdx.x / 4) * 4 + threadIdx.x % 4 + 4 * 8 * 9] = filterTemp10;
        }

        __syncthreads();
    }
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads form one group
    #pragma unroll
    for (int offset = (4 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down_sync(0xffffffff, input1filter1, offset, 4);
        input1filter2 += __shfl_down_sync(0xffffffff, input1filter2, offset, 4);
        input1filter3 += __shfl_down_sync(0xffffffff, input1filter3, offset, 4);
        input1filter4 += __shfl_down_sync(0xffffffff, input1filter4, offset, 4);
        input1filter5 += __shfl_down_sync(0xffffffff, input1filter5, offset, 4);

        input1filter6 += __shfl_down_sync(0xffffffff, input1filter6, offset, 4);
        input1filter7 += __shfl_down_sync(0xffffffff, input1filter7, offset, 4);
        input1filter8 += __shfl_down_sync(0xffffffff, input1filter8, offset, 4);
        input1filter9 += __shfl_down_sync(0xffffffff, input1filter9, offset, 4);
        input1filter10 += __shfl_down_sync(0xffffffff, input1filter10, offset, 4);

        input2filter1 += __shfl_down_sync(0xffffffff, input2filter1, offset, 4);
        input2filter2 += __shfl_down_sync(0xffffffff, input2filter2, offset, 4);
        input2filter3 += __shfl_down_sync(0xffffffff, input2filter3, offset, 4);
        input2filter4 += __shfl_down_sync(0xffffffff, input2filter4, offset, 4);
        input2filter5 += __shfl_down_sync(0xffffffff, input2filter5, offset, 4);

        input2filter6 += __shfl_down_sync(0xffffffff, input2filter6, offset, 4);
        input2filter7 += __shfl_down_sync(0xffffffff, input2filter7, offset, 4);
        input2filter8 += __shfl_down_sync(0xffffffff, input2filter8, offset, 4);
        input2filter9 += __shfl_down_sync(0xffffffff, input2filter9, offset, 4);
        input2filter10 += __shfl_down_sync(0xffffffff, input2filter10, offset, 4);

        input3filter1 += __shfl_down_sync(0xffffffff, input3filter1, offset, 4);
        input3filter2 += __shfl_down_sync(0xffffffff, input3filter2, offset, 4);
        input3filter3 += __shfl_down_sync(0xffffffff, input3filter3, offset, 4);
        input3filter4 += __shfl_down_sync(0xffffffff, input3filter4, offset, 4);
        input3filter5 += __shfl_down_sync(0xffffffff, input3filter5, offset, 4);

        input3filter6 += __shfl_down_sync(0xffffffff, input3filter6, offset, 4);
        input3filter7 += __shfl_down_sync(0xffffffff, input3filter7, offset, 4);
        input3filter8 += __shfl_down_sync(0xffffffff, input3filter8, offset, 4);
        input3filter9 += __shfl_down_sync(0xffffffff, input3filter9, offset, 4);
        input3filter10 += __shfl_down_sync(0xffffffff, input3filter10, offset, 4);

        input4filter1 += __shfl_down_sync(0xffffffff, input4filter1, offset, 4);
        input4filter2 += __shfl_down_sync(0xffffffff, input4filter2, offset, 4);
        input4filter3 += __shfl_down_sync(0xffffffff, input4filter3, offset, 4);
        input4filter4 += __shfl_down_sync(0xffffffff, input4filter4, offset, 4);
        input4filter5 += __shfl_down_sync(0xffffffff, input4filter5, offset, 4);

        input4filter6 += __shfl_down_sync(0xffffffff, input4filter6, offset, 4);
        input4filter7 += __shfl_down_sync(0xffffffff, input4filter7, offset, 4);
        input4filter8 += __shfl_down_sync(0xffffffff, input4filter8, offset, 4);
        input4filter9 += __shfl_down_sync(0xffffffff, input4filter9, offset, 4);
        input4filter10 += __shfl_down_sync(0xffffffff, input4filter10, offset, 4);

        input5filter1 += __shfl_down_sync(0xffffffff, input5filter1, offset, 4);
        input5filter2 += __shfl_down_sync(0xffffffff, input5filter2, offset, 4);
        input5filter3 += __shfl_down_sync(0xffffffff, input5filter3, offset, 4);
        input5filter4 += __shfl_down_sync(0xffffffff, input5filter4, offset, 4);
        input5filter5 += __shfl_down_sync(0xffffffff, input5filter5, offset, 4);

        input5filter6 += __shfl_down_sync(0xffffffff, input5filter6, offset, 4);
        input5filter7 += __shfl_down_sync(0xffffffff, input5filter7, offset, 4);
        input5filter8 += __shfl_down_sync(0xffffffff, input5filter8, offset, 4);
        input5filter9 += __shfl_down_sync(0xffffffff, input5filter9, offset, 4);
        input5filter10 += __shfl_down_sync(0xffffffff, input5filter10, offset, 4);

        input6filter1 += __shfl_down_sync(0xffffffff, input6filter1, offset, 4);
        input6filter2 += __shfl_down_sync(0xffffffff, input6filter2, offset, 4);
        input6filter3 += __shfl_down_sync(0xffffffff, input6filter3, offset, 4);
        input6filter4 += __shfl_down_sync(0xffffffff, input6filter4, offset, 4);
        input6filter5 += __shfl_down_sync(0xffffffff, input6filter5, offset, 4);

        input6filter6 += __shfl_down_sync(0xffffffff, input6filter6, offset, 4);
        input6filter7 += __shfl_down_sync(0xffffffff, input6filter7, offset, 4);
        input6filter8 += __shfl_down_sync(0xffffffff, input6filter8, offset, 4);
        input6filter9 += __shfl_down_sync(0xffffffff, input6filter9, offset, 4);
        input6filter10 += __shfl_down_sync(0xffffffff, input6filter10, offset, 4);

        input7filter1 += __shfl_down_sync(0xffffffff, input7filter1, offset, 4);
        input7filter2 += __shfl_down_sync(0xffffffff, input7filter2, offset, 4);
        input7filter3 += __shfl_down_sync(0xffffffff, input7filter3, offset, 4);
        input7filter4 += __shfl_down_sync(0xffffffff, input7filter4, offset, 4);
        input7filter5 += __shfl_down_sync(0xffffffff, input7filter5, offset, 4);

        input7filter6 += __shfl_down_sync(0xffffffff, input7filter6, offset, 4);
        input7filter7 += __shfl_down_sync(0xffffffff, input7filter7, offset, 4);
        input7filter8 += __shfl_down_sync(0xffffffff, input7filter8, offset, 4);
        input7filter9 += __shfl_down_sync(0xffffffff, input7filter9, offset, 4);
        input7filter10 += __shfl_down_sync(0xffffffff, input7filter10, offset, 4);
    }

    // Store output
    int blockWriteOutputStartIdx = (blockIdx.x / 2) * outputWidth * outputHeight * outputChannel + (blockIdx.x % 2) * outputWidth * outputHeight * (outputChannel / 2);

    if(laneID % 4 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter2;

        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter3;

        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter4;

        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter5;

        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter6;

        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter7;

        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter8;

        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter9;

        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter10;
    }
}

/*
Pointwise Convolution Kernel
InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v3

Grid:
    gridDim.x = (128 * 160 * 7 * 7) / (7 * 7 * 80);
Block:
    blockDim.x = 32 * 7;

warpNum = 7
WarpH = 7
WarpW = 80
Cnum = 4

One thread block contains 7 warps, 7 * 32 = 224 threads. 
Each thread block is responsible for generating 7 * 7 * 80 output data.
Each warp is responsible for generating 7 * 80 output data.

V100S: 128 576 7 160
Kernel: 0.424416 ms
cuDNN:  0.302816 ms
*/

__global__ void InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v3(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {
    
    // warpNum(7) warps in total, each warp uses warpH (7) * Cnum (8) input data each time
    __shared__ float inputSharedBuffer1[7 * 7 * 4];
    __shared__ float inputSharedBuffer2[7 * 7 * 4];

    // each block generates WarpW (80) output channels. every time, a block uses Cnum (4) channels in a filter
    __shared__ float filterSharedBuffer1[4 * 80];
    __shared__ float filterSharedBuffer2[4 * 80];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0, filterTemp2 = 0;

    // to hold operands
    // same number as temp registers
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0, inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0, filterOperand7 = 0, filterOperand8 = 0, filterOperand9 = 0, filterOperand10 = 0;
    
    // to hold intermediate result
    // number of input temp * number of filter temp = 7 * 20
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0; 
    float input1filter6 = 0, input1filter7 = 0, input1filter8 = 0, input1filter9 = 0, input1filter10 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0, input2filter7 = 0, input2filter8 = 0, input2filter9 = 0, input2filter10 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0, input3filter7 = 0, input3filter8 = 0, input3filter9 = 0, input3filter10 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0, input4filter7 = 0, input4filter8 = 0, input4filter9 = 0, input4filter10 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0, input5filter4 = 0, input5filter5 = 0;
    float input5filter6 = 0, input5filter7 = 0, input5filter8 = 0, input5filter9 = 0, input5filter10 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0, input6filter4 = 0, input6filter5 = 0;
    float input6filter6 = 0, input6filter7 = 0, input6filter8 = 0, input6filter9 = 0, input6filter10 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0, input7filter4 = 0, input7filter5 = 0;
    float input7filter6 = 0, input7filter7 = 0, input7filter8 = 0, input7filter9 = 0, input7filter10 = 0;

    int warpID = threadIdx.x / 32;    // each block contains 7 warps, warpID 0 - 6
    int laneID = threadIdx.x % 32;    // each warp contains 32 threads, laneID 0 - 31

    // load Cnum (4) channels of data from input (7 * 7 * 4) and filter (80 * 4), and store into shared buffer 1
    // input
    int blockLoadInputStartIdx = (blockIdx.x / 2) * inputChannel * inputHeight * inputWidth;
    if(threadIdx.x < 7 * 7 * 4) {
        inputSharedBuffer1[threadIdx.x] = input[blockLoadInputStartIdx + threadIdx.x];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 2) * inputChannel * (outputChannel / 2);
    filterSharedBuffer1[threadIdx.x + 32 * 7 * 0] = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 0];       // 56 channels
    if(threadIdx.x < (4 * 80 - 32 * 7)){
        filterSharedBuffer1[threadIdx.x + 32 * 7 * 1] = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 1];   // 24 channels
    }
    __syncthreads();
    
    // For loop begins
    for(int i = 0; i < inputChannel / (2 * 4); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 4;
        if(threadIdx.x < 7 * 7 * 4) {
            inputTemp1 = input[blockLoadInputStartIdx + threadIdx.x];
        }
        
        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 0];
        if(threadIdx.x < (4 * 80 - 32 * 7)){
            filterTemp2 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 1];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 4];
        inputOperand6 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer1[warpID * 7 + (laneID % 4) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 0];
        filterOperand2 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 1];
        filterOperand3 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 2];
        filterOperand4 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 3];
        filterOperand5 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 4];

        filterOperand6 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 5];
        filterOperand7 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 6];
        filterOperand8 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 7];
        filterOperand9 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 8];
        filterOperand10 = filterSharedBuffer1[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 9];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;
        input1filter10 += inputOperand1 * filterOperand10;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 7 * 7 * 4){
            inputSharedBuffer2[threadIdx.x] = inputTemp1;
        }

        filterSharedBuffer2[threadIdx.x + 32 * 7 * 0] = filterTemp1;       // 56 channels
        if(threadIdx.x < (4 * 80 - 32 * 7)){
            filterSharedBuffer2[threadIdx.x + 32 * 7 * 1] = filterTemp2;   // 24 channels
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 4;
        if(threadIdx.x < 7 * 7 * 4) {
            inputTemp1 = input[blockLoadInputStartIdx + threadIdx.x];
        }
        
        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 0];
        if(threadIdx.x < (4 * 80 - 32 * 7)){
            filterTemp2 = filter[blockLoadFilterStartIdx + (threadIdx.x / 4) * inputChannel + (threadIdx.x % 4) + (32 / 4) * 7 * inputChannel * 1];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 4];
        inputOperand6 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer2[warpID * 7 + (laneID % 4) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 0];
        filterOperand2 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 1];
        filterOperand3 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 2];
        filterOperand4 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 3];
        filterOperand5 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 4];

        filterOperand6 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 5];
        filterOperand7 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 6];
        filterOperand8 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 7];
        filterOperand9 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 8];
        filterOperand10 = filterSharedBuffer2[(laneID / 4) * 4 + laneID % 4 + 4 * 8 * 9];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;
        input1filter10 += inputOperand1 * filterOperand10;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 7 * 7 * 4){
            inputSharedBuffer1[threadIdx.x] = inputTemp1;
        }

        filterSharedBuffer1[threadIdx.x + 32 * 7 * 0] = filterTemp1;       // 56 channels
        if(threadIdx.x < (4 * 80 - 32 * 7)){
            filterSharedBuffer1[threadIdx.x + 32 * 7 * 1] = filterTemp2;   // 24 channels
        }
        __syncthreads();
    }
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads form one group
    #pragma unroll
    for (int offset = (4 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down_sync(0xffffffff, input1filter1, offset, 4);
        input1filter2 += __shfl_down_sync(0xffffffff, input1filter2, offset, 4);
        input1filter3 += __shfl_down_sync(0xffffffff, input1filter3, offset, 4);
        input1filter4 += __shfl_down_sync(0xffffffff, input1filter4, offset, 4);
        input1filter5 += __shfl_down_sync(0xffffffff, input1filter5, offset, 4);

        input1filter6 += __shfl_down_sync(0xffffffff, input1filter6, offset, 4);
        input1filter7 += __shfl_down_sync(0xffffffff, input1filter7, offset, 4);
        input1filter8 += __shfl_down_sync(0xffffffff, input1filter8, offset, 4);
        input1filter9 += __shfl_down_sync(0xffffffff, input1filter9, offset, 4);
        input1filter10 += __shfl_down_sync(0xffffffff, input1filter10, offset, 4);

        input2filter1 += __shfl_down_sync(0xffffffff, input2filter1, offset, 4);
        input2filter2 += __shfl_down_sync(0xffffffff, input2filter2, offset, 4);
        input2filter3 += __shfl_down_sync(0xffffffff, input2filter3, offset, 4);
        input2filter4 += __shfl_down_sync(0xffffffff, input2filter4, offset, 4);
        input2filter5 += __shfl_down_sync(0xffffffff, input2filter5, offset, 4);

        input2filter6 += __shfl_down_sync(0xffffffff, input2filter6, offset, 4);
        input2filter7 += __shfl_down_sync(0xffffffff, input2filter7, offset, 4);
        input2filter8 += __shfl_down_sync(0xffffffff, input2filter8, offset, 4);
        input2filter9 += __shfl_down_sync(0xffffffff, input2filter9, offset, 4);
        input2filter10 += __shfl_down_sync(0xffffffff, input2filter10, offset, 4);

        input3filter1 += __shfl_down_sync(0xffffffff, input3filter1, offset, 4);
        input3filter2 += __shfl_down_sync(0xffffffff, input3filter2, offset, 4);
        input3filter3 += __shfl_down_sync(0xffffffff, input3filter3, offset, 4);
        input3filter4 += __shfl_down_sync(0xffffffff, input3filter4, offset, 4);
        input3filter5 += __shfl_down_sync(0xffffffff, input3filter5, offset, 4);

        input3filter6 += __shfl_down_sync(0xffffffff, input3filter6, offset, 4);
        input3filter7 += __shfl_down_sync(0xffffffff, input3filter7, offset, 4);
        input3filter8 += __shfl_down_sync(0xffffffff, input3filter8, offset, 4);
        input3filter9 += __shfl_down_sync(0xffffffff, input3filter9, offset, 4);
        input3filter10 += __shfl_down_sync(0xffffffff, input3filter10, offset, 4);

        input4filter1 += __shfl_down_sync(0xffffffff, input4filter1, offset, 4);
        input4filter2 += __shfl_down_sync(0xffffffff, input4filter2, offset, 4);
        input4filter3 += __shfl_down_sync(0xffffffff, input4filter3, offset, 4);
        input4filter4 += __shfl_down_sync(0xffffffff, input4filter4, offset, 4);
        input4filter5 += __shfl_down_sync(0xffffffff, input4filter5, offset, 4);

        input4filter6 += __shfl_down_sync(0xffffffff, input4filter6, offset, 4);
        input4filter7 += __shfl_down_sync(0xffffffff, input4filter7, offset, 4);
        input4filter8 += __shfl_down_sync(0xffffffff, input4filter8, offset, 4);
        input4filter9 += __shfl_down_sync(0xffffffff, input4filter9, offset, 4);
        input4filter10 += __shfl_down_sync(0xffffffff, input4filter10, offset, 4);

        input5filter1 += __shfl_down_sync(0xffffffff, input5filter1, offset, 4);
        input5filter2 += __shfl_down_sync(0xffffffff, input5filter2, offset, 4);
        input5filter3 += __shfl_down_sync(0xffffffff, input5filter3, offset, 4);
        input5filter4 += __shfl_down_sync(0xffffffff, input5filter4, offset, 4);
        input5filter5 += __shfl_down_sync(0xffffffff, input5filter5, offset, 4);

        input5filter6 += __shfl_down_sync(0xffffffff, input5filter6, offset, 4);
        input5filter7 += __shfl_down_sync(0xffffffff, input5filter7, offset, 4);
        input5filter8 += __shfl_down_sync(0xffffffff, input5filter8, offset, 4);
        input5filter9 += __shfl_down_sync(0xffffffff, input5filter9, offset, 4);
        input5filter10 += __shfl_down_sync(0xffffffff, input5filter10, offset, 4);

        input6filter1 += __shfl_down_sync(0xffffffff, input6filter1, offset, 4);
        input6filter2 += __shfl_down_sync(0xffffffff, input6filter2, offset, 4);
        input6filter3 += __shfl_down_sync(0xffffffff, input6filter3, offset, 4);
        input6filter4 += __shfl_down_sync(0xffffffff, input6filter4, offset, 4);
        input6filter5 += __shfl_down_sync(0xffffffff, input6filter5, offset, 4);

        input6filter6 += __shfl_down_sync(0xffffffff, input6filter6, offset, 4);
        input6filter7 += __shfl_down_sync(0xffffffff, input6filter7, offset, 4);
        input6filter8 += __shfl_down_sync(0xffffffff, input6filter8, offset, 4);
        input6filter9 += __shfl_down_sync(0xffffffff, input6filter9, offset, 4);
        input6filter10 += __shfl_down_sync(0xffffffff, input6filter10, offset, 4);

        input7filter1 += __shfl_down_sync(0xffffffff, input7filter1, offset, 4);
        input7filter2 += __shfl_down_sync(0xffffffff, input7filter2, offset, 4);
        input7filter3 += __shfl_down_sync(0xffffffff, input7filter3, offset, 4);
        input7filter4 += __shfl_down_sync(0xffffffff, input7filter4, offset, 4);
        input7filter5 += __shfl_down_sync(0xffffffff, input7filter5, offset, 4);

        input7filter6 += __shfl_down_sync(0xffffffff, input7filter6, offset, 4);
        input7filter7 += __shfl_down_sync(0xffffffff, input7filter7, offset, 4);
        input7filter8 += __shfl_down_sync(0xffffffff, input7filter8, offset, 4);
        input7filter9 += __shfl_down_sync(0xffffffff, input7filter9, offset, 4);
        input7filter10 += __shfl_down_sync(0xffffffff, input7filter10, offset, 4);
    }

    // Store output
    int blockWriteOutputStartIdx = (blockIdx.x / 2) * outputWidth * outputHeight * outputChannel + (blockIdx.x % 2) * outputWidth * outputHeight * (outputChannel / 2);

    if(laneID % 4 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter2;

        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter3;

        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter4;

        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter5;

        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter6;

        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter7;

        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter8;

        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter9;

        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + warpID * outputWidth + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter10;
    }
}

// ===========================================================================
// Input Size 7 x 7, Input Channel 160, Output Channel 960
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_160_OutChannel_960

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 160 7 960
Kernel: 0.025216 ms
cuDNN: 0.058112ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_160_OutChannel_960
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_160_OutChannel_960
__global__ void InputBatch_1_Input_7x7_InChannel_160_OutChannel_960(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_160_OutChannel_960

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 160 7 960
Kernel: ms
cuDNN: ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_160_OutChannel_960
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_160_OutChannel_960
__global__ void InputBatch_8_Input_7x7_InChannel_160_OutChannel_960(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {


}

// ===========================================================================
// Input Size 7 x 7, Input Channel 960, Output Channel 160
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_960_OutChannel_160

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 960 7 160
Kernel: 0.107904 ms
cuDNN:  0.145344 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_960_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_960_OutChannel_160
__global__ void InputBatch_1_Input_7x7_InChannel_960_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_960_OutChannel_160

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 960 7 160
Kernel: 0.109408 ms
cuDNN:  0.144608 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_960_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_960_OutChannel_160
__global__ void InputBatch_8_Input_7x7_InChannel_960_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_16_Input_7x7_InChannel_960_OutChannel_160

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 16 960 7 160
Kernel: ms
cuDNN:  ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_16_Input_7x7_InChannel_960_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_16_Input_7x7_InChannel_960_OutChannel_160
__global__ void InputBatch_16_Input_7x7_InChannel_960_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

}

// ===========================================================================
// Input Size 7 x 7, Input Channel 960, Output Channel 320
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_960_OutChannel_320

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 960 7 320
Kernel: 0.109440 ms
cuDNN:  0.159040 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_960_OutChannel_320
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_960_OutChannel_320
__global__ void InputBatch_1_Input_7x7_InChannel_960_OutChannel_320(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_960_OutChannel_320

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 960 7 320
Kernel: ms
cuDNN:  ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_960_OutChannel_320
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_960_OutChannel_320
__global__ void InputBatch_8_Input_7x7_InChannel_960_OutChannel_320(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

}

// ===========================================================================
// Input Size 7 x 7, Input Channel 320, Output Channel 1280
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 320 7 1280
Kernel: 0.043136 ms
cuDNN:  0.070688 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280
__global__ void InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 320 7 1280
Kernel: ms
cuDNN:  ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280
__global__ void InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

// ===========================================================================
// Input Size 7 x 7, Input Channel 672, Output Channel 192
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_672_OutChannel_192

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 672 7 192
Kernel: 0.064352 ms
cuDNN:  0.104544 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_672_OutChannel_192
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_672_OutChannel_192
__global__ void InputBatch_1_Input_7x7_InChannel_672_OutChannel_192(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_672_OutChannel_192

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 672 7 192
Kernel: ms
cuDNN:  ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_672_OutChannel_192
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_672_OutChannel_192
__global__ void InputBatch_8_Input_7x7_InChannel_672_OutChannel_192(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {
}

// ===========================================================================
// Input Size 7 x 7, Input Channel 192, Output Channel 1152
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 192 7 1152
Kernel: 0.030432 ms
cuDNN:  0.065888 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152
__global__ void InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 8 192 7 1152
Kernel: ms
cuDNN:  ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152
// On GPU, use this signature: __global__ void InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152
__global__ void InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {
}

// ===========================================================================
// Input Size 7 x 7, Input Channel 1152, Output Channel 192
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 1152 7 192
Kernel: 0.131040 ms
cuDNN:  0.181440 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192
__global__ void InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

// ===========================================================================
// Input Size 7 x 7, Input Channel 1152, Output Channel 320
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320

Grid:
    gridDim.x = outputBatchNumber;
    gridDim.y = outputChannel / 16;
Block:
    blockDim.x = 7;
    blockDim.y = 7;
    blockDim.z = 16;

Each block produces 7 x 7 x 16 output data.
All threads find the corresponding input start index and filter start index, and do the pointwise convolution.
Then all threads store the result to the output data.

V100S: 1 1152 7 320
Kernel: 0.128672 ms
cuDNN:  0.182304 ms
*/

// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320
// On GPU, use this signature: __global__ void InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320
__global__ void InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

/*
To test Pointwise convolution kernels.

Arguments:
    1. Input Batch Number
    2. Input Channel
    3. Input Height
    4. Output Channel
    5. Data Layout Format

Data Format:
    In this testing code, the input data is initialized to be NCHW format, which is same as the input data for layers in the model.
    But the kernel handle data in NHWC format, so we need to convert input data from NCHW to NHWC.
    The output for kernels are in NCHW format, and this would be the input for the next layer in the model, which is a depthwise convolution layer.
    So the output of pointwise kernels doesn't need to be converted.
*/
int main(int argc, char* argv[]) {
    // GPU warm up for benchmarking
    warmup<<<128, 128>>>();

    // Input dimension
    int inputBatchNumber = 0;
    int inputChannel = 0;
    int inputHeight = 0;
    int inputWidth = 0;

    // Filter dimension
    int filterOutChannel = 0;
    int filterInChannel = 0;
    int filterHeight = 0;
    int filterWidth = 0;

    // Output dimension
    int outputBatchNumber = 0;
    int outputChannel = 0;
    int outputHeight = 0;
    int outputWidth = 0;

    float alpha = 1.0;
    float beta = 0.0;

    // Initialize all required parameters
    // Input dimensions
    inputBatchNumber = atoi(argv[1]);
    inputChannel = atoi(argv[2]);
    inputHeight = atoi(argv[3]);
    inputWidth = inputHeight;

    // Filter dimensions
    filterOutChannel = atoi(argv[4]);  // this equals to the number of output channel
    filterInChannel = inputChannel;    // this equals to the number of input channel
    filterHeight = 1;
    filterWidth = filterHeight;

    // Output dimensions
    outputBatchNumber = inputBatchNumber;
    outputChannel = atoi(argv[4]);
    outputHeight = inputHeight;
    outputWidth = inputWidth;

    // Data size
    int inputSize = inputBatchNumber * inputChannel * inputHeight * inputWidth;
    int filterSize = filterOutChannel * filterInChannel * filterHeight * filterWidth;
    int outputSize = outputBatchNumber * outputChannel * outputHeight * outputWidth;

    // allocate host memory and device memory for input data, and copy it from host to device.
    float* hostInput = (float*)malloc(inputSize * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < inputSize; i++) {
        hostInput[i] = (float)(float(rand()) / float((RAND_MAX)) * 5.0);
    }
    float* deviceInput;
    checkCuda(cudaMalloc((void**)&deviceInput, inputSize * sizeof(float)));
    checkCuda(cudaMemcpy(deviceInput, hostInput, inputSize * sizeof(float), cudaMemcpyHostToDevice));

    // allocate host memory and device memory for filter data, and copy it from host to device.
    float* hostFilter = (float*)malloc(filterSize * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < filterSize; i++) {
        hostFilter[i] = (float)(float(rand()) / float((RAND_MAX)) * 5.0);
    }
    float* deviceFilter;
    checkCuda(cudaMalloc((void**)&deviceFilter, filterSize * sizeof(float)));
    checkCuda(cudaMemcpy(deviceFilter, hostFilter, filterSize * sizeof(float), cudaMemcpyHostToDevice));

    // allocate host memory and device memory for kernel output data
    float* hostKernelOutput = (float*)malloc(outputSize * sizeof(float));
    float* deviceKernelOutput;
    checkCuda(cudaMalloc((void**)&deviceKernelOutput, outputSize * sizeof(float)));

    // allocate host memory and device memory for Cudnn output data
    float* hostCudnnOutput = (float*)malloc(outputSize * sizeof(float));
    float* deviceCudnnOutput;
    checkCuda(cudaMalloc((void**)&deviceCudnnOutput, outputSize * sizeof(float)));

    // Use Cuda event to measure running time
    float elapsedTime = 0.0;
    float kernelTime = 0.0;
    float cudnnTime = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel Invocation - Pointwise Kernels
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 576, Output Channel 160
    // ===========================================================================
    if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 8 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_8_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 16 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {

    } else if (inputBatchNumber == 32 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {

    } else if (inputBatchNumber == 64 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {

    } else if (inputBatchNumber == 128 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 7 * 80));
        dim3 blockSize(7 * 32);
        InputBatch_128_Input_7x7_InChannel_576_OutChannel_160_v3<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 160, Output Channel 960
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 160 && outputChannel == 960) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 8 && inputHeight == 7 && inputChannel == 160 && outputChannel == 960) {

    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 960, Output Channel 160
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 960 && outputChannel == 160) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 8 && inputHeight == 7 && inputChannel == 960 && outputChannel == 160) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_8_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 16 && inputHeight == 7 && inputChannel == 960 && outputChannel == 160) {

    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 960, Output Channel 320
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 960 && outputChannel == 320) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 8 && inputHeight == 7 && inputChannel == 960 && outputChannel == 320) {

    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 320, Output Channel 1280
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 320 && outputChannel == 1280) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 672, Output Channel 192
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 672 && outputChannel == 192) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 8 && inputHeight == 7 && inputChannel == 672 && outputChannel == 192) {

    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 192, Output Channel 1152
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 192 && outputChannel == 1152) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } else if (inputBatchNumber == 8 && inputHeight == 7 && inputChannel == 192 && outputChannel == 1152) {

    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 1152, Output Channel 192
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 1152 && outputChannel == 192) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    } 
    // ===========================================================================
    // Input Size 7 x 7, Input Channel 1152, Output Channel 320
    // ===========================================================================
    else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 1152 && outputChannel == 320) {
        cudaEventRecord(start);
        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
    }

    // Copy kernel output from device to host
    checkCuda(cudaMemcpy(hostKernelOutput, deviceKernelOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Create cudnn
    cudnnHandle_t cudnn;
    checkCudnn(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnTensorDescriptor_t outputDesc;

    // input descriptor
    checkCudnn(cudnnCreateTensorDescriptor(&inputDesc));
    checkCudnn(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchNumber, inputChannel, inputHeight, inputWidth));

    // filter descriptor
    checkCudnn(cudnnCreateFilterDescriptor(&filterDesc));
    checkCudnn(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterOutChannel, filterInChannel, filterHeight, filterWidth));

    // output descriptor
    checkCudnn(cudnnCreateTensorDescriptor(&outputDesc));
    checkCudnn(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputBatchNumber, outputChannel, outputHeight, outputWidth));

    // convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    checkCudnn(cudnnCreateConvolutionDescriptor(&convDesc));
    // padding Height/Width = 0; stride Height/Width = 1, dilation Height/Width is 1
    checkCudnn(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // set algorithm
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnFindConvolutionForwardAlgorithm(cudnn,
        inputDesc,
        filterDesc,
        convDesc,
        outputDesc,
        1,
        &returnedAlgoCount,
        &perfResults);
    cudnnConvolutionFwdAlgo_t algo = perfResults.algo;

    // create workspace
    size_t workspaceSize = 0;
    void* workspaceData = nullptr;
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
    checkCuda(cudaMalloc(&workspaceData, workspaceSize));

    // Use CUDNN to check kernel result and measure running time
    cudaEventRecord(start);
    cudnnConvolutionForward(
        cudnn,
        &alpha,
        inputDesc,
        deviceInput,
        filterDesc,
        deviceFilter,
        convDesc,
        algo,
        workspaceData,
        workspaceSize,
        &beta,
        outputDesc,
        deviceCudnnOutput
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed Time for cuDNN Pointwise Convolution: %f ms.\n", elapsedTime);
    cudnnTime = elapsedTime;
    // Copy Cudnn result from device to host
    checkCuda(cudaMemcpy(hostCudnnOutput, deviceCudnnOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    writeCsv(inputBatchNumber, inputChannel, inputHeight, filterHeight, 1, kernelTime, cudnnTime);

    // Compare Kernel result and Cudnn result
    if (compareOutput(outputBatchNumber, outputChannel, outputHeight, outputWidth, hostKernelOutput, hostCudnnOutput, 1) == 0) {
        printf("Kernel Calculation Correct.\n");
    }

    // Free all allocated memory spaces
    free(hostInput);
    free(hostFilter);
    free(hostKernelOutput);
    free(hostCudnnOutput);

    cudaFree(deviceInput);
    cudaFree(deviceFilter);
    cudaFree(deviceKernelOutput);
    cudaFree(deviceCudnnOutput);

    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudaFree(workspaceData);

    checkCuda(cudaDeviceReset());
    return 0;
}

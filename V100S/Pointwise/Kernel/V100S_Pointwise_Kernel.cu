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

//CSV文件头,保存路径
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
        //创建文件
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

/*
Pointwise Convolution Kernel
InputBatch_128_Input_7x7_InChannel_576_OutChannel_160

Grid:
    gridDim.x = (outputBatchNumber * outputChannel * outputHeight * outputWidth) / (20 * 160);
Block:
    blockDim.x = 32 * 4;

InputBatchNumber = 128:
    WarpH = 10
    WarpW = 80
    Cnum = 2
    First-Level Block Size: 20 * 160

V100S: 128 576 7 160
Kernel: ms
cuDNN: ms
*/
/*
// On DCU, use this signature: __global__ __launch_bounds__(1024) void InputBatch_128_Input_7x7_InChannel_576_OutChannel_160
// On GPU, use this signature: __global__ void InputBatch_128_Input_7x7_InChannel_576_OutChannel_160
__global__ void InputBatch_128_Input_7x7_InChannel_576_OutChannel_160(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    // Each block loads Cnum * (WarpH * 2) input elements
    __shared__ float inputData1[2 * 10 * 2];
    __shared__ float inputData2[2 * 10 * 2];

    // For warp 0 and warp2, they handle the upper 80 channels of the output. For warp1 and warp3, they handle the lower 80 channnels of the output.
    // In a warp, each thread needs to use 5 filter elements (80 / (32 / 2)), 32 threads in a warp, and four warps are separated to 2 groups. 
    __shared__ float filterData1[2 * 5 * 32];
    __shared__ float filterData2[2 * 5 * 32];

    int warpSize = 32;
    int warpID = threadIdx.x / warpSize;
    int laneID = threadIdx.x % warpSize;

    float inTmp1, inTmp2, inTmp3, inTmp4, inTmp5, inTmp6, inTmp7, inTmp8, inTmp9, inTmp10;
    float filterTmp1, filterTmp2, filterTmp3, filterTmp4, filterTmp5;

    float inOpd1, inOpd2, inOpd3, inOpd4, inOpd5, inOpd6, inOpd7, inOpd8, inOpd9, inOpd10;
    float filterOpd1, filterOpd2, filterOpd3, filterOpd4, filterOpd5;

    float outResult1_1 = 0.0, outResult1_2 = 0.0, outResult1_3 = 0.0, outResult1_4 = 0.0, outResult1_5 = 0.0, outResult1_6 = 0.0, outResult1_7 = 0.0, outResult1_8 = 0.0, outResult1_9 = 0.0, outResult1_10 = 0.0;
    float outResult2_1 = 0.0, outResult2_2 = 0.0, outResult2_3 = 0.0, outResult2_4 = 0.0, outResult2_5 = 0.0, outResult2_6 = 0.0, outResult2_7 = 0.0, outResult2_8 = 0.0, outResult2_9 = 0.0, outResult2_10 = 0.0;
    float outResult3_1 = 0.0, outResult3_2 = 0.0, outResult3_3 = 0.0, outResult3_4 = 0.0, outResult3_5 = 0.0, outResult3_6 = 0.0, outResult3_7 = 0.0, outResult3_8 = 0.0, outResult3_9 = 0.0, outResult3_10 = 0.0;
    float outResult4_1 = 0.0, outResult4_2 = 0.0, outResult4_3 = 0.0, outResult4_4 = 0.0, outResult4_5 = 0.0, outResult4_6 = 0.0, outResult4_7 = 0.0, outResult4_8 = 0.0, outResult4_9 = 0.0, outResult4_10 = 0.0;
    float outResult5_1 = 0.0, outResult5_2 = 0.0, outResult5_3 = 0.0, outResult5_4 = 0.0, outResult5_5 = 0.0, outResult5_6 = 0.0, outResult5_7 = 0.0, outResult5_8 = 0.0, outResult5_9 = 0.0, outResult5_10 = 0.0;

    // Blocks load "Cnum" channels of input and filter data into shared buffer 
    int blockLoadInputStart = blockIdx.x * 20 * inputChannel;
    int blockLoadFilterStart = 0;

    // Load input
    if(threadIdx.x < 2 * 10 * 2) {
        inputData1[threadIdx.x] = input[blockLoadInputStart + (threadIdx.x / 2) * inputChannel + (threadIdx.x % 2)];
    }

    // Load Filter
    for(int i = threadIdx.x; i < 2 * 5 * 32; i += 128) {
        filterData1[i] = filter[(i / 2) * inputChannel + (i % 2)];
    }
    __syncthreads();

    for(int i = 0; i < inputChannel / (2 * 2); i++) {
        // Load next C_num channels of input and filter and store into R_tmps
        inTmp1 = input[];
        inTmp2 = input[];
        inTmp3 = input[];
        inTmp4 = input[];
        inTmp5 = input[];
        inTmp6 = input[];
        inTmp7 = input[];
        inTmp8 = input[];
        inTmp9 = input[];
        inTmp10 = input[];

        filterTmp1 = filter[];
        filterTmp2 = filter[];
        filterTmp3 = filter[];
        filterTmp4 = filter[];
        filterTmp5 = filter[];

        // Copy data from shared buffer 1 to R_operands
        inOpd1 = inputData1[];
        inOpd2 = inputData1[];
        inOpd3 = inputData1[];
        inOpd4 = inputData1[];
        inOpd5 = inputData1[];
        inOpd6 = inputData1[];
        inOpd7 = inputData1[];
        inOpd8 = inputData1[];
        inOpd9 = inputData1[];
        inOpd10 = inputData1[];

        filterOpd1 = filterData1[];
        filterOpd2 = filterData1[];
        filterOpd3 = filterData1[];
        filterOpd4 = filterData1[];
        filterOpd5 = filterData1[];

        // Convolution and store result to R_results
        outResult1_1 += inOpd1 * filterOpd1;
        outResult1_2 += inOpd2 * filterOpd1;
        outResult1_3 += inOpd3 * filterOpd1;
        outResult1_4 += inOpd4 * filterOpd1;
        outResult1_5 += inOpd5 * filterOpd1;
        outResult1_6 += inOpd6 * filterOpd1;
        outResult1_7 += inOpd7 * filterOpd1;
        outResult1_8 += inOpd8 * filterOpd1;
        outResult1_9 += inOpd9 * filterOpd1;
        outResult1_10 += inOpd10 * filterOpd1;

        outResult2_1 += inOpd1 * filterOpd2;
        outResult2_2 += inOpd2 * filterOpd2;
        outResult2_3 += inOpd3 * filterOpd2;
        outResult2_4 += inOpd4 * filterOpd2;
        outResult2_5 += inOpd5 * filterOpd2;
        outResult2_6 += inOpd6 * filterOpd2;
        outResult2_7 += inOpd7 * filterOpd2;
        outResult2_8 += inOpd8 * filterOpd2;
        outResult2_9 += inOpd9 * filterOpd2;
        outResult2_10 += inOpd10 * filterOpd2;

        outResult3_1 += inOpd1 * filterOpd3;
        outResult3_2 += inOpd2 * filterOpd3;
        outResult3_3 += inOpd3 * filterOpd3;
        outResult3_4 += inOpd4 * filterOpd3;
        outResult3_5 += inOpd5 * filterOpd3;
        outResult3_6 += inOpd6 * filterOpd3;
        outResult3_7 += inOpd7 * filterOpd3;
        outResult3_8 += inOpd8 * filterOpd3;
        outResult3_9 += inOpd9 * filterOpd3;
        outResult3_10 += inOpd10 * filterOpd3;

        outResult4_1 += inOpd1 * filterOpd4;
        outResult4_2 += inOpd2 * filterOpd4;
        outResult4_3 += inOpd3 * filterOpd4;
        outResult4_4 += inOpd4 * filterOpd4;
        outResult4_5 += inOpd5 * filterOpd4;
        outResult4_6 += inOpd6 * filterOpd4;
        outResult4_7 += inOpd7 * filterOpd4;
        outResult4_8 += inOpd8 * filterOpd4;
        outResult4_9 += inOpd9 * filterOpd4;
        outResult4_10 += inOpd10 * filterOpd4;

        outResult5_1 += inOpd1 * filterOpd5;
        outResult5_2 += inOpd2 * filterOpd5;
        outResult5_3 += inOpd3 * filterOpd5;
        outResult5_4 += inOpd4 * filterOpd5;
        outResult5_5 += inOpd5 * filterOpd5;
        outResult5_6 += inOpd6 * filterOpd5;
        outResult5_7 += inOpd7 * filterOpd5;
        outResult5_8 += inOpd8 * filterOpd5;
        outResult5_9 += inOpd9 * filterOpd5;
        outResult5_10 += inOpd10 * filterOpd5;

        // Copy data from R_tmps to shared buffer2
        inputData2[] = inTmp1;
        inputData2[] = inTmp2;
        inputData2[] = inTmp3;
        inputData2[] = inTmp4;
        inputData2[] = inTmp5;
        inputData2[] = inTmp6;
        inputData2[] = inTmp7;
        inputData2[] = inTmp8;
        inputData2[] = inTmp9;
        inputData2[] = inTmp10;

        filterData2[] = filterTmp1;
        filterData2[] = filterTmp2;
        filterData2[] = filterTmp3;
        filterData2[] = filterTmp4;
        filterData2[] = filterTmp5;

        __syncthreads();
        // Swap shared buffer1 and buffer2 and repeat
        // Load next C_num channels of input and filter and store into R_tmps
        inTmp1 = input[];
        inTmp2 = input[];
        inTmp3 = input[];
        inTmp4 = input[];
        inTmp5 = input[];
        inTmp6 = input[];
        inTmp7 = input[];
        inTmp8 = input[];
        inTmp9 = input[];
        inTmp10 = input[];

        filterTmp1 = filter[];
        filterTmp2 = filter[];
        filterTmp3 = filter[];
        filterTmp4 = filter[];
        filterTmp5 = filter[];

        // Copy data from shared buffer 2 to R_operands
        inOpd1 = inputData2[];
        inOpd2 = inputData2[];
        inOpd3 = inputData2[];
        inOpd4 = inputData2[];
        inOpd5 = inputData2[];
        inOpd6 = inputData2[];
        inOpd7 = inputData2[];
        inOpd8 = inputData2[];
        inOpd9 = inputData2[];
        inOpd10 = inputData2[];

        filterOpd1 = filterData2[];
        filterOpd2 = filterData2[];
        filterOpd3 = filterData2[];
        filterOpd4 = filterData2[];
        filterOpd5 = filterData2[];

        // Convolution and store result to R_results
        outResult1_1 += inOpd1 * filterOpd1;
        outResult1_2 += inOpd2 * filterOpd1;
        outResult1_3 += inOpd3 * filterOpd1;
        outResult1_4 += inOpd4 * filterOpd1;
        outResult1_5 += inOpd5 * filterOpd1;
        outResult1_6 += inOpd6 * filterOpd1;
        outResult1_7 += inOpd7 * filterOpd1;
        outResult1_8 += inOpd8 * filterOpd1;
        outResult1_9 += inOpd9 * filterOpd1;
        outResult1_10 += inOpd10 * filterOpd1;

        outResult2_1 += inOpd1 * filterOpd2;
        outResult2_2 += inOpd2 * filterOpd2;
        outResult2_3 += inOpd3 * filterOpd2;
        outResult2_4 += inOpd4 * filterOpd2;
        outResult2_5 += inOpd5 * filterOpd2;
        outResult2_6 += inOpd6 * filterOpd2;
        outResult2_7 += inOpd7 * filterOpd2;
        outResult2_8 += inOpd8 * filterOpd2;
        outResult2_9 += inOpd9 * filterOpd2;
        outResult2_10 += inOpd10 * filterOpd2;

        outResult3_1 += inOpd1 * filterOpd3;
        outResult3_2 += inOpd2 * filterOpd3;
        outResult3_3 += inOpd3 * filterOpd3;
        outResult3_4 += inOpd4 * filterOpd3;
        outResult3_5 += inOpd5 * filterOpd3;
        outResult3_6 += inOpd6 * filterOpd3;
        outResult3_7 += inOpd7 * filterOpd3;
        outResult3_8 += inOpd8 * filterOpd3;
        outResult3_9 += inOpd9 * filterOpd3;
        outResult3_10 += inOpd10 * filterOpd3;

        outResult4_1 += inOpd1 * filterOpd4;
        outResult4_2 += inOpd2 * filterOpd4;
        outResult4_3 += inOpd3 * filterOpd4;
        outResult4_4 += inOpd4 * filterOpd4;
        outResult4_5 += inOpd5 * filterOpd4;
        outResult4_6 += inOpd6 * filterOpd4;
        outResult4_7 += inOpd7 * filterOpd4;
        outResult4_8 += inOpd8 * filterOpd4;
        outResult4_9 += inOpd9 * filterOpd4;
        outResult4_10 += inOpd10 * filterOpd4;

        outResult5_1 += inOpd1 * filterOpd5;
        outResult5_2 += inOpd2 * filterOpd5;
        outResult5_3 += inOpd3 * filterOpd5;
        outResult5_4 += inOpd4 * filterOpd5;
        outResult5_5 += inOpd5 * filterOpd5;
        outResult5_6 += inOpd6 * filterOpd5;
        outResult5_7 += inOpd7 * filterOpd5;
        outResult5_8 += inOpd8 * filterOpd5;
        outResult5_9 += inOpd9 * filterOpd5;
        outResult5_10 += inOpd10 * filterOpd5;

        // Copy data from R_tmps to shared buffer2
        inputData1[] = inTmp1;
        inputData1[] = inTmp2;
        inputData1[] = inTmp3;
        inputData1[] = inTmp4;
        inputData1[] = inTmp5;
        inputData1[] = inTmp6;
        inputData1[] = inTmp7;
        inputData1[] = inTmp8;
        inputData1[] = inTmp9;
        inputData1[] = inTmp10;

        filterData1[] = filterTmp1;
        filterData1[] = filterTmp2;
        filterData1[] = filterTmp3;
        filterData1[] = filterTmp4;
        filterData1[] = filterTmp5;
    }

    // Segmented parallel reduction to store final output
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
Kernel: 0.087872 ms
cuDNN:  0.153664 ms
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

Idea:


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
        /*
        cudaEventRecord(start);

        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_16_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
        */
    } else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 160 && outputChannel == 960) {
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
        /*
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
        */
    } else if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 960 && outputChannel == 160) {
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
        /*
        cudaEventRecord(start);

        // Convolution
        dim3 gridSize(outputBatchNumber, outputChannel / 16);
        dim3 blockSize(7, 7, 16);
        InputBatch_16_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
            inputBatchNumber, inputChannel, inputHeight, inputWidth,
            filterOutChannel, filterInChannel, filterHeight, filterWidth,
            outputBatchNumber, outputChannel, outputHeight, outputWidth);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime = elapsedTime;
        printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
        inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, elapsedTime);
        */
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

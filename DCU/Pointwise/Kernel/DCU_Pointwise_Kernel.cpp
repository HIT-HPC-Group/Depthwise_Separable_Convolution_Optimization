#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <random>
#include <vector>
#include <fstream>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include "warmup.h"

#include "InputBatch_128_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_64_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_32_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_16_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_8_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_1_Input_112x112_InChannel_32_OutChannel_16.h"

#include "InputBatch_128_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_64_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_32_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_16_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_8_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_1_Input_112x112_InChannel_16_OutChannel_96.h"

#include "InputBatch_128_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_64_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_32_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_16_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_8_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_1_Input_56x56_InChannel_96_OutChannel_24.h"

#include "InputBatch_128_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_64_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_32_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_16_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_8_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_1_Input_56x56_InChannel_24_OutChannel_144.h"

#include "InputBatch_128_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_64_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_32_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_16_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_8_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_1_Input_56x56_InChannel_144_OutChannel_24.h"

#include "InputBatch_128_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_64_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_32_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_16_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_8_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_1_Input_28x28_InChannel_144_OutChannel_32.h"

#include "InputBatch_128_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_64_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_32_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_16_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_8_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_1_Input_28x28_InChannel_32_OutChannel_192.h"

#include "InputBatch_128_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_64_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_32_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_16_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_8_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_1_Input_28x28_InChannel_192_OutChannel_32.h"

#include "InputBatch_128_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_64_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_32_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_16_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_8_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_1_Input_28x28_InChannel_144_OutChannel_40.h"

#include "InputBatch_128_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_64_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_32_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_16_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_8_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_1_Input_28x28_InChannel_40_OutChannel_240.h"

#include "InputBatch_128_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_64_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_32_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_16_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_8_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_1_Input_28x28_InChannel_240_OutChannel_40.h"

#include "InputBatch_128_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_64_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_32_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_16_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_8_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_1_Input_14x14_InChannel_192_OutChannel_64.h"

#include "InputBatch_128_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_64_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_32_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_16_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_8_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_1_Input_14x14_InChannel_64_OutChannel_384.h"

#include "InputBatch_128_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_64_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_32_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_16_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_8_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_1_Input_14x14_InChannel_384_OutChannel_64.h"

#include "InputBatch_128_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_64_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_32_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_16_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_8_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_1_Input_14x14_InChannel_384_OutChannel_96.h"

#include "InputBatch_128_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_64_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_32_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_16_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_8_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_1_Input_14x14_InChannel_96_OutChannel_576.h"

#include "InputBatch_128_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_64_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_32_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_16_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_8_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_1_Input_14x14_InChannel_576_OutChannel_96.h"

#include "InputBatch_128_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_64_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_32_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_16_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_8_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_1_Input_14x14_InChannel_240_OutChannel_80.h"

#include "InputBatch_128_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_64_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_32_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_16_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_8_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_1_Input_14x14_InChannel_80_OutChannel_480.h"

#include "InputBatch_128_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_64_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_32_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_16_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_8_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_1_Input_14x14_InChannel_480_OutChannel_80.h"

#include "InputBatch_128_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_64_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_32_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_16_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_8_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_1_Input_14x14_InChannel_480_OutChannel_112.h"

#include "InputBatch_128_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_64_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_32_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_16_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_8_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_1_Input_14x14_InChannel_112_OutChannel_672.h"

#include "InputBatch_128_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_64_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_32_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_16_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_8_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_1_Input_14x14_InChannel_672_OutChannel_112.h"

#include "InputBatch_128_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_64_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_32_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_16_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_8_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_1_Input_7x7_InChannel_576_OutChannel_160.h"

#include "InputBatch_128_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_64_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_32_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_16_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_8_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_1_Input_7x7_InChannel_160_OutChannel_960.h"

#include "InputBatch_128_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_64_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_32_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_16_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_8_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_1_Input_7x7_InChannel_960_OutChannel_160.h"

#include "InputBatch_128_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_64_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_32_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_16_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_8_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_1_Input_7x7_InChannel_960_OutChannel_320.h"

#include "InputBatch_128_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_64_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_32_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_16_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280.h"

#include "InputBatch_128_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_64_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_32_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_16_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_8_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_1_Input_7x7_InChannel_672_OutChannel_192.h"

#include "InputBatch_128_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_64_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_32_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_16_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152.h"

#include "InputBatch_128_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_64_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_32_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_16_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_8_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192.h"

#include "InputBatch_128_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_64_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_32_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_16_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_8_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320.h"

using namespace std;

/*
Hip and MIOpen Error Handling

checkHip(err)  - to check if an HIP API call returned some error.
checkKernel()   - to check if the kernel invocation is failed.
checkMIOpen(err) - to check if an MIOpen API call returned some error.
*/
#define checkHip(err) __checkHip(err, __FILE__, __LINE__)
#define checkKernel() __checkKernel(__FILE__, __LINE__)
#define checkMIOpen(err) __checkMIOpen(err, __FILE__, __LINE__)

inline void __checkHip(hipError_t err, const char* file, const int line) {
	if (hipSuccess != err) {
		printf("checkHip() failed at %s : %i : %s\n", file, line, hipGetErrorString(err));
		exit(-1);
	}
}

inline void __checkKernel(const char* file, const int line) {
	hipError_t err = hipGetLastError();
	if (hipSuccess != err) {
		printf("checkKernel() failed at %s : %i : %s\n", file, line, hipGetErrorString(err));
		exit(-1);
	}
}

inline void __checkMIOpen(miopenStatus_t err, const char* file, const int line) {
    if (miopenStatusSuccess != err) {
        printf("checkMIOpen() failed at %s : %i : %s\n", file, line,miopenGetErrorString(err));
        exit(-1);
    }
}

/*
compareOutput():
	Compare the result calculated by our kernel and that by the MIOpen library.
	Use MIOpen library as a reference.
Input:
	n            - batch number
	c            - channel number
	h            - height
	w            - width
	kernelOutput - output data of our kernel
	miopenOutput  - output data of the MIOpen
	delta        - a small value. Allowed numerical differece between each element
Output:
	-1           - our kernel is wrong
	0            - out kernel is correct
*/
int compareOutput(int n, int c, int h, int w, const float* kernelOutput, const float* miopenOutput, float delta) {
	int i, j, k, l;

	// Loop over each element, and compare the value.
	// If the difference is small, then accept, or, reject and return.
	for (i = 0; i < n; i++) {
		for (j = 0; j < c; j++) {
			for (k = 0; k < h; k++) {
				for (l = 0; l < w; l++) {
					if (abs(kernelOutput[i * c * h * w + j * h * w + k * w + l] - miopenOutput[i * c * h * w + j * h * w + k * w + l]) > delta) {
						printf("%f, %f\n", kernelOutput[i * c * h * w + j * h * w + k * w + l], miopenOutput[i * c * h * w + j * h * w + k * w + l]);
						printf("Wrong! Output Batch Idx: %d, Channel Idx: %d, Row Idx: %d, Col Idx: %d\n", i, j, k, l);
						return -1;
					}
				}
			}
		}
	}
	return 0;
}

int main(int argc, char* argv[]) {

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
    checkHip(hipMalloc((void**)&deviceInput, inputSize * sizeof(float)));
    checkHip(hipMemcpy(deviceInput, hostInput, inputSize * sizeof(float), hipMemcpyHostToDevice));

    // allocate host memory and device memory for filter data, and copy it from host to device.
    float* hostFilter = (float*)malloc(filterSize * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < filterSize; i++) {
        hostFilter[i] = (float)(float(rand()) / float((RAND_MAX)) * 5.0);
    }
    float* deviceFilter;
    checkHip(hipMalloc((void**)&deviceFilter, filterSize * sizeof(float)));
    checkHip(hipMemcpy(deviceFilter, hostFilter, filterSize * sizeof(float), hipMemcpyHostToDevice));

    // allocate host memory and device memory for kernel output data
    float* hostKernelOutput = (float*)malloc(outputSize * sizeof(float));
    float* deviceKernelOutput;
    checkHip(hipMalloc((void**)&deviceKernelOutput, outputSize * sizeof(float)));

    // allocate host memory and device memory for MIOpen output data
    float* hostMiopenOutput = (float*)malloc(outputSize * sizeof(float));
    float* deviceMiopenOutput;
    checkHip(hipMalloc((void**)&deviceMiopenOutput, outputSize * sizeof(float)));

    // Use Hip event to measure running time
    float elapsedTime = 0.0;
    float kernelTime = 0.0;
    float miopenTime = 0.0;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // GPU warm up for benchmarking
    warmup<<<1024, 128>>>();

    // Kernel Invocation - Pointwise Kernels
	if(inputBatchNumber == 1) {
		if(inputHeight == 112) {
			if(inputChannel == 32 && outputChannel == 16) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 16 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 56) {
			if(inputChannel == 96 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 12 * 8));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 24 && outputChannel == 144) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 24 * 4));
				dim3 blockSize(7 * 64);
				InputBatch_1_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 28) {
			if(inputChannel == 144 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 32 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 40 && outputChannel == 240) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 14) {
			if(inputChannel == 192 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 2 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 64 && outputChannel == 384) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 96 && outputChannel == 576) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 576 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 80 && outputChannel == 480) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 4 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_1_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 112 && outputChannel == 672) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 7) {
			if(inputChannel == 576 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 2 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 160 && outputChannel == 960) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 320 && outputChannel == 1280) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 1152) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 18 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
				hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		}
	} else if(inputBatchNumber == 8) {
		if(inputHeight == 112) {
			if(inputChannel == 32 && outputChannel == 16) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 16 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                        hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 56) {
			if(inputChannel == 96 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 24 && outputChannel == 144) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 28) {
			if(inputChannel == 144 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 32 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 40 && outputChannel == 240) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 14) {
			if(inputChannel == 192 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 64 && outputChannel == 384) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 64 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 96 && outputChannel == 576) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 192 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_8_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 576 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 80 && outputChannel == 480) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 240 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 28 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 112 && outputChannel == 672) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 28 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 7) {
			if(inputChannel == 576 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 160 && outputChannel == 960) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 120 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 10 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 320 && outputChannel == 1280) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 1152) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 36 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 12 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_8_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		}
	} else if(inputBatchNumber == 16) {
		if(inputHeight == 112) {
			if(inputChannel == 32 && outputChannel == 16) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 16 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 56) {
			if(inputChannel == 96 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 24 && outputChannel == 144) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 28) {
			if(inputChannel == 144 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 32 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 40 && outputChannel == 240) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 14) {
			if(inputChannel == 192 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 64 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 64 && outputChannel == 384) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 96 && outputChannel == 576) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 576 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 80 && outputChannel == 480) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 160 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 112 && outputChannel == 672) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 7) {
			if(inputChannel == 576 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 160 && outputChannel == 960) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 320 && outputChannel == 1280) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 1152) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_16_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		}
	} else if(inputBatchNumber == 32) {
		if(inputHeight == 112) {
			if(inputChannel == 32 && outputChannel == 16) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 16 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 56) {
			if(inputChannel == 96 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 24 && outputChannel == 144) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 28) {
			if(inputChannel == 144 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 32 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 40 && outputChannel == 240) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 14) {
			if(inputChannel == 192 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 64 && outputChannel == 384) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 64 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 96 && outputChannel == 576) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 576 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 80 && outputChannel == 480) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 240 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 112 && outputChannel == 672) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 7) {
			if(inputChannel == 576 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 160 && outputChannel == 960) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 96 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_32_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 320 && outputChannel == 1280) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 1152) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 96 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_32_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_32_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		}
	} else if(inputBatchNumber == 64) {
		if(inputHeight == 112) {
			if(inputChannel == 32 && outputChannel == 16) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 16 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 56) {
			if(inputChannel == 96 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 24 && outputChannel == 144) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 28) {
			if(inputChannel == 144 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 32 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 40 && outputChannel == 240) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 14) {
			if(inputChannel == 192 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 64 && outputChannel == 384) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 96 && outputChannel == 576) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 576 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 80 && outputChannel == 480) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 112 && outputChannel == 672) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 7) {
			if(inputChannel == 576 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 160 && outputChannel == 960) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_64_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 320 && outputChannel == 1280) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 1152) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_64_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_64_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		}
	} else if(inputBatchNumber == 128) {
		if(inputHeight == 112) {
			if(inputChannel == 32 && outputChannel == 16) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 28));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 16 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 56) {
			if(inputChannel == 96 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 24 && outputChannel == 144) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 24) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 28) {
			if(inputChannel == 144 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 32 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 32) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 144 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 40 && outputChannel == 240) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 40) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 14) {
			if(inputChannel == 192 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 64 && outputChannel == 384) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 64) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 384 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 96 && outputChannel == 576) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 576 && outputChannel == 96) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 240 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 80 && outputChannel == 480) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 80) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 480 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 112 && outputChannel == 672) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 14));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 112) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		} else if(inputHeight == 7) {
			if(inputChannel == 576 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 160 && outputChannel == 960) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_128_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 160) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 960 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 320 && outputChannel == 1280) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 672 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 192 && outputChannel == 1152) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
				dim3 blockSize(7 * 64);
				InputBatch_128_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 192) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			} else if(inputChannel == 1152 && outputChannel == 320) {
				hipEventRecord(start);
				dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
				dim3 blockSize(4 * 64);
				InputBatch_128_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterOutChannel, filterInChannel, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth);
                hipEventRecord(stop);
        		hipEventSynchronize(stop);
        		hipEventElapsedTime(&elapsedTime, start, stop);
        		kernelTime = elapsedTime;
			}
		}
	}

    // Copy kernel output from device to host
    checkHip(hipMemcpy(hostKernelOutput, deviceKernelOutput, outputSize * sizeof(float), hipMemcpyDeviceToHost));
	
    // Create miopen
    miopenHandle_t miopen;
    miopenCreate(&miopen);
    
    // input descriptor
    miopenTensorDescriptor_t inputDesc;
    miopenCreateTensorDescriptor(&inputDesc);
    miopenSet4dTensorDescriptor(inputDesc, miopenFloat, inputBatchNumber, inputChannel, inputHeight, inputWidth);
    
    // filter descriptor
    miopenTensorDescriptor_t filterDesc;
    miopenCreateTensorDescriptor(&filterDesc);
    miopenSet4dTensorDescriptor(filterDesc, miopenFloat, filterOutChannel, filterInChannel, filterHeight, filterWidth);
    
    // output descriptor
    miopenTensorDescriptor_t outputDesc;
    miopenCreateTensorDescriptor(&outputDesc);
    miopenSet4dTensorDescriptor(outputDesc, miopenFloat, outputBatchNumber, outputChannel, outputHeight, outputWidth);
    
    // convolution descriptor
    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 1, 1, 1, 1);
    
    // create workspace
    size_t workspaceSize = 0;
    void* workspaceData = nullptr;
    miopenConvolutionForwardGetWorkSpaceSize(miopen, inputDesc, filterDesc, convDesc, outputDesc, &workspaceSize);
    checkHip(hipMalloc(&workspaceData, workspaceSize));

    // set algorithm
    int returnedAlgoCount = 0;
    miopenConvAlgoPerf_t *miopenPerfResults = new miopenConvAlgoPerf_t[1];

    miopenFindConvolutionForwardAlgorithm(
        miopen, inputDesc, deviceInput,
        filterDesc,deviceFilter,
        convDesc,
        outputDesc, deviceMiopenOutput, 1,
        &returnedAlgoCount, miopenPerfResults, workspaceData,
        workspaceSize, false);

    // Use MIOpen to check kernel result and measure running time
    hipEventRecord(start);
    miopenConvolutionForward(
	    miopen, &alpha, inputDesc, deviceInput,
        filterDesc, deviceFilter,
        convDesc, miopenPerfResults->fwd_algo, &beta,
        outputDesc, deviceMiopenOutput, workspaceData,
        workspaceSize);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    miopenTime = elapsedTime;

    // Copy MIOpen result from device to host
    checkHip(hipMemcpy(hostMiopenOutput, deviceMiopenOutput, outputSize * sizeof(float), hipMemcpyDeviceToHost));

    // Compare Kernel result and MIOpen result
    if (compareOutput(outputBatchNumber, outputChannel, outputHeight, outputWidth, hostKernelOutput, hostMiopenOutput, 1) == 0) {
        printf("Kernel Calculation Correct.\n");
		printf("MIOpen time : %f ms.\n", miopenTime);
		printf("Kernel time : %f ms.\n", kernelTime);
    }

    // Free all allocated memory spaces
    free(hostInput);
    free(hostFilter);
    free(hostKernelOutput);
    free(hostMiopenOutput);

    hipFree(deviceInput);
    hipFree(deviceFilter);
    hipFree(deviceKernelOutput);
    hipFree(deviceMiopenOutput);

    miopenDestroy(miopen);
    miopenDestroyTensorDescriptor(inputDesc);
    miopenDestroyTensorDescriptor(outputDesc);
    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(filterDesc);
    hipFree(workspaceData);

    checkHip(hipDeviceReset());
    return 0;
}

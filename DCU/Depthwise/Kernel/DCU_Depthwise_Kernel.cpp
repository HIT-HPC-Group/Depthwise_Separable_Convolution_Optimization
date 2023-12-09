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
#include "Filter3x3_Input7x7_Stride1.h"
#include "Filter3x3_Input14x14_Stride1.h"
#include "Filter3x3_Input14x14_Stride2.h"
#include "Filter3x3_Input28x28_Stride1.h"
#include "Filter3x3_Input28x28_Stride2.h"
#include "Filter3x3_Input56x56_Stride1.h"
#include "Filter3x3_Input56x56_Stride2.h"
#include "Filter3x3_Input112x112_Stride1.h"
#include "Filter3x3_Input112x112_Stride2.h"
#include "Filter5x5_Input7x7_Stride1.h"
#include "Filter5x5_Input14x14_Stride1.h"
#include "Filter5x5_Input14x14_Stride2.h"
#include "Filter5x5_Input28x28_Stride1.h"
#include "Filter5x5_Input56x56_Stride2.h"

using namespace std;

/*
Hip and MIOpen Error Handling

checkHip(err) - to check if an HIP API call returned some error.
checkKernel() - to check if the kernel invocation is failed.
*/
#define checkHip(err) __checkHip(err, __FILE__, __LINE__)
#define checkKernel() __checkKernel(__FILE__, __LINE__)

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

/*
To test depthwise convolution kernels.
*/
int main(int argc, char* argv[]) {
	// Input dimension
	int inputBatchNumber = 0;
	int inputChannel = 0;
	int inputHeight = 0;
	int inputWidth = 0;

	// Filter dimension
	int filterLayerNumber = 0;
	int filterChannel = 0;
	int filterHeight = 0;
	int filterWidth = 0;

	// Output dimension
	int outputBatchNumber = 0;
	int outputChannel = 0;
	int outputHeight = 0;
	int outputWidth = 0;

	// padding on height and width
	int paddingHeight = 0;
	int paddingWidth = 0;

	// stride
	int stride = 1;

	float alpha = 1.0;
	float beta = 0.0;

	// Initialize all required parameters
	// Input dimensions
	inputBatchNumber = atoi(argv[1]);
	inputChannel = atoi(argv[2]);
	inputHeight = atoi(argv[3]);
	inputWidth = inputHeight;           // Assume that inputs are square

	// Filter dimensions
	filterLayerNumber = inputChannel;
	filterChannel = 1;
	filterHeight = atoi(argv[4]);
	filterWidth = filterHeight;         // Assume that filters are square

	// Padding size
	if (filterWidth == 3) {
		paddingHeight = 1;
		paddingWidth = 1;
	}
	else if (filterWidth == 5) {
		paddingHeight = 2;
		paddingWidth = 2;
	}

	// Stride
	stride = atoi(argv[5]);

	// Output dimensions
	outputBatchNumber = inputBatchNumber;
	outputChannel = inputChannel;
	outputHeight = (inputHeight + paddingHeight * 2 - filterHeight) / stride + 1;
	outputWidth = (inputWidth + paddingWidth * 2 - filterWidth) / stride + 1;

	// Data size
	int inputSize = inputBatchNumber * inputChannel * inputHeight * inputWidth;
	int filterSize = filterLayerNumber * filterChannel * filterHeight * filterWidth;
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

	// Kernel Invocation
	if (stride == 1) {
		if (filterHeight == 3) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				hipEventRecord(start);
				Filter3x3_Input7x7_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 16);
				dim3 blockSize(14 * 16, 1);
				hipEventRecord(start);
				Filter3x3_Input14x14_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(28 * 8, 1);
				hipEventRecord(start);
				Filter3x3_Input28x28_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel);
				dim3 blockSize(4 * 56, 1);
				hipEventRecord(start);
				Filter3x3_Input56x56_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 112) {
				dim3 gridSize(outputBatchNumber, outputChannel * 4);
				dim3 blockSize(2 * 112, 1);
				hipEventRecord(start);
				Filter3x3_Input112x112_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
		else if (filterHeight == 5) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				hipEventRecord(start);
				Filter5x5_Input7x7_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 16);
				dim3 blockSize(14 * 16, 1);
				hipEventRecord(start);
				Filter5x5_Input14x14_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(28 * 8, 1);
				hipEventRecord(start);
				Filter5x5_Input28x28_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
	}
	else if (stride == 2) {
		if (filterHeight == 3) {
			if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				hipEventRecord(start);
				Filter3x3_Input14x14_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(14 * 8, 1);
				hipEventRecord(start);
				Filter3x3_Input28x28_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel / 2);
				dim3 blockSize(28 * 2, 1);
				hipEventRecord(start);
				Filter3x3_Input56x56_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 112) {
				dim3 gridSize(outputBatchNumber, outputChannel * 2);
				dim3 blockSize(56 * 4, 1);
				hipEventRecord(start);
				Filter3x3_Input112x112_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
		else if (filterHeight == 5) {
			if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				hipEventRecord(start);
				Filter5x5_Input14x14_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				hipEventRecord(stop);
				hipEventSynchronize(stop);
				hipEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel / 2);
				dim3 blockSize(28 * 2, 1);
				hipEventRecord(start);
				Filter5x5_Input56x56_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
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
    miopenSet4dTensorDescriptor(filterDesc, miopenFloat, filterLayerNumber, filterChannel, filterHeight, filterWidth);

    // output descriptor
    miopenTensorDescriptor_t outputDesc;
    miopenCreateTensorDescriptor(&outputDesc);
    miopenSet4dTensorDescriptor(outputDesc, miopenFloat, outputBatchNumber, outputChannel, outputHeight, outputWidth);
    
    // convolution descriptor
    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    
    miopenInitConvolutionDescriptor(convDesc,miopenConvolution, paddingHeight, paddingWidth, stride, stride, 1, 1);
    miopenSetConvolutionGroupCount(convDesc, inputChannel);

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
        filterDesc, deviceFilter,
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
    hipEventElapsedTime(&elapsedTime, start, stop);\
    miopenTime = elapsedTime;

    // Copy MIOpen result from device to host
    checkHip(hipMemcpy(hostMiopenOutput, deviceMiopenOutput, outputSize * sizeof(float), hipMemcpyDeviceToHost));

    // Compare Kernel result and MIOpen result
    if (compareOutput(outputBatchNumber, outputChannel, outputHeight, outputWidth, hostKernelOutput, hostMiopenOutput, 1) == 0) {
		printf("Kernel Calculation Correct.\n");
		printf("MIOpen time : %f ms.\n", miopenTime);
		printf("Kernel time : %f ms.\n", kernelTime);
    }

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
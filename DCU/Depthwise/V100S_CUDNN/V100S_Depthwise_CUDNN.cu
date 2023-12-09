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

/*
CUDNN Error Handling

checkCuda(err)  - to check if an CUDA API call returned some error.
checkCudnn(err) - to check if an CUDNN API call returned some error.
*/
#define checkCuda(err) __checkCuda(err, __FILE__, __LINE__)
#define checkCudnn(err) __checkCudnn(err, __FILE__, __LINE__)

inline void __checkCuda(cudaError_t err, const char* file, const int line) {
	if (cudaSuccess != err) {
		printf("checkCuda() failed at %s : %i : %s\n", file, line, cudaGetErrorString(err));
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
* warmup()
* To get GPU initialization ready
*/
__global__ void warmup() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

/*
To test cuDNN depthwise convolution.
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

	// allocate host memory and device memory for Cudnn output data
	float* hostCudnnOutput = (float*)malloc(outputSize * sizeof(float));
	float* deviceCudnnOutput;
	checkCuda(cudaMalloc((void**)&deviceCudnnOutput, outputSize * sizeof(float)));

	// Use Cuda event to measure running time
	float elapsedTime = 0.0;
	float cudnnTime = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Create cudnn
	cudnnHandle_t cudnn;
	checkCudnn(cudnnCreate(&cudnn));

	// input descriptor
	cudnnTensorDescriptor_t inputDesc;
	checkCudnn(cudnnCreateTensorDescriptor(&inputDesc));
	checkCudnn(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inputBatchNumber, inputChannel, inputHeight, inputWidth));

	// filter descriptor
	cudnnFilterDescriptor_t filterDesc;
	checkCudnn(cudnnCreateFilterDescriptor(&filterDesc));
	checkCudnn(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterLayerNumber, filterChannel, filterHeight, filterWidth));

	// output descriptor
	cudnnTensorDescriptor_t outputDesc;
	checkCudnn(cudnnCreateTensorDescriptor(&outputDesc));
	checkCudnn(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outputBatchNumber, outputChannel, outputHeight, outputWidth));

	// convolution descriptor
	cudnnConvolutionDescriptor_t convDesc;
	checkCudnn(cudnnCreateConvolutionDescriptor(&convDesc));
	// dilation is 1
	checkCudnn(cudnnSetConvolution2dDescriptor(convDesc, paddingHeight, paddingWidth, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	checkCudnn(cudnnSetConvolutionGroupCount(convDesc, inputChannel));

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

	// Measure cuDNN time
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
	cudnnTime = elapsedTime;
	printf("cuDNN Calculation Finished.\n");
	printf("cuDNN time : %f ms.\n", cudnnTime);

	// Copy Cudnn result from device to host
	checkCuda(cudaMemcpy(hostCudnnOutput, deviceCudnnOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

	free(hostInput);
	free(hostFilter);
	free(hostCudnnOutput);

	cudaFree(deviceInput);
	cudaFree(deviceFilter);
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
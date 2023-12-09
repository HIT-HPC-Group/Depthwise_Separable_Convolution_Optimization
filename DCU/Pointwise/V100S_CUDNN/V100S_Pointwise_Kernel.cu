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
* To get GPU initialization ready
*/
__global__ void warmup() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

/*
To test Pointwise cuDNN.
*/
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

    // measure cudnn time
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

    // Free all allocated memory spaces
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
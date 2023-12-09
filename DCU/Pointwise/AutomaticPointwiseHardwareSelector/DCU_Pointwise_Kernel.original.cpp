#include "mykernel.h"
#include <thread>
#include <fstream>
#include <chrono>
using namespace std;

/*
CUDA and CUDNN Error Handling

checkCuda(err)  - to check if an CUDA API call returned some error.
checkKernel()   - to check if the kernel invocation is failed.
checkCudnn(err) - to check if an CUDNN API call returned some error.
*/
#define checkCuda(err) __checkCuda(err, __FILE__, __LINE__)
#define checkKernel() __checkKernel(__FILE__, __LINE__)
#define checkCudnn(err) __checkCudnn(err, __FILE__, __LINE__)

inline void __checkCuda(hipError_t err, const char* file, const int line) {
    if (hipSuccess != err) {
        printf("checkCuda() failed at %s : %i : %s\n", file, line, hipGetErrorString(err));
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

inline void __checkCudnn(miopenStatus_t err, const char* file, const int line) {
    if (miopenStatusSuccess != err) {
        printf("checkCudnn() failed at %s : %i : %s\n", file, line,miopenGetErrorString(err));
        exit(-1);
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
                        //printLayer(n,c,h,w,kernelOutput);
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
To test Pointwise convolution kernels.
*/
int main(int argc, char* argv[]) {
    // GPU warm up for benchmarking
    hipLaunchKernelGGL(warmup, 128, 128, 0, 0);

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
    checkCuda(hipMalloc((void**)&deviceInput, inputSize * sizeof(float)));
    checkCuda(hipMemcpy(deviceInput, hostInput, inputSize * sizeof(float), hipMemcpyHostToDevice));

    // allocate host memory and device memory for filter data, and copy it from host to device.
    float* hostFilter = (float*)malloc(filterSize * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < filterSize; i++) {
        hostFilter[i] = (float)(float(rand()) / float((RAND_MAX)) * 5.0);
    }

    float* deviceFilter;
    checkCuda(hipMalloc((void**)&deviceFilter, filterSize * sizeof(float)));
    checkCuda(hipMemcpy(deviceFilter, hostFilter, filterSize * sizeof(float), hipMemcpyHostToDevice));

    // allocate host memory and device memory for kernel output data
    float* hostKernelOutput = (float*)malloc(outputSize * sizeof(float));

    float* deviceKernelOutput;
    checkCuda(hipMalloc((void**)&deviceKernelOutput, outputSize * sizeof(float)));

    // allocate host memory and device memory for Cudnn output data
    float* hostCudnnOutput = (float*)malloc(outputSize * sizeof(float));
    float* deviceCudnnOutput;
    checkCuda(hipMalloc((void**)&deviceCudnnOutput, outputSize * sizeof(float)));

    // Use Cuda event to measure running time
    float elapsedTime = 0.0;
    float kernelTime = 0.0;
    float cudnnTime = 0.0;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    int warpSize = 64;

    // change parameters
    int warpNumPerBlock = 4;
   	int outputWidthPerWarp = 56;
    int outputHeightPerWarp = 16;

    kernelTime = 0.0;
    // Convolution
    dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (outputWidthPerWarp * outputHeightPerWarp * warpNumPerBlock));
    dim3 blockSize(warpNumPerBlock * warpSize);
    // change kernel name
    hipEventRecord(start);
    InputBatch_8_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,
        inputBatchNumber, inputChannel, inputHeight, inputWidth,
        filterOutChannel, filterInChannel, filterHeight, filterWidth,
        outputBatchNumber, outputChannel, outputHeight, outputWidth);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime;
    this_thread::sleep_for(chrono::seconds(1));

    printf("Elapsed Time for Pointwise Convolution Input Batch %d, Input %d x %d, Input Channel %d, Ouput Channel %d: %f ms.\n", 
    inputBatchNumber, inputHeight, inputWidth, inputChannel, outputChannel, kernelTime);

    // Copy kernel output from device to host
    checkCuda(hipMemcpy(hostKernelOutput, deviceKernelOutput, outputSize * sizeof(float), hipMemcpyDeviceToHost));

    // Create cudnn
    miopenHandle_t handle;
    miopenCreate(&handle);
    
    // input descriptor
    miopenTensorDescriptor_t inputDesc;
    miopenCreateTensorDescriptor(&inputDesc);
    miopenSet4dTensorDescriptor(inputDesc,miopenFloat,inputBatchNumber, inputChannel, inputHeight, inputWidth);
    
    // filter descriptor
    miopenTensorDescriptor_t filterDesc;
    miopenCreateTensorDescriptor(&filterDesc);
    miopenSet4dTensorDescriptor(filterDesc,miopenFloat, filterOutChannel, filterInChannel, filterHeight, filterWidth);
    

    // output descriptor
    miopenTensorDescriptor_t outputDesc;
    miopenCreateTensorDescriptor(&outputDesc);
    miopenSet4dTensorDescriptor(outputDesc, miopenFloat, outputBatchNumber, outputChannel, outputHeight, outputWidth);
    
    // convolution descriptor
    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    
    miopenInitConvolutionDescriptor(convDesc,miopenConvolution, 0, 0, 1, 1, 1, 1);
    
    // create workspace
    size_t workspaceSize = 0;
    void* workspaceData = nullptr;
    miopenConvolutionForwardGetWorkSpaceSize(handle, inputDesc, filterDesc, convDesc, outputDesc, &workspaceSize);
    checkCuda(hipMalloc(&workspaceData, workspaceSize));

    // set algorithm
    int returnedAlgoCount = 0;
    miopenConvAlgoPerf_t *miopenPerfResults = new miopenConvAlgoPerf_t[1];

    miopenFindConvolutionForwardAlgorithm(
        handle, inputDesc, deviceInput,
        filterDesc,deviceFilter,
        convDesc,
        outputDesc, deviceCudnnOutput, 1,
        &returnedAlgoCount, miopenPerfResults, workspaceData,
        workspaceSize, false  // exhaustiveSearch
        );

    // Use CUDNN to check kernel result and measure running time
    std::cout<<"miopenConvolutionFwdAlgo_t algo = "<<miopenPerfResults->fwd_algo<<std::endl;
   	hipEvent_t miopen_start, miopen_stop;
    hipEventCreate(&miopen_start);
    hipEventCreate(&miopen_stop);
    hipEventRecord(miopen_start);
    miopenConvolutionForward(
	    handle, &alpha, inputDesc, deviceInput,
        filterDesc, deviceFilter,
        convDesc, miopenPerfResults->fwd_algo, &beta,
        outputDesc, deviceCudnnOutput, workspaceData,
        workspaceSize);
    hipEventRecord(miopen_stop);
    hipEventSynchronize(miopen_stop);
    hipEventElapsedTime(&elapsedTime, miopen_start, miopen_stop);
    printf("Elapsed Time for MIOpen Pointwise Convolution: %f ms.\n", elapsedTime);
    cudnnTime = elapsedTime;

    // Copy Cudnn result from device to host
    checkCuda(hipMemcpy(hostCudnnOutput, deviceCudnnOutput, outputSize * sizeof(float), hipMemcpyDeviceToHost));

    // Compare Kernel result and Cudnn result
    if (compareOutput(outputBatchNumber, outputChannel, outputHeight, outputWidth, hostKernelOutput, hostCudnnOutput, 1) == 0) {
        printf("Kernel Calculation Correct.\n");
        ofstream output("output.txt");
        output << kernelTime << " " << cudnnTime;
        output.close();
    }

    // Free all allocated memory spaces
    free(hostInput);
    free(hostFilter);
    free(hostKernelOutput);
    free(hostCudnnOutput);

    hipFree(deviceInput);
    hipFree(deviceFilter);
    hipFree(deviceKernelOutput);
    hipFree(deviceCudnnOutput);

    miopenDestroy(handle);
    miopenDestroyTensorDescriptor(inputDesc);
    miopenDestroyTensorDescriptor(outputDesc);
    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(filterDesc);
    hipFree(workspaceData);

    checkCuda(hipDeviceReset());
    return 0;
}

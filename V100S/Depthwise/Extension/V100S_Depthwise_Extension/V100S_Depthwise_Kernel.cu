#include <torch/extension.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 7 x 7, stride 1, padding 1

The number of channel must be multiple of 32.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1) 7 x 7 x 960 -> 7 x 7 x 960, stride = 1, filter = 3
	2) 7 x 7 x 1152 -> 7 x 7 x 1152, stride = 1, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input7x7_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 32 channels is a group.
	__shared__ float filterData[32 * 9];	// filter is 3 x 3 = 9
	__shared__ float inputData[32 * 7 * 9]; // original input is 7 x 7, padded to be 9 x 9. ignore up and bottom padding, so 7 x 9

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 32;
	int blockSize = blockDim.x * blockDim.y;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	// load rest of the filter value. 9 * 32 in total
	if (threadIdx.x < 9 * 32 - blockSize) {
		filterData[blockSize + threadIdx.x] = filter[blockSize + filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + paddedWidth - 1] = 0; // right side padding
	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 32 * 9 * 1] = input[inputLoadSrcIdx + 32 * 7 * 1];
	inputData[inputLoadDstIdx + 32 * 9 * 2] = input[inputLoadSrcIdx + 32 * 7 * 2];
	inputData[inputLoadDstIdx + 32 * 9 * 3] = input[inputLoadSrcIdx + 32 * 7 * 3];
	inputData[inputLoadDstIdx + 32 * 9 * 4] = input[inputLoadSrcIdx + 32 * 7 * 4];
	inputData[inputLoadDstIdx + 32 * 9 * 5] = input[inputLoadSrcIdx + 32 * 7 * 5];
	inputData[inputLoadDstIdx + 32 * 9 * 6] = input[inputLoadSrcIdx + 32 * 7 * 6];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 2 times:
	// 		1. filter's 2nd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 2nd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	// 2nd row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 2nd row of input)
	//		3. filter's 1st row (when filter is sliding through the 3rd row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 3rd row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 2nd row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 3rd row of input)
	//		3. filter's 1st row (when filter is sliding through the 4th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 4th row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 3rd row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 4th row of input)
	//		3. filter's 1st row (when filter is sliding through the 5th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 5th row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 4th row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 5th row of input)
	//		3. filter's 1st row (when filter is sliding through the 6th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 6th row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 5th row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 6th row of input)
	//		3. filter's 1st row (when filter is sliding through the 7th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 7th row
	// convolve with filter 2 times:
	// 		1. filter's 3rd row (when filter is sliding through the 6th row of input)
	//		2. filter's 2nd row (when filter is sliding through the 7th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum0 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 7 x 7, stride 1, padding 2

The number of channel must be multiple of 32.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1) 7 x 7 x 1152 -> 7 x 7 x 1152, stride = 1, fitler = 5
*/
template <typename scalar_t>
__global__ void Filter5x5_Input7x7_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 32 channels is a group.
	__shared__ float filterData[32 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[32 * 7 * 11]; // original input is 7 x 7, padded to be 11 x 11. ignore up and bottom padding, so 7 x 11

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2, sum3, sum4;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 32;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	filterData[threadIdx.x + 32 * 7] = filter[filterLoadSrcIdx + 32 * 7];
	filterData[threadIdx.x + 32 * 7 * 2] = filter[filterLoadSrcIdx + 32 * 7 * 2];
	// load rest of the filter value. 25 * 32 in total
	if (threadIdx.x < 25 * 32 - 3 * 32 * 7) {
		filterData[32 * 7 * 3 + threadIdx.x] = filter[32 * 7 * 3 + filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;
	inputData[leftPaddingIdx + 9] = 0; // right side padding
	inputData[leftPaddingIdx + 10] = 0; // right side padding
	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 32 * 11 * 1] = input[inputLoadSrcIdx + 32 * 7 * 1];
	inputData[inputLoadDstIdx + 32 * 11 * 2] = input[inputLoadSrcIdx + 32 * 7 * 2];
	inputData[inputLoadDstIdx + 32 * 11 * 3] = input[inputLoadSrcIdx + 32 * 7 * 3];
	inputData[inputLoadDstIdx + 32 * 11 * 4] = input[inputLoadSrcIdx + 32 * 7 * 4];
	inputData[inputLoadDstIdx + 32 * 11 * 5] = input[inputLoadSrcIdx + 32 * 7 * 5];
	inputData[inputLoadDstIdx + 32 * 11 * 6] = input[inputLoadSrcIdx + 32 * 7 * 6];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / inputWidth) * paddedWidth * inputHeight + threadIdx.x % inputWidth;
	int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 5] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 4 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	// 		2. filter's 3rd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 2nd row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 4th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;
	sum3 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 4] * inTemp4;

	// 3rd row
	// convolve with filter 5 times:
	//		1. filter's 5th row (when filter is sliding through the 1st row of input)
	// 		2. filter's 4th row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 3rd row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 4th row of input) 
	//		5. filter's 1st row (when filter is sliding through the 5th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 5] * inTemp0;
	sum4 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 6] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 7] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 8] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 9] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 4th row
	// convolve with filter 5 times:
	//		1. filter's 5th row (when filter is sliding through the 2nd row of input)
	// 		2. filter's 4th row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 4th row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 5th row of input) 
	//		5. filter's 1st row (when filter is sliding through the 6th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 10] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 5] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 11] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 6] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 12] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 7] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 13] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 8] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 14] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 9] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 5th row
	// convolve with filter 5 times:
	//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
	// 		2. filter's 4th row (when filter is sliding through the 4th row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 5th row of input) 
	// 		4. filter's 2nd row (when filter is sliding through the 6th row of input) 
	//		5. filter's 1st row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 15] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 16] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 17] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 18] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 19] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 6th row
	// convolve with filter 4 times:
	//		1. filter's 5th row (when filter is sliding through the 4th row of input)
	//		2. filter's 4th row (when filter is sliding through the 5th row of input)
	// 		3. filter's 3rd row (when filter is sliding through the 6th row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 20] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 21] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 22] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 23] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 24] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;

	output[outputIdx] = sum3 * alpha + beta;
	outputIdx += outputWidth;

	// 7th row
	// convolve with filter 3 times:
	// 		1. filter's 5th row (when filter is sliding through the 5th row of input) 
	//		2. filter's 4th row (when filter is sliding through the 6th row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;

	output[outputIdx] = sum4 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum1 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 14 x 14, stride 1, padding 1

The number of channel must be multiple of 16.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	14 x 14 x 384 -> 14 x 14 x 384, stride = 1, filter = 3
	2)	14 x 14 x 480 -> 14 x 14 x 480, stride = 1, filter = 3
	3)	14 x 14 x 576 -> 14 x 14 x 576, stride = 1, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input14x14_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 16 channels is a group.
	__shared__ float filterData[16 * 9];	// filter is 3 x 3 = 9
	__shared__ float inputData[16 * 14 * 16]; // original input is 14 x 14, padded to be 16 x 16. ignore up and bottom padding, so 14 x 16

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 16;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 16 * 9) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 15] = 0; // right side padding

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 16 * 16 * 1] = input[inputLoadSrcIdx + 16 * 14 * 1];
	inputData[inputLoadDstIdx + 16 * 16 * 2] = input[inputLoadSrcIdx + 16 * 14 * 2];
	inputData[inputLoadDstIdx + 16 * 16 * 3] = input[inputLoadSrcIdx + 16 * 14 * 3];
	inputData[inputLoadDstIdx + 16 * 16 * 4] = input[inputLoadSrcIdx + 16 * 14 * 4];
	inputData[inputLoadDstIdx + 16 * 16 * 5] = input[inputLoadSrcIdx + 16 * 14 * 5];
	inputData[inputLoadDstIdx + 16 * 16 * 6] = input[inputLoadSrcIdx + 16 * 14 * 6];
	inputData[inputLoadDstIdx + 16 * 16 * 7] = input[inputLoadSrcIdx + 16 * 14 * 7];
	inputData[inputLoadDstIdx + 16 * 16 * 8] = input[inputLoadSrcIdx + 16 * 14 * 8];
	inputData[inputLoadDstIdx + 16 * 16 * 9] = input[inputLoadSrcIdx + 16 * 14 * 9];
	inputData[inputLoadDstIdx + 16 * 16 * 10] = input[inputLoadSrcIdx + 16 * 14 * 10];
	inputData[inputLoadDstIdx + 16 * 16 * 11] = input[inputLoadSrcIdx + 16 * 14 * 11];
	inputData[inputLoadDstIdx + 16 * 16 * 12] = input[inputLoadSrcIdx + 16 * 14 * 12];
	inputData[inputLoadDstIdx + 16 * 16 * 13] = input[inputLoadSrcIdx + 16 * 14 * 13];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % inputWidth;
	int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 2 times:
	// 		1. filter's 2nd row (when filter is sliding through the 1st row of input) 
	//		2. filter's 1st row (when filter is sliding through the 2nd row of input) 
	inTemp0 = inputData[inputAccessBase];
	sum0 = filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	// 2nd row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 3rd row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 3rd row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 2nd row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 4th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 4th row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 3rd row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 4th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 5th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 5th row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 4th row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 5th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 6th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 6th row
	// convolve with filter 3 times:
	//		1. filter's 3rd row (when filter is sliding through the 5th row of input)
	// 		2. filter's 2nd row (when filter is sliding through the 6th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 7th row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 6th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 7th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 8th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 8th row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 7th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 8th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 9th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 9th row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 8th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 9th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 10th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 10th row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 9th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 10th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 11th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 11st row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 10th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 11th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 12th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 12nd row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 11th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 12th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 13rd row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 12th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 13th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 14th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 14th row
	// convolve with filter 2 times:
	//		1. filter's 2nd row (when filter is sliding through the 13th row of input) 
	//		2. filter's 1st row (when filter is sliding through the 14th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum1 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 14 x 14, stride 2, padding 1

The number of channel must be multiple of 32.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	14 x 14 x 576 -> 14 x 14 x 576, stride = 2, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input14x14_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 32 channels is a group.
	__shared__ float filterData[32 * 9];	// filter is 3 x 3 = 9
	__shared__ float inputData[32 * 14 * 16]; // original input is 14 x 14, padded to be 16 x 16. ignore up and bottom padding, so 14 x 16

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 32;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	// load rest of the filter value. 9 * 32 in total
	if (threadIdx.x < 9 * 32 - 7 * 32) {
		filterData[7 * 32 + threadIdx.x] = filter[7 * 32 + filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;		// left padding upper half
	inputData[leftPaddingIdx + 16 * inputHeight * paddedWidth] = 0; // left padding bottom half
	inputData[leftPaddingIdx + 15] = 0; // right padding upper half
	inputData[leftPaddingIdx + 16 * inputHeight * paddedWidth + 15] = 0; // right padding bottom half

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 16 * 16 * 1] = input[inputLoadSrcIdx + 16 * 14 * 1];
	inputData[inputLoadDstIdx + 16 * 16 * 2] = input[inputLoadSrcIdx + 16 * 14 * 2];
	inputData[inputLoadDstIdx + 16 * 16 * 3] = input[inputLoadSrcIdx + 16 * 14 * 3];
	inputData[inputLoadDstIdx + 16 * 16 * 4] = input[inputLoadSrcIdx + 16 * 14 * 4];
	inputData[inputLoadDstIdx + 16 * 16 * 5] = input[inputLoadSrcIdx + 16 * 14 * 5];
	inputData[inputLoadDstIdx + 16 * 16 * 6] = input[inputLoadSrcIdx + 16 * 14 * 6];
	inputData[inputLoadDstIdx + 16 * 16 * 7] = input[inputLoadSrcIdx + 16 * 14 * 7];
	inputData[inputLoadDstIdx + 16 * 16 * 8] = input[inputLoadSrcIdx + 16 * 14 * 8];
	inputData[inputLoadDstIdx + 16 * 16 * 9] = input[inputLoadSrcIdx + 16 * 14 * 9];
	inputData[inputLoadDstIdx + 16 * 16 * 10] = input[inputLoadSrcIdx + 16 * 14 * 10];
	inputData[inputLoadDstIdx + 16 * 16 * 11] = input[inputLoadSrcIdx + 16 * 14 * 11];
	inputData[inputLoadDstIdx + 16 * 16 * 12] = input[inputLoadSrcIdx + 16 * 14 * 12];
	inputData[inputLoadDstIdx + 16 * 16 * 13] = input[inputLoadSrcIdx + 16 * 14 * 13];
	inputData[inputLoadDstIdx + 16 * 16 * 14] = input[inputLoadSrcIdx + 16 * 14 * 14];
	inputData[inputLoadDstIdx + 16 * 16 * 15] = input[inputLoadSrcIdx + 16 * 14 * 15];
	inputData[inputLoadDstIdx + 16 * 16 * 16] = input[inputLoadSrcIdx + 16 * 14 * 16];
	inputData[inputLoadDstIdx + 16 * 16 * 17] = input[inputLoadSrcIdx + 16 * 14 * 17];
	inputData[inputLoadDstIdx + 16 * 16 * 18] = input[inputLoadSrcIdx + 16 * 14 * 18];
	inputData[inputLoadDstIdx + 16 * 16 * 19] = input[inputLoadSrcIdx + 16 * 14 * 19];
	inputData[inputLoadDstIdx + 16 * 16 * 20] = input[inputLoadSrcIdx + 16 * 14 * 20];
	inputData[inputLoadDstIdx + 16 * 16 * 21] = input[inputLoadSrcIdx + 16 * 14 * 21];
	inputData[inputLoadDstIdx + 16 * 16 * 22] = input[inputLoadSrcIdx + 16 * 14 * 22];
	inputData[inputLoadDstIdx + 16 * 16 * 23] = input[inputLoadSrcIdx + 16 * 14 * 23];
	inputData[inputLoadDstIdx + 16 * 16 * 24] = input[inputLoadSrcIdx + 16 * 14 * 24];
	inputData[inputLoadDstIdx + 16 * 16 * 25] = input[inputLoadSrcIdx + 16 * 14 * 25];
	inputData[inputLoadDstIdx + 16 * 16 * 26] = input[inputLoadSrcIdx + 16 * 14 * 26];
	inputData[inputLoadDstIdx + 16 * 16 * 27] = input[inputLoadSrcIdx + 16 * 14 * 27];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth + threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth * 2;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 1st row of input) 
	inTemp0 = inputData[inputAccessBase];
	sum0 = filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	// 2nd row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 3rd row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 3rd row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 3rd row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	// 4th row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 3rd row of input)
	//		2. filter's 1st row (when filter is sliding through the 5th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 5th row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 5th row of input)
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	// 6th row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 5th row of input)
	//		2. filter's 1st row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 7th row
	// convolve with filter 1 time:
	//		1. filter's 2nd row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	// 8th row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 7th row of input) 
	//		2. filter's 1nd row (when filter is sliding through the 9th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 9th row
	// convolve with filter 1 time:
	//		1. filter's 2nd row (when filter is sliding through the 9th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	// 10th row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 9th row of input) 
	//		2. filter's 1nd row (when filter is sliding through the 11th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 11st row
	// convolve with filter 1 time:
	//		1. filter's 2nd row (when filter is sliding through the 11th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	// 12nd row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 11th row of input) 
	//		2. filter's 1nd row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 13rd row
	// convolve with filter 1 time:
	//		1. filter's 2nd row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	// 14th row
	// convolve with filter 1 time:
	//		1. filter's 3rd row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 14 x 14, stride 1, padding 2

The number of channel must be multiple of 16.
Used in the MobileNet V2 and EfficientNet B0, in case of.
	1)	14 x 14 x 480 -> 14 x 14 x 480, stride = 1, filter = 5
	2)	14 x 14 x 672 -> 14 x 14 x 672, stride = 1, filter = 5

*/
template <typename scalar_t>
__global__ void Filter5x5_Input14x14_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 16 channels is a group.
	__shared__ float filterData[16 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[16 * 14 * 18]; // original input is 14 x 14, padded to be 18 x 18. ignore up and bottom padding, so 14 x 18

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2, sum3, sum4;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 16;
	// int blockSize = blockDim.x * blockDim.y;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 8 * 25) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
		filterData[threadIdx.x + 8 * 25] = filter[filterLoadSrcIdx + 8 * 25];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;
	inputData[leftPaddingIdx + 16] = 0; // right side padding
	inputData[leftPaddingIdx + 17] = 0; // right side padding

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 16 * 18 * 1] = input[inputLoadSrcIdx + 16 * 14 * 1];
	inputData[inputLoadDstIdx + 16 * 18 * 2] = input[inputLoadSrcIdx + 16 * 14 * 2];
	inputData[inputLoadDstIdx + 16 * 18 * 3] = input[inputLoadSrcIdx + 16 * 14 * 3];
	inputData[inputLoadDstIdx + 16 * 18 * 4] = input[inputLoadSrcIdx + 16 * 14 * 4];
	inputData[inputLoadDstIdx + 16 * 18 * 5] = input[inputLoadSrcIdx + 16 * 14 * 5];
	inputData[inputLoadDstIdx + 16 * 18 * 6] = input[inputLoadSrcIdx + 16 * 14 * 6];
	inputData[inputLoadDstIdx + 16 * 18 * 7] = input[inputLoadSrcIdx + 16 * 14 * 7];
	inputData[inputLoadDstIdx + 16 * 18 * 8] = input[inputLoadSrcIdx + 16 * 14 * 8];
	inputData[inputLoadDstIdx + 16 * 18 * 9] = input[inputLoadSrcIdx + 16 * 14 * 9];
	inputData[inputLoadDstIdx + 16 * 18 * 10] = input[inputLoadSrcIdx + 16 * 14 * 10];
	inputData[inputLoadDstIdx + 16 * 18 * 11] = input[inputLoadSrcIdx + 16 * 14 * 11];
	inputData[inputLoadDstIdx + 16 * 18 * 12] = input[inputLoadSrcIdx + 16 * 14 * 12];
	inputData[inputLoadDstIdx + 16 * 18 * 13] = input[inputLoadSrcIdx + 16 * 14 * 13];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth;
	int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 5] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 4 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	// 		2. filter's 3rd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 2nd row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 4th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;
	sum3 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 4] * inTemp4;

	#pragma unroll
	for (int i = 0; i < 2; i++) {
		// 3rd row, 8th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 1st row of input)
		// 		2. filter's 4th row (when filter is sliding through the 2nd row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 3rd row of input) 
		//		4. filter's 2nd row (when filter is sliding through the 4th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
		sum3 = sum3 + filterData[filterAccessBase + 5] * inTemp0;
		sum4 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 6] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 7] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 8] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 9] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;

		// 4th row, 9th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 2nd row of input)
		// 		2. filter's 4th row (when filter is sliding through the 3rd row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 4th row of input) 
		//		4. filter's 2nd row (when filter is sliding through the 5th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 6th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
		sum3 = sum3 + filterData[filterAccessBase + 10] * inTemp0;
		sum4 = sum4 + filterData[filterAccessBase + 5] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 11] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 6] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 12] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 7] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 13] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 8] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 14] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 9] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		// 5th row, 10th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
		// 		2. filter's 4th row (when filter is sliding through the 4th row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 5th row of input) 
		// 		4. filter's 2nd row (when filter is sliding through the 6th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
		sum3 = sum3 + filterData[filterAccessBase + 15] * inTemp0;
		sum4 = sum4 + filterData[filterAccessBase + 10] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 16] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 11] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 17] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 12] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 18] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 13] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 19] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 14] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += outputWidth;

		// 6th row, 11th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 4th row of input)
		// 		2. filter's 4th row (when filter is sliding through the 5th row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 6th row of input) 
		// 		4. filter's 2nd row (when filter is sliding through the 7th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 8th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 20] * inTemp0;
		sum4 = sum4 + filterData[filterAccessBase + 15] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;
		sum2 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 21] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 16] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 22] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 17] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 23] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 18] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 24] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 19] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum3 * alpha + beta;
		outputIdx += outputWidth;

		// 7th row, 12th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 5th row of input)
		// 		2. filter's 4th row (when filter is sliding through the 6th row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 7th row of input) 
		// 		4. filter's 2nd row (when filter is sliding through the 8th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 9th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 20] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;
		sum3 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 21] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 22] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 23] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 24] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum4 * alpha + beta;
		outputIdx += outputWidth;
	}

	// 13th row
	// convolve with filter 4 times:
	//		1. filter's 5th row (when filter is sliding through the 11th row of input)
	//		2. filter's 4th row (when filter is sliding through the 12th row of input)
	// 		3. filter's 3rd row (when filter is sliding through the 13th row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 14th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 9] * inTemp4;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 14th row
	// convolve with filter 3 times:
	// 		1. filter's 5th row (when filter is sliding through the 12th row of input) 
	//		2. filter's 4th row (when filter is sliding through the 13th row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 14th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 14] * inTemp4;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum3 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 14 x 14, stride 2, padding 2

The number of channel must be multiple of 32.
Used in the MobileNet V2 and EfficientNet B0, in case of.
	1)	14 x 14 x 672 -> 14 x 14 x 672, stride = 2, filter = 5
*/
template <typename scalar_t>
__global__ void Filter5x5_Input14x14_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 32 channels is a group.
	__shared__ float filterData[32 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[32 * 14 * 18]; // original input is 14 x 14, padded to be 18 x 18. ignore up and bottom padding, so 14 x 18

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 32;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 8 * 25) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];	// 8 * 25
		filterData[threadIdx.x + 8 * 25] = filter[filterLoadSrcIdx + 8 * 25]; // 16 * 25
		filterData[threadIdx.x + 8 * 25 * 2] = filter[filterLoadSrcIdx + 8 * 25 * 2]; // 24 * 25
		filterData[threadIdx.x + 8 * 25 * 3] = filter[filterLoadSrcIdx + 8 * 25 * 3]; // 32 * 25, filter loaded
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	// Upper half, left side
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;
	// Upper half, right side
	inputData[leftPaddingIdx + 16] = 0;
	inputData[leftPaddingIdx + 17] = 0;

	// Bottom half, left side
	inputData[leftPaddingIdx + 16 * inputHeight * paddedWidth] = 0;
	inputData[leftPaddingIdx + 16 * inputHeight * paddedWidth + 1] = 0;
	// Bottom half, right side
	inputData[leftPaddingIdx + 16 * inputHeight * paddedWidth + 16] = 0;
	inputData[leftPaddingIdx + 16 * inputHeight * paddedWidth + 17] = 0;
	__syncthreads();


	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 16 * 18 * 1] = input[inputLoadSrcIdx + 16 * 14 * 1];
	inputData[inputLoadDstIdx + 16 * 18 * 2] = input[inputLoadSrcIdx + 16 * 14 * 2];
	inputData[inputLoadDstIdx + 16 * 18 * 3] = input[inputLoadSrcIdx + 16 * 14 * 3];
	inputData[inputLoadDstIdx + 16 * 18 * 4] = input[inputLoadSrcIdx + 16 * 14 * 4];
	inputData[inputLoadDstIdx + 16 * 18 * 5] = input[inputLoadSrcIdx + 16 * 14 * 5];
	inputData[inputLoadDstIdx + 16 * 18 * 6] = input[inputLoadSrcIdx + 16 * 14 * 6];
	inputData[inputLoadDstIdx + 16 * 18 * 7] = input[inputLoadSrcIdx + 16 * 14 * 7];
	inputData[inputLoadDstIdx + 16 * 18 * 8] = input[inputLoadSrcIdx + 16 * 14 * 8];
	inputData[inputLoadDstIdx + 16 * 18 * 9] = input[inputLoadSrcIdx + 16 * 14 * 9];
	inputData[inputLoadDstIdx + 16 * 18 * 10] = input[inputLoadSrcIdx + 16 * 14 * 10];
	inputData[inputLoadDstIdx + 16 * 18 * 11] = input[inputLoadSrcIdx + 16 * 14 * 11];
	inputData[inputLoadDstIdx + 16 * 18 * 12] = input[inputLoadSrcIdx + 16 * 14 * 12];
	inputData[inputLoadDstIdx + 16 * 18 * 13] = input[inputLoadSrcIdx + 16 * 14 * 13];
	inputData[inputLoadDstIdx + 16 * 18 * 14] = input[inputLoadSrcIdx + 16 * 14 * 14];
	inputData[inputLoadDstIdx + 16 * 18 * 15] = input[inputLoadSrcIdx + 16 * 14 * 15];
	inputData[inputLoadDstIdx + 16 * 18 * 16] = input[inputLoadSrcIdx + 16 * 14 * 16];
	inputData[inputLoadDstIdx + 16 * 18 * 17] = input[inputLoadSrcIdx + 16 * 14 * 17];
	inputData[inputLoadDstIdx + 16 * 18 * 18] = input[inputLoadSrcIdx + 16 * 14 * 18];
	inputData[inputLoadDstIdx + 16 * 18 * 19] = input[inputLoadSrcIdx + 16 * 14 * 19];
	inputData[inputLoadDstIdx + 16 * 18 * 20] = input[inputLoadSrcIdx + 16 * 14 * 20];
	inputData[inputLoadDstIdx + 16 * 18 * 21] = input[inputLoadSrcIdx + 16 * 14 * 21];
	inputData[inputLoadDstIdx + 16 * 18 * 22] = input[inputLoadSrcIdx + 16 * 14 * 22];
	inputData[inputLoadDstIdx + 16 * 18 * 23] = input[inputLoadSrcIdx + 16 * 14 * 23];
	inputData[inputLoadDstIdx + 16 * 18 * 24] = input[inputLoadSrcIdx + 16 * 14 * 24];
	inputData[inputLoadDstIdx + 16 * 18 * 25] = input[inputLoadSrcIdx + 16 * 14 * 25];
	inputData[inputLoadDstIdx + 16 * 18 * 26] = input[inputLoadSrcIdx + 16 * 14 * 26];
	inputData[inputLoadDstIdx + 16 * 18 * 27] = input[inputLoadSrcIdx + 16 * 14 * 27];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth * 2;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 2 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 2 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	//		2. filter's 2nd row (when filter is sliding through the 3rd row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;

	// 3rd row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 1st row of input)
	//		2. filter's 3rd row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 5th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 4th row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 3rd row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 5th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;

	// 5th row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
	//		2. filter's 3rd row (when filter is sliding through the 5th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 6th row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 5th row of input) 
	// 		2. filter's 2nd row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;

	// 7th row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 5th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 7th row of input)
	//		3. filter's 1st row (when filter is sliding through the 9th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 8th row
	// convolve with filter 2 times:
	//		1. filter's 4th row (when filter is sliding through the 7th row of input)
	//		2. filter's 2nd row (when filter is sliding through the 9th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;

	// 9th row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 7th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 9th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 11th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 10th row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 9th row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 11th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;

	// 11th row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 9th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 11th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 12th row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 11th row of input) 
	// 		2. filter's 2nd row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;

	// 13th row
	// convolve with filter 2 times:
	//		1. filter's 5th row (when filter is sliding through the 11th row of input)
	// 		2. filter's 3rd row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 14th row
	// convolve with filter 1 time:
	//		1. filter's 4th row (when filter is sliding through the 13th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 28 x 28, stride 1, padding 1

The number of channel must be multiple of 8.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	28 x 28 x 240 -> 28 x 28 x 240, stride = 1, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input28x28_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 8 channels is a group.
	__shared__ float filterData[8 * 9];	// filter is 3 x 3 = 9
	__shared__ float inputData[8 * 28 * 30]; // original input is 28 x 28, padded to be 30 x 30. ignore up and bottom padding, so 28 x 30

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 8;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 8 * 9) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 29] = 0; // right side padding

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 8 * 30 * 1] = input[inputLoadSrcIdx + 8 * 28 * 1];
	inputData[inputLoadDstIdx + 8 * 30 * 2] = input[inputLoadSrcIdx + 8 * 28 * 2];
	inputData[inputLoadDstIdx + 8 * 30 * 3] = input[inputLoadSrcIdx + 8 * 28 * 3];
	inputData[inputLoadDstIdx + 8 * 30 * 4] = input[inputLoadSrcIdx + 8 * 28 * 4];
	inputData[inputLoadDstIdx + 8 * 30 * 5] = input[inputLoadSrcIdx + 8 * 28 * 5];
	inputData[inputLoadDstIdx + 8 * 30 * 6] = input[inputLoadSrcIdx + 8 * 28 * 6];
	inputData[inputLoadDstIdx + 8 * 30 * 7] = input[inputLoadSrcIdx + 8 * 28 * 7];
	inputData[inputLoadDstIdx + 8 * 30 * 8] = input[inputLoadSrcIdx + 8 * 28 * 8];
	inputData[inputLoadDstIdx + 8 * 30 * 9] = input[inputLoadSrcIdx + 8 * 28 * 9];
	inputData[inputLoadDstIdx + 8 * 30 * 10] = input[inputLoadSrcIdx + 8 * 28 * 10];
	inputData[inputLoadDstIdx + 8 * 30 * 11] = input[inputLoadSrcIdx + 8 * 28 * 11];
	inputData[inputLoadDstIdx + 8 * 30 * 12] = input[inputLoadSrcIdx + 8 * 28 * 12];
	inputData[inputLoadDstIdx + 8 * 30 * 13] = input[inputLoadSrcIdx + 8 * 28 * 13];
	inputData[inputLoadDstIdx + 8 * 30 * 14] = input[inputLoadSrcIdx + 8 * 28 * 14];
	inputData[inputLoadDstIdx + 8 * 30 * 15] = input[inputLoadSrcIdx + 8 * 28 * 15];
	inputData[inputLoadDstIdx + 8 * 30 * 16] = input[inputLoadSrcIdx + 8 * 28 * 16];
	inputData[inputLoadDstIdx + 8 * 30 * 17] = input[inputLoadSrcIdx + 8 * 28 * 17];
	inputData[inputLoadDstIdx + 8 * 30 * 18] = input[inputLoadSrcIdx + 8 * 28 * 18];
	inputData[inputLoadDstIdx + 8 * 30 * 19] = input[inputLoadSrcIdx + 8 * 28 * 19];
	inputData[inputLoadDstIdx + 8 * 30 * 20] = input[inputLoadSrcIdx + 8 * 28 * 20];
	inputData[inputLoadDstIdx + 8 * 30 * 21] = input[inputLoadSrcIdx + 8 * 28 * 21];
	inputData[inputLoadDstIdx + 8 * 30 * 22] = input[inputLoadSrcIdx + 8 * 28 * 22];
	inputData[inputLoadDstIdx + 8 * 30 * 23] = input[inputLoadSrcIdx + 8 * 28 * 23];
	inputData[inputLoadDstIdx + 8 * 30 * 24] = input[inputLoadSrcIdx + 8 * 28 * 24];
	inputData[inputLoadDstIdx + 8 * 30 * 25] = input[inputLoadSrcIdx + 8 * 28 * 25];
	inputData[inputLoadDstIdx + 8 * 30 * 26] = input[inputLoadSrcIdx + 8 * 28 * 26];
	inputData[inputLoadDstIdx + 8 * 30 * 27] = input[inputLoadSrcIdx + 8 * 28 * 27];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth;
	int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 2 times:
	// 		1. filter's 2nd row (when filter is sliding through the 1st row of input) 
	//		2. filter's 1st row (when filter is sliding through the 2nd row of input) 
	inTemp0 = inputData[inputAccessBase];
	sum0 = filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		// 2nd row, 5th row, 8th row, 11th row, 14th row, 17th row, 20th row, 23rd row
		// convolve with filter 3 times:
		//		1. filter's 3rd row (when filter is sliding through the 1st row of input)
		// 		2. filter's 2nd row (when filter is sliding through the 2nd row of input) 
		//		3. filter's 1st row (when filter is sliding through the 3rd row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
		sum2 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;

		// 3rd row
		// convolve with filter 3 times:
		//		1. filter's 3rd row (when filter is sliding through the 2nd row of input)
		// 		2. filter's 2nd row (when filter is sliding through the 3rd row of input) 
		//		3. filter's 1st row (when filter is sliding through the 4th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		// 4th row, 7th row
		// convolve with filter three times:
		//		1. filter's 3rd row (when filter is sliding through the 3rd row of input)
		// 		2. filter's 2nd row (when filter is sliding through the 4th row of input) 
		//		3. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += outputWidth;
	}

	// 26th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 27th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 28th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum0 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 28 x 28, stride 2, padding 1

The number of channel must be multiple of 8.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	28 x 28 x 192 -> 14 x 14 x 192, stride = 2, filter = 3
	1)	28 x 28 x 240 -> 14 x 14 x 240, stride = 2, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input28x28_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 8 channels is a group.
	__shared__ float filterData[8 * 9];	// filter is 3 x 3 = 9
	__shared__ float inputData[8 * 28 * 30]; // original input is 28 x 28, padded to be 30 x 30. ignore up and bottom padding, so 28 x 30

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 8;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 8 * 9) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 29] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + 29] = 0;

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;	// each thread find its own load destination.

	#pragma unroll
	for (int i = 0; i < 56; i++) {
		inputData[inputLoadDstIdx + 4 * 30 * i] = input[inputLoadSrcIdx + 4 * 28 * i];
	}
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth * 2;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 1st row of input) 
	inTemp0 = inputData[inputAccessBase];
	sum0 = filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	// 2nd row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 3rd row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	#pragma unroll
	for (int i = 0; i < 6; i++) {
		// 3rd row
		// convolve with filter 1 time:
		// 		1. filter's 2nd row (when filter is sliding through the 3rd row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

		// 4th row
		// convolve with filter 2 times:
		//		1. filter's 3rd row (when filter is sliding through the 3rd row of input)
		//		2. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		// 5th row
		// convolve with filter 1 time:
		// 		1. filter's 2nd row (when filter is sliding through the 5th row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

		// 6th row
		// convolve with filter 2 times:
		//		1. filter's 3rd row (when filter is sliding through the 5th row of input)
		//		2. filter's 1st row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;
	}

	// 27th row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 27th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	// 28th row
	// convolve with filter 1 time:
	// 		1. filter's 3rd row (when filter is sliding through the 27th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;
}

/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 28 x 28, stride 1, padding 2

The number of channel must be multiple of 8.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	28 x 28 x 240 -> 28 x 28 x 240, stride = 1, filter = 5
*/
template <typename scalar_t>
__global__ void Filter5x5_Input28x28_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 8 channels is a group.
	__shared__ float filterData[8 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[8 * 28 * 32]; // original input is 28 x 28, padded to be 32 x 32. ignore up and bottom padding, so 28 x 32

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2, sum3, sum4;  // to accumulate the row sum result. rolling recycle.
	// cuuint64_t exchange;

	int channelGroupSize = 8;
	// int blockSize = blockDim.x * blockDim.y;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 8 * 25) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;
	inputData[leftPaddingIdx + 30] = 0; // right side padding
	inputData[leftPaddingIdx + 31] = 0; // right side padding

	__syncthreads();


	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 8 * 32 * 1] = input[inputLoadSrcIdx + 32 * 7 * 1];
	inputData[inputLoadDstIdx + 8 * 32 * 2] = input[inputLoadSrcIdx + 32 * 7 * 2];
	inputData[inputLoadDstIdx + 8 * 32 * 3] = input[inputLoadSrcIdx + 32 * 7 * 3];
	inputData[inputLoadDstIdx + 8 * 32 * 4] = input[inputLoadSrcIdx + 32 * 7 * 4];
	inputData[inputLoadDstIdx + 8 * 32 * 5] = input[inputLoadSrcIdx + 32 * 7 * 5];
	inputData[inputLoadDstIdx + 8 * 32 * 6] = input[inputLoadSrcIdx + 32 * 7 * 6];
	inputData[inputLoadDstIdx + 8 * 32 * 7] = input[inputLoadSrcIdx + 32 * 7 * 7];
	inputData[inputLoadDstIdx + 8 * 32 * 8] = input[inputLoadSrcIdx + 32 * 7 * 8];
	inputData[inputLoadDstIdx + 8 * 32 * 9] = input[inputLoadSrcIdx + 32 * 7 * 9];
	inputData[inputLoadDstIdx + 8 * 32 * 10] = input[inputLoadSrcIdx + 32 * 7 * 10];
	inputData[inputLoadDstIdx + 8 * 32 * 11] = input[inputLoadSrcIdx + 32 * 7 * 11];
	inputData[inputLoadDstIdx + 8 * 32 * 12] = input[inputLoadSrcIdx + 32 * 7 * 12];
	inputData[inputLoadDstIdx + 8 * 32 * 13] = input[inputLoadSrcIdx + 32 * 7 * 13];
	inputData[inputLoadDstIdx + 8 * 32 * 14] = input[inputLoadSrcIdx + 32 * 7 * 14];
	inputData[inputLoadDstIdx + 8 * 32 * 15] = input[inputLoadSrcIdx + 32 * 7 * 15];
	inputData[inputLoadDstIdx + 8 * 32 * 16] = input[inputLoadSrcIdx + 32 * 7 * 16];
	inputData[inputLoadDstIdx + 8 * 32 * 17] = input[inputLoadSrcIdx + 32 * 7 * 17];
	inputData[inputLoadDstIdx + 8 * 32 * 18] = input[inputLoadSrcIdx + 32 * 7 * 18];
	inputData[inputLoadDstIdx + 8 * 32 * 19] = input[inputLoadSrcIdx + 32 * 7 * 19];
	inputData[inputLoadDstIdx + 8 * 32 * 20] = input[inputLoadSrcIdx + 32 * 7 * 20];
	inputData[inputLoadDstIdx + 8 * 32 * 21] = input[inputLoadSrcIdx + 32 * 7 * 21];
	inputData[inputLoadDstIdx + 8 * 32 * 22] = input[inputLoadSrcIdx + 32 * 7 * 22];
	inputData[inputLoadDstIdx + 8 * 32 * 23] = input[inputLoadSrcIdx + 32 * 7 * 23];
	inputData[inputLoadDstIdx + 8 * 32 * 24] = input[inputLoadSrcIdx + 32 * 7 * 24];
	inputData[inputLoadDstIdx + 8 * 32 * 25] = input[inputLoadSrcIdx + 32 * 7 * 25];
	inputData[inputLoadDstIdx + 8 * 32 * 26] = input[inputLoadSrcIdx + 32 * 7 * 26];
	inputData[inputLoadDstIdx + 8 * 32 * 27] = input[inputLoadSrcIdx + 32 * 7 * 27];
	__syncthreads();

	// convolution
	int outputIdx = inputLoadIdxBase + (threadIdx.x / inputWidth) * inputHeight * inputWidth + threadIdx.x % inputWidth;

	int inputAccessBase = (threadIdx.x / inputWidth) * paddedWidth * inputHeight + threadIdx.x % inputWidth;
	int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;
	// 1st row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 5] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 4 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	// 		2. filter's 3rd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 2nd row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 4th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;
	sum3 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 4] * inTemp4;

	#pragma unroll
	for (int i = 0; i < 4; i++) {
		// 3rd row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 1st row of input)
		// 		2. filter's 4th row (when filter is sliding through the 2nd row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 3rd row of input) 
		//		4. filter's 2nd row (when filter is sliding through the 4th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
		sum3 = sum3 + filterData[filterAccessBase + 5] * inTemp0;
		sum4 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 6] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 7] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 8] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 9] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += inputWidth;

		// 4th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 2nd row of input)
		// 		2. filter's 4th row (when filter is sliding through the 3rd row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 4th row of input) 
		//		4. filter's 2nd row (when filter is sliding through the 5th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 6th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
		sum3 = sum3 + filterData[filterAccessBase + 10] * inTemp0;
		sum4 = sum4 + filterData[filterAccessBase + 5] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 11] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 6] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 12] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 7] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 13] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 8] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 14] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 9] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += inputWidth;

		// 5th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
		// 		2. filter's 4th row (when filter is sliding through the 4th row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 5th row of input) 
		// 		4. filter's 2nd row (when filter is sliding through the 6th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
		sum3 = sum3 + filterData[filterAccessBase + 15] * inTemp0;
		sum4 = sum4 + filterData[filterAccessBase + 10] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 16] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 11] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 17] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 12] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 18] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 13] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 19] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 14] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += inputWidth;

		// 6th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 4th row of input)
		// 		2. filter's 4th row (when filter is sliding through the 5th row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 6th row of input) 
		// 		4. filter's 2nd row (when filter is sliding through the 7th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 8th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 20] * inTemp0;
		sum4 = sum4 + filterData[filterAccessBase + 15] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;
		sum2 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 21] * inTemp1;
		sum4 = sum4 + filterData[filterAccessBase + 16] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 22] * inTemp2;
		sum4 = sum4 + filterData[filterAccessBase + 17] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 23] * inTemp3;
		sum4 = sum4 + filterData[filterAccessBase + 18] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum3 = sum3 + filterData[filterAccessBase + 24] * inTemp4;
		sum4 = sum4 + filterData[filterAccessBase + 19] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum3 * alpha + beta;
		outputIdx += inputWidth;

		// 7th row
		// convolve with filter 5 times:
		//		1. filter's 5th row (when filter is sliding through the 5th row of input)
		// 		2. filter's 4th row (when filter is sliding through the 6th row of input) 
		//		3. filter's 3rd row (when filter is sliding through the 7th row of input) 
		// 		4. filter's 2nd row (when filter is sliding through the 8th row of input) 
		//		5. filter's 1st row (when filter is sliding through the 9th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 20] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;
		sum3 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 21] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;
		sum3 = sum3 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 22] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;
		sum3 = sum3 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 23] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;
		sum3 = sum3 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum4 = sum4 + filterData[filterAccessBase + 24] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
		sum3 = sum3 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum4 * alpha + beta;
		outputIdx += inputWidth;
	}

	// 23rd row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 5] * inTemp0;
	sum4 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 6] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 7] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 8] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 9] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += inputWidth;

	// 24th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 10] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 5] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 11] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 6] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 12] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 7] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 13] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 8] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 14] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 9] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += inputWidth;

	// 25th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 15] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 16] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 17] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 18] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 19] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += inputWidth;

	// 26th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 20] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 21] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 22] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 23] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 24] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum3 * alpha + beta;
	outputIdx += inputWidth;

	// 27th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
	output[outputIdx] = sum4 * alpha + beta;
	outputIdx += inputWidth;

	// 28th row
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += inputWidth;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += inputWidth;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += inputWidth;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 56 x 56, stride 1, padding 1

Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	56 x 56 x 144 -> 56 x 56 x 144, stride = 1, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input56x56_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	__shared__ float filterData[9];	// filter is 3 x 3 = 9
	__shared__ float inputData[58 * 58]; // original input is 56 x 56, padded to be 58 x 58.

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 1;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 9) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set padding
	if (threadIdx.x >= 32 && threadIdx.x < 88) {
		int leftPaddingIdx = (threadIdx.x - 31) * 58;
		inputData[leftPaddingIdx] = 0;
		inputData[leftPaddingIdx + 57] = 0;
	}
	if (threadIdx.x >= 96 && threadIdx.x < 154) {
		inputData[threadIdx.x - 96] = 0;
	}
	if (threadIdx.x >= 160 && threadIdx.x < 218) {
		inputData[threadIdx.x - 160 + 58 * 57] = 0;
	}
	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1 + paddedWidth;	// each thread find its own load destination.

	#pragma unroll
	for (int i = 0; i < 14; i++) {
		inputData[inputLoadDstIdx + 4 * 58 * i] = input[inputLoadSrcIdx + 4 * 56 * i];
	}

	__syncthreads();

	// convolution
	int outputIdx = inputLoadIdxBase + (threadIdx.x / inputWidth) * 14 * inputWidth + threadIdx.x % inputWidth;

	// 4 * 56 threads are separated to 4 groups.
	// first group handles 1 - 14 row
	// second group handles 15 - 28 row
	// third group handles 29 - 42 row
	// forth group handles 43 - 56 row
	int inputAccessBase = (threadIdx.x / inputWidth) * paddedWidth * 14 + threadIdx.x % inputWidth;
	// int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int filterAccessBase = 0;
	int inputAccessOffset = 0;

	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;


	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;


	#pragma unroll
	for (int i = 0; i < 4; i++) {
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;
		sum2 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += inputWidth;

		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += inputWidth;

		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += inputWidth;
	}
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += inputWidth;

	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 56 x 56, stride 2, padding 1

Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	56 x 56 x 144 -> 28 x 28 x 144, stride = 2, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input56x56_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	__shared__ float filterData[2 * 9];	// filter is 3 x 3 = 9
	__shared__ float inputData[2 * 56 * 58]; // original input is 56 x 56, padded to be 58 x 58. ignore up and bottom padding, so 56 x 58

	float inTemp0, inTemp1, inTemp2;
	float sum0, sum1;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 2;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 2 * 9)
	{
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 57] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + 57] = 0;

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;	// each thread find its own load destination.

	#pragma unroll
	for (int i = 0; i < 112; i++) {
		inputData[inputLoadDstIdx + 58 * i] = input[inputLoadSrcIdx + 56 * i];
	}

	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth + threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth * 2;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 1st row of input) 
	inTemp0 = inputData[inputAccessBase];
	sum0 = filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1];
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2];
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

	// 2nd row
	// convolve with filter 2 times:
	//		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 3rd row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	#pragma unroll
	for (int i = 0; i < 13; i++) {
		// 3rd row
		// convolve with filter 1 time:
		// 		1. filter's 2nd row (when filter is sliding through the 3rd row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

		// 4th row
		// convolve with filter 2 times:
		//		1. filter's 3rd row (when filter is sliding through the 3rd row of input)
		//		2. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		// 5th row
		// convolve with filter 1 time:
		// 		1. filter's 2nd row (when filter is sliding through the 5th row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp2;

		// 6th row
		// convolve with filter 2 times:
		//		1. filter's 3rd row (when filter is sliding through the 5th row of input)
		//		2. filter's 1st row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;
	}

	// 55th row
	// convolve with filter 1 time:
	// 		1. filter's 2nd row (when filter is sliding through the 27th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp2;

	// 56th row
	// convolve with filter 1 time:
	// 		1. filter's 3rd row (when filter is sliding through the 27th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp0;

	inTemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp1;

	inTemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;
}

/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 56 x 56, stride 2, padding 2

Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	56 x 56 x 144 -> 28 x 28 x 144, stride = 2, filter = 5
*/
template <typename scalar_t>
__global__ void Filter5x5_Input56x56_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	__shared__ float filterData[2 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[2 * 56 * 60]; // original input is 56 x 56, padded to be 60 x 60. ignore up and bottom padding, so 56 x 60

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 2;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 2 * 25) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;

	inputData[leftPaddingIdx + paddedWidth - 2] = 0;
	inputData[leftPaddingIdx + paddedWidth - 1] = 0;

	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + 1] = 0;

	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + paddedWidth - 2] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + paddedWidth - 1] = 0;

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	#pragma unroll
	for (int i = 0; i < 112; i++) {
		inputData[inputLoadDstIdx + 60 * i] = input[inputLoadSrcIdx + 56 * i];
	}

	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth + threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth * 2;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 2 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 2 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	//		2. filter's 2nd row (when filter is sliding through the 3rd row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		// 3rd row, 45
		// convolve with filter 3 times:
		//		1. filter's 5th row (when filter is sliding through the 1st row of input)
		//		2. filter's 3rd row (when filter is sliding through the 3rd row of input) 
		//		3. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
		sum2 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;

		// 4th row
		// convolve with filter 2 times:
		// 		1. filter's 4th row (when filter is sliding through the 3rd row of input) 
		//		2. filter's 2nd row (when filter is sliding through the 5th row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;

		// 5th row
		// convolve with filter 3 times:
		//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
		//		2. filter's 3rd row (when filter is sliding through the 5th row of input) 
		//		3. filter's 1st row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		// 6th row
		// convolve with filter 2 times:
		// 		1. filter's 4th row (when filter is sliding through the 5th row of input) 
		// 		2. filter's 2nd row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;

		// 7th row
		// convolve with filter 3 times:
		//		1. filter's 5th row (when filter is sliding through the 5th row of input)
		//		2. filter's 3rd row (when filter is sliding through the 7th row of input)
		//		3. filter's 1st row (when filter is sliding through the 9th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += outputWidth;

		// 8th row, 50th row
		// convolve with filter 2 times:
		//		1. filter's 4th row (when filter is sliding through the 7th row of input)
		//		2. filter's 2nd row (when filter is sliding through the 9th row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	}


	// 51st row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 49th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 51th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 53rd row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 52nd row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 51st row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 53rd row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;

	// 53rd row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 51st row of input)
	//		2. filter's 3rd row (when filter is sliding through the 53rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 55th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 54th row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 53rd row of input) 
	// 		2. filter's 2nd row (when filter is sliding through the 55th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;

	// 55th row
	// convolve with filter 2 times:
	//		1. filter's 5th row (when filter is sliding through the 53th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 55th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 56th row
	// convolve with filter 1 times:
	//		1. filter's 4th row (when filter is sliding through the 55th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 112 x 112, stride 1, padding 1


Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	112 x 112 x 32 -> 112 x 112 x 32, stride = 1, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input112x112_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// filter is 3 x 3. 9 elements in total
	__shared__ float filterData[9];
	// 4 blocks handle one 112 x 112 input. Each block handles 28 rows. With padding, each row has 114 elements.
	__shared__ float inputData[31 * 114];

	float intemp0, intemp1, intemp2;
	float sum0, sum1, sum2;

	int paddedWidth = inputWidth + 2 * padding;
	int blockGroup = 4;

	// load filter
	int filterLoadSrcIdx = blockIdx.y / blockGroup * filterHeight * filterWidth + threadIdx.x;
	if (threadIdx.x < filterWidth * filterHeight) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	int leftPaddingIdx = 0;
	// set padding
	if (threadIdx.x >= 32 && threadIdx.x < 62) {
		leftPaddingIdx = (threadIdx.x - 32) * paddedWidth;
		inputData[leftPaddingIdx] = 0;						// left padding
		inputData[leftPaddingIdx + paddedWidth - 1] = 0;	// right padding
	}
	if (threadIdx.x >= 112) {
		inputData[threadIdx.x - 111] = 0;					// Top padding
		inputData[threadIdx.x - 111 + 29 * paddedWidth] = 0;// Bottom padding
	}
	__syncthreads();

	int inputLoadIdxBase = blockIdx.x * inputHeight * inputWidth * inputChannel +
		blockIdx.y / blockGroup * inputWidth * inputHeight +
		(blockIdx.y & 3) * inputHeight / blockGroup * inputWidth;

	// block 0 needs to process 28 rows + bottom 1 row, no upper padding.
	// block 1 needs to process 28 rows + upper 1 row + bottom 1 row
	// block 2 needs to process 28 rows + upper 1 row + bottom 1 row
	// block 3 needs to process 28 rows + upper 1 row, no bottom padding
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x - inputWidth;
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;
	if ((blockIdx.y & 3) == 0) {
		inputLoadSrcIdx += inputWidth;
		inputLoadDstIdx += paddedWidth;
	}

	// each block load 28 rows, and each time load 2 rows, so 14 times
	#pragma unroll
	for (int i = 0; i < 14; i++) {
		inputData[inputLoadDstIdx + 2 * 114 * i] = input[inputLoadSrcIdx + 2 * 112 * i];
	}
	// block3 do not need to load extra 1 bottom row. 
	if ((blockIdx.y & 3) != 3) {
		inputData[inputLoadDstIdx + 2 * 114 * 14] = input[inputLoadSrcIdx + 2 * 112 * 14];

	} else {
		if (threadIdx.x < 112) {
			inputData[inputLoadDstIdx + 2 * 114 * 14] = input[inputLoadSrcIdx + 2 * 112 * 14];
		}
	}
	__syncthreads();

	// for 224 threads in a block, first 112 threads process first 14 rows, second 112 threads process rest of the 14 rows
	int outputIdx = blockIdx.x * outputHeight * outputWidth * outputChannel +
		(blockIdx.y / blockGroup) * outputHeight * outputWidth +
		(blockIdx.y & 3) * (outputHeight / blockGroup) * outputWidth +
		(threadIdx.x / outputWidth) * 14 * outputWidth + threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / inputWidth) * 14 * paddedWidth + threadIdx.x % inputWidth;
	int inputAccessOffset = 0;

	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[2] * intemp2;

	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[3] * intemp0;
	sum1 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[4] * intemp1;
	sum1 = sum1 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[5] * intemp2;
	sum1 = sum1 + filterData[2] * intemp2;

	#pragma unroll
	for (int i = 0; i < 4; i++) {
		inputAccessOffset += paddedWidth;
		intemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[6] * intemp0;
		sum1 = sum1 + filterData[3] * intemp0;
		sum2 = filterData[0] * intemp0;
		intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum0 = sum0 + filterData[7] * intemp1;
		sum1 = sum1 + filterData[4] * intemp1;
		sum2 = sum2 + filterData[1] * intemp1;
		intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum0 = sum0 + filterData[8] * intemp2;
		sum1 = sum1 + filterData[5] * intemp2;
		sum2 = sum2 + filterData[2] * intemp2;

		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;

		inputAccessOffset += paddedWidth;
		intemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[6] * intemp0;
		sum2 = sum2 + filterData[3] * intemp0;
		sum0 = filterData[0] * intemp0;
		intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum1 = sum1 + filterData[7] * intemp1;
		sum2 = sum2 + filterData[4] * intemp1;
		sum0 = sum0 + filterData[1] * intemp1;
		intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum1 = sum1 + filterData[8] * intemp2;
		sum2 = sum2 + filterData[5] * intemp2;
		sum0 = sum0 + filterData[2] * intemp2;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		inputAccessOffset += paddedWidth;
		intemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[6] * intemp0;
		sum0 = sum0 + filterData[3] * intemp0;
		sum1 = filterData[0] * intemp0;
		intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
		sum2 = sum2 + filterData[7] * intemp1;
		sum0 = sum0 + filterData[4] * intemp1;
		sum1 = sum1 + filterData[1] * intemp1;
		intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
		sum2 = sum2 + filterData[8] * intemp2;
		sum0 = sum0 + filterData[5] * intemp2;
		sum1 = sum1 + filterData[2] * intemp2;

		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += outputWidth;
	}

	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[6] * intemp0;
	sum1 = sum1 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[7] * intemp1;
	sum1 = sum1 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[8] * intemp2;
	sum1 = sum1 + filterData[5] * intemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += inputWidth;

	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[6] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[7] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[8] * intemp2;

	output[outputIdx] = sum1 * alpha + beta;
}

/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 112 x 112, stride 2, padding 1
2 blocks process 1 channel.

Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	112 x 112 x 96 -> 56 x 56 x 96, stride = 2, filter = 3
*/
template <typename scalar_t>
__global__ void Filter3x3_Input112x112_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// filter is 3 x 3. 9 elements in total
	__shared__ float filterData[9];
	// 2 blocks handle one 112 x 112 input. Each block handles 56 rows. With padding, 58 rows, each row has 114 elements
	__shared__ float inputData[59 * 114];

	float intemp0, intemp1, intemp2;
	float sum0, sum1;

	int paddedWidth = inputWidth + 2 * padding;
	int blockGroup = 2;

	// load filter
	int filterLoadSrcIdx = blockIdx.y / blockGroup * filterHeight * filterWidth + threadIdx.x;
	if (threadIdx.x < filterWidth * filterHeight) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	int leftPaddingIdx = 0;
	// set padding
	if (threadIdx.x >= 32 && threadIdx.x < 90) {
		leftPaddingIdx = (threadIdx.x - 32) * paddedWidth;
		inputData[leftPaddingIdx] = 0;						// left padding
		inputData[leftPaddingIdx + paddedWidth - 1] = 0;	// right padding
	}
	if (threadIdx.x >= 112) {
		inputData[threadIdx.x - 111] = 0;					// Top padding
		inputData[threadIdx.x - 111 + 57 * paddedWidth] = 0;// Bottom padding
	}
	__syncthreads();

	int inputLoadIdxBase = blockIdx.x * inputHeight * inputWidth * inputChannel +
		blockIdx.y / blockGroup * inputHeight * inputWidth +
		(blockIdx.y & 1) * inputHeight / blockGroup * inputWidth;

	// block 0 needs to process 56 rows + bottom 1 row, no upper padding.
	// block 1 needs to process 56 rows + upper 1 row + bottom 1 row
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x - inputWidth;
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 2 + threadIdx.x + 1;
	if ((blockIdx.y & 1) == 0) {
		inputLoadSrcIdx += inputWidth;
		inputLoadDstIdx += paddedWidth;
	}

	// each block load 56 rows, and each time load 2 rows, so 28 times
	#pragma unroll
	for (int i = 0; i < 28; i++) {
		inputData[inputLoadDstIdx + 2 * 114 * i] = input[inputLoadSrcIdx + 2 * 112 * i];
	}
	// block1 do not need to load extra 1 bottom row. 
	if ((blockIdx.y & 1) != 1) {
		inputData[inputLoadDstIdx + 2 * 114 * 28] = input[inputLoadSrcIdx + 2 * 112 * 28];
	}
	else {
		if (threadIdx.x < 112) {
			inputData[inputLoadDstIdx + 2 * 114 * 28] = input[inputLoadSrcIdx + 2 * 112 * 28];
		}
	}
	__syncthreads();

	// for a 224-thread block, every 56-thread group processes 14 rows in the input, and write 7 rows in the output
	int outputIdx = blockIdx.x * outputHeight * outputWidth * outputChannel +
		(blockIdx.y / blockGroup) * outputHeight * outputWidth +
		(blockIdx.y & 1) * (outputHeight / blockGroup) * outputWidth +
		(threadIdx.x / outputWidth) * 7 * outputWidth + 
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * 14 * paddedWidth + threadIdx.x % outputWidth * 2;
	int inputAccessOffset = 0;

	// row 0
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 =	filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[2] * intemp2;

	// row 1
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[5] * intemp2;

	// row 2
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[6] * intemp0;
	sum1 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[7] * intemp1;
	sum1 = sum1 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[8] * intemp2;
	sum1 = sum1 + filterData[2] * intemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// row 3
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[5] * intemp2;

	// row 4
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[0] * intemp0;
	sum1 = sum1 + filterData[6] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[1] * intemp1;
	sum1 = sum1 + filterData[7] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[2] * intemp2;
	sum1 = sum1 + filterData[8] * intemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// row 5
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[5] * intemp2;

	// row 6
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[6] * intemp0;
	sum1 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[7] * intemp1;
	sum1 = sum1 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[8] * intemp2;
	sum1 = sum1 + filterData[2] * intemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// row 7
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[5] * intemp2;

	// row 8
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[6] * intemp0;
	sum0 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[7] * intemp1;
	sum0 = sum0 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[8] * intemp2;
	sum0 = sum0 + filterData[2] * intemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// row 9
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[5] * intemp2;

	// row 10
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[6] * intemp0;
	sum1 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[7] * intemp1;
	sum1 = sum1 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[8] * intemp2;
	sum1 = sum1 + filterData[2] * intemp2;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// row 11
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[5] * intemp2;

	// row 12
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[6] * intemp0;
	sum0 = filterData[0] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum1 = sum1 + filterData[7] * intemp1;
	sum0 = sum0 + filterData[1] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum1 = sum1 + filterData[8] * intemp2;
	sum0 = sum0 + filterData[2] * intemp2;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// row 13
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[3] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[4] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[5] * intemp2;

	// row 14
	inputAccessOffset += paddedWidth;
	intemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[6] * intemp0;
	intemp1 = inputData[inputAccessBase + inputAccessOffset + 1];
	sum0 = sum0 + filterData[7] * intemp1;
	intemp2 = inputData[inputAccessBase + inputAccessOffset + 2];
	sum0 = sum0 + filterData[8] * intemp2;

	output[outputIdx] = sum0 * alpha + beta;
}

// Use Dispatch function to invoke kernel
torch::Tensor optimizedDepthwise_cuda_forward(
    torch::Tensor input,
    torch::Tensor filter,
    int filterHeight,
    int stride) {

    // numel() - return the total number o elements in the tensor
    //TORCH_CHECK(input.numel() > 0 && input.dim() == 4);
    //TORCH_CHECK(filter.numel() > 0 && filter.dim() == 4);

    auto inputShape = input.sizes();
    auto filterShape = filter.sizes();

    // filter size should be (outputChannel, 1, kH, kW)
    //TORCH_CHECK(filterShape[1] == 1);

    // input size should be (inputBatchNumber, inputChannel, inputHeight, inputWidth)
    //TORCH_CHECK(filterShape[0] % inputShape[1] == 0);

    int inputBatchNumber = inputShape[0];
    int inputChannel = inputShape[1];
    int inputHeight = inputShape[2];
    int inputWidth = inputShape[3];

	int filterLayerNumber = inputChannel;

    int paddingHeight = 0;
    int paddingWidth = 0;
	if(filterHeight == 3) {
		paddingHeight = paddingWidth = 1;
	} else if(filterHeight == 5) {
		paddingHeight = paddingWidth = 2;
	}

    int outputBatchNumber = inputBatchNumber;
    int outputChannel = inputChannel;
    int outputHeight = (inputHeight + paddingHeight * 2 - filterHeight) / stride + 1;
    int outputWidth = (inputWidth + paddingWidth * 2 - filterHeight) / stride + 1;

	// Critical bug: need to create output tensor with correct options
	torch::Tensor output = torch::empty({outputBatchNumber, outputChannel, outputHeight, outputWidth}, torch::kCUDA);
	
    float alpha = 1.0f;
	float beta = 0.0f;

	// Critical bug: SECOND PARAMETER(the string one) MUST MATCH the name of THIS FUNCTION
    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimizedDepthwise_cuda_forward", [&] {

	if (stride == 1) {
		if (filterHeight == 3) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				Filter3x3_Input7x7_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 16);
				dim3 blockSize(14 * 16, 1);
				Filter3x3_Input14x14_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(28 * 8, 1);
				Filter3x3_Input28x28_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel);
				dim3 blockSize(4 * 56, 1);
				Filter3x3_Input56x56_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 112) {
				dim3 gridSize(outputBatchNumber, outputChannel * 4);
				dim3 blockSize(2 * 112, 1);
				Filter3x3_Input112x112_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
		}
		else if (filterHeight == 5) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				Filter5x5_Input7x7_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 16);
				dim3 blockSize(14 * 16, 1);
				Filter5x5_Input14x14_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(28 * 8, 1);
				Filter5x5_Input28x28_Stride1 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
		}
	}
	else if (stride == 2) {
		if (filterHeight == 3) {
			if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				Filter3x3_Input14x14_Stride2 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8); // if channel group size = 16, shared memory exceeded.
				dim3 blockSize(14 * 8, 1);
				Filter3x3_Input28x28_Stride2 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel / 2);
				dim3 blockSize(28 * 2, 1);
				Filter3x3_Input56x56_Stride2 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 112) {
				dim3 gridSize(outputBatchNumber, outputChannel * 2);
				dim3 blockSize(56 * 4, 1);
				Filter3x3_Input112x112_Stride2 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
		}
		else if (filterHeight == 5) {
			if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				Filter5x5_Input14x14_Stride2 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel / 2);
				dim3 blockSize(28 * 2, 1);
				Filter5x5_Input56x56_Stride2 << <gridSize, blockSize >> > (
					input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterHeight,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
			}
		}
	}
	
	});

	return output;
}

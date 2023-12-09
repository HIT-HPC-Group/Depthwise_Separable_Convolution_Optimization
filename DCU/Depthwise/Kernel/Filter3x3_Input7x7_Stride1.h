#include <hip/hip_runtime.h>
/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 7 x 7, stride 1, padding 1

The number of channel must be multiple of 32.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1) 7 x 7 x 960 -> 7 x 7 x 960, stride = 1, filter = 3
	2) 7 x 7 x 1152 -> 7 x 7 x 1152, stride = 1, filter = 3
*/
__global__ void Filter3x3_Input7x7_Stride1(const float* input, const float* filter, float* output,
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

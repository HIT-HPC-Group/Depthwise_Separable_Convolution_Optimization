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

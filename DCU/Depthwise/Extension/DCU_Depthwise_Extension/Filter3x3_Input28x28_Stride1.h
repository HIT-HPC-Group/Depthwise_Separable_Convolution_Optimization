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

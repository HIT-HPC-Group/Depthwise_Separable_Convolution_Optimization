/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 14 x 14, stride 1, padding 2

The number of channel must be multiple of 16.
Used in the MobileNet V2 and EfficientNet B0, in case of.
	1)	14 x 14 x 480 -> 14 x 14 x 480, stride = 1, filter = 5
	2)	14 x 14 x 672 -> 14 x 14 x 672, stride = 1, filter = 5

*/
__global__ void Filter5x5_Input14x14_Stride1(const float* input, const float* filter, float* output,
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
	//		4. filter's 1st row (when filter is sliding through the 4th row of input) 
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

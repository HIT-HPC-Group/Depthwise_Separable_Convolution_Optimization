/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 56 x 56, stride 1, padding 1

Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	56 x 56 x 144 -> 56 x 56 x 144, stride = 1, filter = 3
*/
__global__ void Filter3x3_Input56x56_Stride1(const float* input, const float* filter, float* output,
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

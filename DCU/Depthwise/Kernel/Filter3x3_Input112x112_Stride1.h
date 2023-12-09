/*
Depthwise Convolution Kernel.

Case: filter 3 x 3, input 112 x 112, stride 1, padding 1


Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	112 x 112 x 32 -> 112 x 112 x 32, stride = 1, filter = 3
*/
__global__ void Filter3x3_Input112x112_Stride1(const float* input, const float* filter, float* output,
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

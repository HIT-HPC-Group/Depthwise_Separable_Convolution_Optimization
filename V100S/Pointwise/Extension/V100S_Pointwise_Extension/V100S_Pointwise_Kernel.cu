#include <torch/extension.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ===========================================================================
// Input Size 7 x 7, Input Channel 576, Output Channel 160
// ===========================================================================
/*
Pointwise Convolution Kernel
InputBatch_1_Input_7x7_InChannel_576_OutChannel_160
*/

template <typename scalar_t>
__global__ void InputBatch_1_Input_7x7_InChannel_576_OutChannel_160(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    int channelGroup = 16;
    int inputSize = inputChannel * inputHeight * inputWidth;
    int inputChannelSize = inputHeight * inputWidth;
    int outputSize = outputChannel * outputHeight * outputWidth;
    int outputGroupSize = channelGroup * outputHeight * outputWidth;
    int outputChannelSize = outputHeight * outputWidth;

    int outputIdx = blockIdx.x * outputSize + blockIdx.y * outputGroupSize + threadIdx.z * outputChannelSize + threadIdx.y * outputWidth + threadIdx.x;

    // Pointwise convolution
    float partialResult = 0.0f;
    for (int j = 0; j < filterInChannel; j++) {
        int inputAccessIdx = blockIdx.x * inputSize + j * inputChannelSize + threadIdx.y * inputWidth + threadIdx.x;
        int filterAccessIdx = (blockIdx.y * channelGroup + threadIdx.z) * filterInChannel + j;
        partialResult += input[inputAccessIdx] * filter[filterAccessIdx];
    }

    // Store output
    output[outputIdx] = partialResult;
}

// Use Dispatch function to invoke kernel
torch::Tensor optimizedPointwise_cuda_forward(
    torch::Tensor input,
    torch::Tensor filter) {

    auto inputShape = input.sizes();
    auto filterShape = filter.sizes();

    int inputBatchNumber = inputShape[0];
    int inputChannel = inputShape[1];
    int inputHeight = inputShape[2];
    int inputWidth = inputShape[3];

	int filterOutChannel = filterShape[0];
	int filterInChannel = filterShape[1];

    int outputBatchNumber = inputBatchNumber;
    int outputChannel = filterOutChannel;
    int outputHeight = inputHeight;
    int outputWidth = inputWidth;

	torch::Tensor output = torch::empty({outputBatchNumber, outputChannel, outputHeight, outputWidth}, torch::kCUDA);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimizedPointwise_cuda_forward", [&] {

		if (inputBatchNumber == 1 && inputHeight == 7 && inputChannel == 576 && outputChannel == 160) {
			dim3 gridSize(outputBatchNumber, outputChannel / 16);
			dim3 blockSize(7, 7, 16);
			InputBatch_1_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>(
				input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
				inputBatchNumber, inputChannel, inputHeight, inputWidth,
				filterOutChannel, filterInChannel, filterHeight, filterWidth,
				outputBatchNumber, outputChannel, outputHeight, outputWidth);
    	}

	});

	return output;
}

#include <torch/extension.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

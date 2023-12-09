#include <torch/extension.h>
#include <hip/hip_runtime.h>

#include "Filter3x3_Input7x7_Stride1.h"
#include "Filter5x5_Input7x7_Stride1.h"
#include "Filter3x3_Input14x14_Stride1.h"
#include "Filter3x3_Input14x14_Stride2.h"
#include "Filter5x5_Input14x14_Stride1.h"
#include "Filter5x5_Input14x14_Stride2.h"
#include "Filter3x3_Input28x28_Stride1.h"
#include "Filter3x3_Input28x28_Stride2.h"
#include "Filter5x5_Input28x28_Stride1.h"
#include "Filter3x3_Input56x56_Stride1.h"
#include "Filter3x3_Input56x56_Stride2.h"
#include "Filter5x5_Input56x56_Stride2.h"
#include "Filter3x3_Input112x112_Stride1.h"
#include "Filter3x3_Input112x112_Stride2.h"

// Use Dispatch function to invoke kernel
torch::Tensor optimizedDepthwise_cuda_forward(
    torch::Tensor input,
    torch::Tensor filter,
    int filterHeight,
    int stride) {

    auto inputShape = input.sizes();
    auto filterShape = filter.sizes();

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

	torch::Tensor output = torch::empty({outputBatchNumber, outputChannel, outputHeight, outputWidth}, torch::kCUDA);
	
    float alpha = 1.0f;
	float beta = 0.0f;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimizedDepthwise_cuda_forward", [&] {

	if (stride == 1) {
		if (filterHeight == 3) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				Filter3x3_Input7x7_Stride1 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input14x14_Stride1 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input28x28_Stride1 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input56x56_Stride1 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input112x112_Stride1 <<<gridSize, blockSize >>> (
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
				Filter5x5_Input7x7_Stride1 <<<gridSize, blockSize >>> (
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
				Filter5x5_Input14x14_Stride1 <<<gridSize, blockSize >>> (
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
				Filter5x5_Input28x28_Stride1 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input14x14_Stride2 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input28x28_Stride2 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input56x56_Stride2 <<<gridSize, blockSize >>> (
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
				Filter3x3_Input112x112_Stride2 <<<gridSize, blockSize >>> (
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
				Filter5x5_Input14x14_Stride2 <<<gridSize, blockSize >>> (
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
				Filter5x5_Input56x56_Stride2 <<<gridSize, blockSize >>> (
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

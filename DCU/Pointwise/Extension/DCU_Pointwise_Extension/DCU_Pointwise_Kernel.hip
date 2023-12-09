#include <torch/extension.h>
#include <hip/hip_runtime.h>

#include "InputBatch_128_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_64_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_32_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_16_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_8_Input_112x112_InChannel_32_OutChannel_16.h"
#include "InputBatch_1_Input_112x112_InChannel_32_OutChannel_16.h"

#include "InputBatch_128_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_64_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_32_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_16_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_8_Input_112x112_InChannel_16_OutChannel_96.h"
#include "InputBatch_1_Input_112x112_InChannel_16_OutChannel_96.h"

#include "InputBatch_128_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_64_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_32_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_16_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_8_Input_56x56_InChannel_96_OutChannel_24.h"
#include "InputBatch_1_Input_56x56_InChannel_96_OutChannel_24.h"

#include "InputBatch_128_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_64_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_32_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_16_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_8_Input_56x56_InChannel_24_OutChannel_144.h"
#include "InputBatch_1_Input_56x56_InChannel_24_OutChannel_144.h"

#include "InputBatch_128_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_64_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_32_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_16_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_8_Input_56x56_InChannel_144_OutChannel_24.h"
#include "InputBatch_1_Input_56x56_InChannel_144_OutChannel_24.h"

#include "InputBatch_128_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_64_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_32_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_16_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_8_Input_28x28_InChannel_144_OutChannel_32.h"
#include "InputBatch_1_Input_28x28_InChannel_144_OutChannel_32.h"

#include "InputBatch_128_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_64_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_32_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_16_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_8_Input_28x28_InChannel_32_OutChannel_192.h"
#include "InputBatch_1_Input_28x28_InChannel_32_OutChannel_192.h"

#include "InputBatch_128_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_64_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_32_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_16_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_8_Input_28x28_InChannel_192_OutChannel_32.h"
#include "InputBatch_1_Input_28x28_InChannel_192_OutChannel_32.h"

#include "InputBatch_128_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_64_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_32_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_16_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_8_Input_28x28_InChannel_144_OutChannel_40.h"
#include "InputBatch_1_Input_28x28_InChannel_144_OutChannel_40.h"

#include "InputBatch_128_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_64_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_32_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_16_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_8_Input_28x28_InChannel_40_OutChannel_240.h"
#include "InputBatch_1_Input_28x28_InChannel_40_OutChannel_240.h"

#include "InputBatch_128_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_64_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_32_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_16_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_8_Input_28x28_InChannel_240_OutChannel_40.h"
#include "InputBatch_1_Input_28x28_InChannel_240_OutChannel_40.h"

#include "InputBatch_128_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_64_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_32_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_16_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_8_Input_14x14_InChannel_192_OutChannel_64.h"
#include "InputBatch_1_Input_14x14_InChannel_192_OutChannel_64.h"

#include "InputBatch_128_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_64_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_32_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_16_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_8_Input_14x14_InChannel_64_OutChannel_384.h"
#include "InputBatch_1_Input_14x14_InChannel_64_OutChannel_384.h"

#include "InputBatch_128_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_64_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_32_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_16_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_8_Input_14x14_InChannel_384_OutChannel_64.h"
#include "InputBatch_1_Input_14x14_InChannel_384_OutChannel_64.h"

#include "InputBatch_128_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_64_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_32_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_16_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_8_Input_14x14_InChannel_384_OutChannel_96.h"
#include "InputBatch_1_Input_14x14_InChannel_384_OutChannel_96.h"

#include "InputBatch_128_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_64_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_32_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_16_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_8_Input_14x14_InChannel_96_OutChannel_576.h"
#include "InputBatch_1_Input_14x14_InChannel_96_OutChannel_576.h"

#include "InputBatch_128_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_64_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_32_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_16_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_8_Input_14x14_InChannel_576_OutChannel_96.h"
#include "InputBatch_1_Input_14x14_InChannel_576_OutChannel_96.h"

#include "InputBatch_128_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_64_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_32_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_16_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_8_Input_14x14_InChannel_240_OutChannel_80.h"
#include "InputBatch_1_Input_14x14_InChannel_240_OutChannel_80.h"

#include "InputBatch_128_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_64_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_32_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_16_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_8_Input_14x14_InChannel_80_OutChannel_480.h"
#include "InputBatch_1_Input_14x14_InChannel_80_OutChannel_480.h"

#include "InputBatch_128_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_64_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_32_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_16_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_8_Input_14x14_InChannel_480_OutChannel_80.h"
#include "InputBatch_1_Input_14x14_InChannel_480_OutChannel_80.h"

#include "InputBatch_128_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_64_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_32_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_16_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_8_Input_14x14_InChannel_480_OutChannel_112.h"
#include "InputBatch_1_Input_14x14_InChannel_480_OutChannel_112.h"

#include "InputBatch_128_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_64_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_32_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_16_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_8_Input_14x14_InChannel_112_OutChannel_672.h"
#include "InputBatch_1_Input_14x14_InChannel_112_OutChannel_672.h"

#include "InputBatch_128_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_64_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_32_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_16_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_8_Input_14x14_InChannel_672_OutChannel_112.h"
#include "InputBatch_1_Input_14x14_InChannel_672_OutChannel_112.h"

#include "InputBatch_128_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_64_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_32_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_16_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_8_Input_7x7_InChannel_576_OutChannel_160.h"
#include "InputBatch_1_Input_7x7_InChannel_576_OutChannel_160.h"

#include "InputBatch_128_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_64_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_32_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_16_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_8_Input_7x7_InChannel_160_OutChannel_960.h"
#include "InputBatch_1_Input_7x7_InChannel_160_OutChannel_960.h"

#include "InputBatch_128_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_64_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_32_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_16_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_8_Input_7x7_InChannel_960_OutChannel_160.h"
#include "InputBatch_1_Input_7x7_InChannel_960_OutChannel_160.h"

#include "InputBatch_128_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_64_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_32_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_16_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_8_Input_7x7_InChannel_960_OutChannel_320.h"
#include "InputBatch_1_Input_7x7_InChannel_960_OutChannel_320.h"

#include "InputBatch_128_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_64_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_32_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_16_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280.h"
#include "InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280.h"

#include "InputBatch_128_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_64_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_32_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_16_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_8_Input_7x7_InChannel_672_OutChannel_192.h"
#include "InputBatch_1_Input_7x7_InChannel_672_OutChannel_192.h"

#include "InputBatch_128_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_64_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_32_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_16_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152.h"
#include "InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152.h"

#include "InputBatch_128_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_64_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_32_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_16_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_8_Input_7x7_InChannel_1152_OutChannel_192.h"
#include "InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192.h"

#include "InputBatch_128_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_64_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_32_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_16_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_8_Input_7x7_InChannel_1152_OutChannel_320.h"
#include "InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320.h"

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
	int filterHeight = 1;
	int filterWidth = 1;

    int outputBatchNumber = inputBatchNumber;
    int outputChannel = filterOutChannel;
    int outputHeight = inputHeight;
    int outputWidth = inputWidth;

	torch::Tensor output = torch::empty({outputBatchNumber, outputChannel, outputHeight, outputWidth}, torch::kCUDA);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimizedPointwise_cuda_forward", [&] {
		
		// Kernel Invocation - Pointwise Kernels
		if(inputBatchNumber == 1) {
			if(inputHeight == 112) {
				if(inputChannel == 32 && outputChannel == 16) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 16 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 56) {
				if(inputChannel == 96 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 12 * 8));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 24 && outputChannel == 144) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 24 * 4));
					dim3 blockSize(7 * 64);
					InputBatch_1_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 28) {
				if(inputChannel == 144 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 32 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 40 && outputChannel == 240) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 14) {
				if(inputChannel == 192 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 2 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 64 && outputChannel == 384) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 96 && outputChannel == 576) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 576 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 80 && outputChannel == 480) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 4 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_1_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 112 && outputChannel == 672) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 7) {
				if(inputChannel == 576 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 2 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 160 && outputChannel == 960) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 320 && outputChannel == 1280) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 1152) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 18 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 4 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_1_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			}
		} else if(inputBatchNumber == 8) {
			if(inputHeight == 112) {
				if(inputChannel == 32 && outputChannel == 16) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 16 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
							 
				}
			} else if(inputHeight == 56) {
				if(inputChannel == 96 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 24 && outputChannel == 144) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 28) {
				if(inputChannel == 144 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 32 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 40 && outputChannel == 240) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 14) {
				if(inputChannel == 192 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 64 && outputChannel == 384) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 64 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 96 && outputChannel == 576) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 192 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_8_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 576 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 80 && outputChannel == 480) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 240 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 28 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 112 && outputChannel == 672) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 28 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 7) {
				if(inputChannel == 576 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 160 && outputChannel == 960) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 120 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 8 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 10 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 320 && outputChannel == 1280) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 1152) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 36 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 12 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_8_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			}
		} else if(inputBatchNumber == 16) {
			if(inputHeight == 112) {
				if(inputChannel == 32 && outputChannel == 16) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 16 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 56) {
				if(inputChannel == 96 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 24 && outputChannel == 144) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 28) {
				if(inputChannel == 144 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 32 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 40 && outputChannel == 240) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 14) {
				if(inputChannel == 192 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 64 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 64 && outputChannel == 384) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 96 && outputChannel == 576) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 576 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 80 && outputChannel == 480) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 160 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 112 && outputChannel == 672) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 7) {
				if(inputChannel == 576 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 160 && outputChannel == 960) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 320 && outputChannel == 1280) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 1152) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_16_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			}
		} else if(inputBatchNumber == 32) {
			if(inputHeight == 112) {
				if(inputChannel == 32 && outputChannel == 16) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 16 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 56) {
				if(inputChannel == 96 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 24 && outputChannel == 144) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 28) {
				if(inputChannel == 144 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 32 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 40 && outputChannel == 240) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 14) {
				if(inputChannel == 192 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 64 && outputChannel == 384) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 64 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 96 && outputChannel == 576) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 576 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 80 && outputChannel == 480) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 240 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 112 && outputChannel == 672) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 7) {
				if(inputChannel == 576 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 160 && outputChannel == 960) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 96 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_32_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 320 && outputChannel == 1280) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 1152) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 96 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_32_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_32_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			}
		} else if(inputBatchNumber == 64) {
			if(inputHeight == 112) {
				if(inputChannel == 32 && outputChannel == 16) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 16 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 56) {
				if(inputChannel == 96 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 24 && outputChannel == 144) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 28) {
				if(inputChannel == 144 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 32 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 40 && outputChannel == 240) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 14) {
				if(inputChannel == 192 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 64 && outputChannel == 384) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 96 && outputChannel == 576) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 576 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 80 && outputChannel == 480) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 112 && outputChannel == 672) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 7) {
				if(inputChannel == 576 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 160 && outputChannel == 960) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_64_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 320 && outputChannel == 1280) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 1152) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_64_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 20 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_64_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			}
		} else if(inputBatchNumber == 128) {
			if(inputHeight == 112) {
				if(inputChannel == 32 && outputChannel == 16) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 16 * 28));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_112x112_InChannel_32_OutChannel_16<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 16 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 8));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_112x112_InChannel_16_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 56) {
				if(inputChannel == 96 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_56x56_InChannel_96_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 24 && outputChannel == 144) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 144 * 4));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_56x56_InChannel_24_OutChannel_144<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 24) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_56x56_InChannel_144_OutChannel_24<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 28) {
				if(inputChannel == 144 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_28x28_InChannel_144_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 32 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 96 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_28x28_InChannel_32_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 32) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_28x28_InChannel_192_OutChannel_32<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 144 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_28x28_InChannel_144_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 40 && outputChannel == 240) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_28x28_InChannel_40_OutChannel_240<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 40) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_28x28_InChannel_240_OutChannel_40<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 14) {
				if(inputChannel == 192 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_192_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 64 && outputChannel == 384) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_64_OutChannel_384<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 64) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 32 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_384_OutChannel_64<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 384 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_384_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 96 && outputChannel == 576) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 48 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_96_OutChannel_576<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 576 && outputChannel == 96) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_576_OutChannel_96<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 240 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_240_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 80 && outputChannel == 480) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 80 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_80_OutChannel_480<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 80) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_480_OutChannel_80<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 480 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_480_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 112 && outputChannel == 672) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 112 * 14));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_112_OutChannel_672<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 112) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 56 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_14x14_InChannel_672_OutChannel_112<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			} else if(inputHeight == 7) {
				if(inputChannel == 576 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_576_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 160 && outputChannel == 960) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_128_Input_7x7_InChannel_160_OutChannel_960<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 160) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_960_OutChannel_160<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 960 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_960_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 320 && outputChannel == 1280) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_320_OutChannel_1280<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 672 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_672_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 192 && outputChannel == 1152) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (7 * 48 * 7));
					dim3 blockSize(7 * 64);
					InputBatch_128_Input_7x7_InChannel_192_OutChannel_1152<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 192) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 24 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_1152_OutChannel_192<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				} else if(inputChannel == 1152 && outputChannel == 320) {
					
					dim3 gridSize(outputBatchNumber * outputHeight * outputWidth * outputChannel / (4 * 40 * 7));
					dim3 blockSize(4 * 64);
					InputBatch_128_Input_7x7_InChannel_1152_OutChannel_320<<<gridSize, blockSize>>>( input.data_ptr<scalar_t>(), filter.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
						inputBatchNumber, inputChannel, inputHeight, inputWidth,
						filterOutChannel, filterInChannel, filterHeight, filterWidth,
						outputBatchNumber, outputChannel, outputHeight, outputWidth);
					 
				}
			}
		}
	});

	return output;
}

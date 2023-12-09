#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
const int DCU_WARP_SIZE = 64;

int inputChannel, inputSize, outputChannel, inputBatchSize;
int warpNumPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize;
int horizontalRepeat, verticalRepeat;
char name[1000];

int gcd(int x, int y) {
	if (!y) return x;
	return gcd(y, x % y);
}

void init(char *argv[]) {

	inputChannel=atoi(argv[1]);
	inputSize=atoi(argv[2]);
	outputChannel=atoi(argv[3]);
	inputBatchSize=atoi(argv[4]);
	warpNumPerBlock=atoi(argv[5]);
	outputChannelPerWarp=atoi(argv[6]);
	outputWidthPerWarp=atoi(argv[7]);
	channelGroupSize=atoi(argv[8]);
	horizontalRepeat=atoi(argv[9]);
	verticalRepeat=atoi(argv[10]);
	printf("Input Channel: ");
	printf("%d\n", inputChannel);
	printf("Input Height/Width: ");
	printf("%d\n", inputSize);
	printf("Output Channel: ");
	printf("%d\n", outputChannel);
	printf("Input Batch Size: ");
	printf("%d\n", inputBatchSize);
	printf("Warp Number Per Block: ");
	printf("%d\n", warpNumPerBlock);
	printf("Output Channel Per Warp: ");
	printf("%d\n", outputChannelPerWarp);
	printf("Output Width Per Warp: ");
	printf("%d\n", outputWidthPerWarp);
	printf("Channel Group Size: ");
	printf("%d\n", channelGroupSize);
	printf("Horizontal Repeat: ");
	printf("%d\n", horizontalRepeat);
	printf("Vertical Repeat: ");
	printf("%d\n", verticalRepeat);
	sprintf(name, "InputBatch_%d_Input_%dx%d_InChannel_%d_OutChannel_%d.h", inputBatchSize, inputSize, inputSize, inputChannel, outputChannel);
	freopen(argv[11], "w", stdout);
	//freopen(name, "w", stdout);
}

void print_comment() {
	puts("/*");
	puts("Pointwise Convolution Kernel");
	printf("InputBatch_%d_Input_%dx%d_InChannel_%d_OutChannel_%d\n", inputBatchSize, inputSize, inputSize, inputChannel, outputChannel);
	puts("");
	puts("Grid:");
	printf("    gridDim.x = (%d * %d * %d * %d) / (%d * %d * %d);\n", inputBatchSize, outputChannel, inputSize, inputSize, warpNumPerBlock, outputWidthPerWarp, outputChannelPerWarp);
	puts("Block:");
	printf("    blockDim.x = %d * %d;\n", DCU_WARP_SIZE, warpNumPerBlock);
	puts("");
	printf("warpNumPerBlock = %d\n", warpNumPerBlock);
	printf("outputWidthPerWarp = %d\n", outputWidthPerWarp);
	printf("outputChannelPerWarp = %d\n", outputChannelPerWarp);
	printf("channelGroupSize = %d\n", channelGroupSize);
	printf("horizontalRepeat = %d\n", horizontalRepeat);
	printf("verticalRepeat = %d\n", verticalRepeat);
	puts("");
	printf("One thread block contains %d warps, %d * %d = %d threads.\n", warpNumPerBlock, warpNumPerBlock, DCU_WARP_SIZE, warpNumPerBlock * DCU_WARP_SIZE);
	printf("Each thread block is responsible for generating %d * %d * %d output data.\n", warpNumPerBlock, outputWidthPerWarp, outputChannelPerWarp);
	printf("Each warp is responsible for generating %d * %d output data.\n", outputWidthPerWarp, outputChannelPerWarp);
	puts("");
	printf("DCU: %d %d %d %d\n", inputBatchSize, inputChannel, inputSize, outputChannel);
	puts("Kernel: TODO: ms");
	puts("miopen: TODO: ms");
	puts("*/");
	puts("");
}

void print_header() {
	puts("#include \"hip/hip_runtime.h\"");
	puts("#include <stdio.h>");
	puts("#include <iostream>");
	puts("#include <cstdlib>");
	puts("#include <cmath>");
	puts("#include <hip/hip_runtime.h>");
	puts("#include <miopen/miopen.h>");
	puts("#include <stdlib.h>");
	puts("#include <iomanip>");
	puts("#include <time.h>");
	puts("#include <random>");
	puts("#include <vector>");
	puts("#include <fstream>");
	puts("#include \"hip/hip_runtime.h\"");
	puts("");
	puts("#ifdef AMD_PLATFORM");
	puts("//#define __shfl_down_sync(mask, var, srcLine,warpsize) __shfl_down(var, srcLine, warpsize)");
	puts("#define CSVPATH \"DCU_Pointwise_result.csv\"");
	puts("// vector<string> csvHeader={\"Input Batch\",\"Input Channel\",\"Height\",\"Filter Size\",\"Stride\",\"DCU Kernel(ms)\",\"HipDNN(ms)\"};");
	puts("#else");
	puts("#define CSVPATH \"V100S_Pointwise_result.csv\"");
	puts("// vector<string> csvHeader={\"Input Batch\",\"Input Channel\",\"Height\",\"Filter Size\",\"Stride\",\"V100S Kernel(ms)\",\"Cudnn(ms)\"};");
	puts("#endif");
	puts("");
	puts("");
}

void print_kernel() {
	int g = gcd(inputSize, horizontalRepeat);
	int times = horizontalRepeat / g;
	
	int threadPerBlock = warpNumPerBlock * DCU_WARP_SIZE;
	printf("__global__ ");
	if (threadPerBlock > 256) {
		printf("__launch_bounds__(1024) ");
	}
	printf("void InputBatch_%d_Input_%dx%d_InChannel_%d_OutChannel_%d(const float* input, const float* filter, float* output,\n", inputBatchSize, inputSize, inputSize, inputChannel, outputChannel);
	puts("    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,");
	puts("    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,");
	puts("    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {");
	puts("");
	printf("    __shared__ float inputSharedBuffer1[%d * %d * %d];\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize);
	printf("    __shared__ float inputSharedBuffer2[%d * %d * %d];\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize);
	puts("");
	printf("    __shared__ float filterSharedBuffer1[%d * %d * %d];\n", verticalRepeat, outputChannelPerWarp, channelGroupSize);
	printf("    __shared__ float filterSharedBuffer2[%d * %d * %d];\n", verticalRepeat, outputChannelPerWarp, channelGroupSize);
	puts("");
	puts("    // to hold loaded operands temp.");
	int inputTemp = ceil(1.0 * horizontalRepeat * outputWidthPerWarp * channelGroupSize / (warpNumPerBlock * DCU_WARP_SIZE));
	int inputTempReminder = horizontalRepeat * outputWidthPerWarp * channelGroupSize % (warpNumPerBlock * DCU_WARP_SIZE);
	for (int i = 1; i <= inputTemp; ++i) {
		if (i % 5 == 1) {
			printf("    float");
		}
		printf(" inputTemp%d = 0", i);
		if (i == inputTemp || i % 5 == 0) {
			puts(";");
		}
		else {
			printf(",");
		}
	}
	int filterTemp = ceil(1.0 * verticalRepeat * outputChannelPerWarp * channelGroupSize / (warpNumPerBlock * DCU_WARP_SIZE));
	int filterTempReminder = verticalRepeat * outputChannelPerWarp * channelGroupSize % (warpNumPerBlock * DCU_WARP_SIZE);
	for (int i = 1; i <= filterTemp; ++i) {
		if (i % 5 == 1) {
			printf("    float");
		}
		printf(" filterTemp%d = 0", i);
		if (i == filterTemp || i % 5 == 0) {
			puts(";");
		}
		else {
			printf(",");
		}
	}
	puts("");
	puts("    // to hold operands");
	puts("    // same number as temp registers");
	int inputOperand = outputWidthPerWarp;
	for (int i = 1; i <= inputOperand; ++i) {
		if (i % 5 == 1) {
			printf("    float");
		}
		printf(" inputOperand%d = 0", i);
		if (i == inputOperand || i % 5 == 0) {
			puts(";");
		}
		else {
			printf(",");
		}
	}
	int filterOperand = outputChannelPerWarp / (DCU_WARP_SIZE / channelGroupSize);
	for (int i = 1; i <= filterOperand; ++i) {
		if (i % 5 == 1) {
			printf("    float");
		}
		printf(" filterOperand%d = 0", i);
		if (i == filterOperand || i % 5 == 0) {
			puts(";");
		}
		else {
			printf(",");
		}
	}
	puts("");
	puts("    // to hold intermediate result");
	for (int i = 1; i <= inputOperand; ++i) {
		for (int j = 1; j <= filterOperand; ++j) {
			if (j % 5 == 1) {
				printf("    float");
			}
			printf(" input%dfilter%d = 0", i, j);
			if (j == filterOperand || j % 5 == 0) {
				puts(";");
			}
			else {
				printf(",");
			}
		}
		puts("");
	}
	printf("    int warpID = threadIdx.x / %d;\n", DCU_WARP_SIZE);
	printf("    int laneID = threadIdx.x %% %d;\n", DCU_WARP_SIZE);
	puts("");
	
	// int blockLoadInputStartIdx = blockIdx.x / inputBlockPerBatch * inputSize * inputSize * inputChannel
	// + (blockIdx.x % inputBlockPerBatch) / inputBlockPerY * inputSize
	// + (blockIdx.x % inputBlockPerY) / inputRepeat;
	
	// inputSharedBuffer1[threadIdx.x] = input[blockLoadInputStartIdx
	// + finishedNum * perNum
	// + threadIdx.x / blockLayer * inputLayer
	// + (threadIdx.x % blockLayer) / outputWidthPerWarp * inputSize
	// + threadIdx.x % outputWidthPerWarp];
	
	puts("    // input");
	int inputXNum = inputSize / (outputWidthPerWarp * times);
	int inputYNum = inputSize / (horizontalRepeat / times);
	int inputRepeat = outputChannel / (outputChannelPerWarp * verticalRepeat);
	int inputBlockPerBatch = inputXNum * inputYNum * inputRepeat;
	int inputBlockPerY = inputXNum * inputRepeat;
	int inputPicSize = inputSize * inputSize * inputChannel;
	int blockLayer = outputWidthPerWarp * horizontalRepeat;
	int inputLayer = inputSize * inputSize;
	printf("    int blockLoadInputStartIdx = blockIdx.x / %d * %d + (blockIdx.x %% %d) / %d * %d + (blockIdx.x %% %d) / %d * %d;\n", inputBlockPerBatch, inputPicSize, inputBlockPerBatch, inputBlockPerY, inputSize * (horizontalRepeat / times), inputBlockPerY, inputRepeat, outputWidthPerWarp * times);
	for (int i = 0; i < inputTemp; ++i) {
		if (i == inputTemp - 1 && inputTempReminder != 0) {
			printf("    if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
			printf("        inputSharedBuffer1[threadIdx.x + %d * %d] = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i, threadPerBlock, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
			puts("    }");
		}
		else {
			printf("    inputSharedBuffer1[threadIdx.x + %d * %d] = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i, threadPerBlock, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
		}
	}
	puts("");
	puts("    // filter");
	int channelPerBlock = outputChannelPerWarp * verticalRepeat;
	int offsetChannelPerBlock = channelPerBlock * inputChannel;
	printf("    int blockLoadFilterStartIdx = (blockIdx.x %% %d) * %d;\n", inputRepeat, offsetChannelPerBlock);
	for (int i = 0; i < filterTemp; ++i) {
		if (i == filterTemp - 1 && filterTempReminder != 0) {
			printf("    if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
			printf("        filterSharedBuffer1[threadIdx.x + %d * %d] = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i, threadPerBlock, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
			puts("    }");
		}
		else {
			printf("    filterSharedBuffer1[threadIdx.x + %d * %d] = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i, threadPerBlock, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
		}
	}
	puts("    __syncthreads();\n");
	puts("");
	puts("    // For loop begins");
	printf("    for(int i = 0; i < (inputChannel - %d) / (2 * %d); i++) {\n", channelGroupSize, channelGroupSize);
	puts("        // load next group of Cnum channels");
	printf("        blockLoadInputStartIdx += %d * %d * %d;\n", inputSize, inputSize, channelGroupSize);
	for (int i = 0; i < inputTemp; ++i) {
		if (i == inputTemp - 1 && inputTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            inputTemp%d = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i + 1, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
			puts("        }");
		}
		else {
			printf("        inputTemp%d = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i + 1, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
		}
	}
	puts("");
	printf("        blockLoadFilterStartIdx += %d;\n", channelGroupSize);
	for (int i = 0; i < filterTemp; ++i) {
		if (i == filterTemp - 1 && filterTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            filterTemp%d = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i + 1, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
			puts("        }");
		}
		else {
			printf("        filterTemp%d = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i + 1, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
		}
	}
	puts("");
	puts("        // Copy operands from shared buffer 1 into Operands Registers");
	for (int i = 1; i <= inputOperand; ++i) {
		printf("        inputOperand%d = inputSharedBuffer1[((warpID / %d) %% %d) * %d + (laneID %% %d) * %d * %d + %d];\n", i, verticalRepeat, horizontalRepeat, outputWidthPerWarp, channelGroupSize, outputWidthPerWarp, horizontalRepeat, i - 1);
		if (i == inputOperand || i % 5 == 0) {
			puts("");
		}
	}
	for (int i = 1; i <= filterOperand; ++i) {
		printf("        filterOperand%d = filterSharedBuffer1[(warpID %% %d) * %d + laneID + %d * %d];\n", i, verticalRepeat, outputChannelPerWarp * channelGroupSize, i - 1, DCU_WARP_SIZE);
		if (i == filterOperand || i % 5 == 0) {
			puts("");
		}
	}
	puts("        // Compute and Accumulate result in Result Registers");
	for (int i = 1; i <= inputOperand; ++i) {
		for (int j = 1; j <= filterOperand; ++j) {
			printf("        input%dfilter%d += inputOperand%d * filterOperand%d;\n", i, j, i, j);
			if (j == filterOperand || j % 5 == 0) {
				puts("");
			}
		}
	}
	puts("        // Copy Temp Registers to shared buffer 2");
	for (int i = 0; i < inputTemp; ++i) {
		if (i == inputTemp - 1 && inputTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            inputSharedBuffer2[threadIdx.x + %d * %d] = inputTemp%d;\n", i, threadPerBlock, i + 1);
			puts("        }");
		}
		else {
			printf("        inputSharedBuffer2[threadIdx.x + %d * %d] = inputTemp%d;\n", i, threadPerBlock, i + 1);
		}
	}
	puts("");
	for (int i = 0; i < filterTemp; ++i) {
		if (i == filterTemp - 1 && filterTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            filterSharedBuffer2[threadIdx.x + %d * %d] = filterTemp%d;\n", i, threadPerBlock, i + 1);
			puts("        }");
		}
		else {
			printf("        filterSharedBuffer2[threadIdx.x + %d * %d] = filterTemp%d;\n", i, threadPerBlock, i + 1);
		}
	}
	puts("        __syncthreads();");
	puts("");
	puts("        // Exchange shared buffer 1 and shared buffer 2 and repeat");
	puts("        // load next group of Cnum channels");
	printf("        blockLoadInputStartIdx += %d * %d * %d;\n", inputSize, inputSize, channelGroupSize);
	for (int i = 0; i < inputTemp; ++i) {
		if (i == inputTemp - 1 && inputTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            inputTemp%d = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i + 1, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
			puts("        }");
		}
		else {
			printf("        inputTemp%d = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i + 1, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
		}
	}
	puts("");
	printf("        blockLoadFilterStartIdx += %d;\n", channelGroupSize);
	for (int i = 0; i < filterTemp; ++i) {
		if (i == filterTemp - 1 && filterTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            filterTemp%d = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i + 1, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
			puts("        }");
		}
		else {
			printf("        filterTemp%d = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i + 1, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
		}
	}
	puts("");
	puts("        // Copy operands from shared buffer 2 into Operands Registers");
	for (int i = 1; i <= inputOperand; ++i) {
		printf("        inputOperand%d = inputSharedBuffer2[((warpID / %d) %% %d) * %d + (laneID %% %d) * %d * %d + %d];\n", i, verticalRepeat, horizontalRepeat, outputWidthPerWarp, channelGroupSize, outputWidthPerWarp, horizontalRepeat, i - 1);
		if (i == inputOperand || i % 5 == 0) {
			puts("");
		}
	}
	for (int i = 1; i <= filterOperand; ++i) {
		printf("        filterOperand%d = filterSharedBuffer2[(warpID %% %d) * %d + laneID + %d * %d];\n", i, verticalRepeat, outputChannelPerWarp * channelGroupSize, i - 1, DCU_WARP_SIZE);
		if (i == filterOperand || i % 5 == 0) {
			puts("");
		}
	}
	puts("        // Compute and Accumulate result in Result Registers");
	for (int i = 1; i <= inputOperand; ++i) {
		for (int j = 1; j <= filterOperand; ++j) {
			printf("        input%dfilter%d += inputOperand%d * filterOperand%d;\n", i, j, i, j);
			if (j == filterOperand || j % 5 == 0) {
				puts("");
			}
		}
	}
	puts("        // Copy Temp Registers to shared buffer 1");
	for (int i = 0; i < inputTemp; ++i) {
		if (i == inputTemp - 1 && inputTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            inputSharedBuffer1[threadIdx.x + %d * %d] = inputTemp%d;\n", i, threadPerBlock, i + 1);
			puts("        }");
		}
		else {
			printf("        inputSharedBuffer1[threadIdx.x + %d * %d] = inputTemp%d;\n", i, threadPerBlock, i + 1);
		}
	}
	puts("");
	for (int i = 0; i < filterTemp; ++i) {
		if (i == filterTemp - 1 && filterTempReminder != 0) {
			printf("        if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
			printf("            filterSharedBuffer1[threadIdx.x + %d * %d] = filterTemp%d;\n", i, threadPerBlock, i + 1);
			puts("        }");
		}
		else {
			printf("        filterSharedBuffer1[threadIdx.x + %d * %d] = filterTemp%d;\n", i, threadPerBlock, i + 1);
		}
	}
	puts("        __syncthreads();");
	puts("    }");
	if (inputChannel % (2 * channelGroupSize) != 0) {
		puts("    // Copy operands from shared buffer 1 into Operands Registers");
		for (int i = 1; i <= inputOperand; ++i) {
			printf("    inputOperand%d = inputSharedBuffer1[((warpID / %d) %% %d) * %d + (laneID %% %d) * %d * %d + %d];\n", i, verticalRepeat, horizontalRepeat, outputWidthPerWarp, channelGroupSize, outputWidthPerWarp, horizontalRepeat, i - 1);
			if (i == inputOperand || i % 5 == 0) {
				puts("");
			}
		}
		for (int i = 1; i <= filterOperand; ++i) {
			printf("    filterOperand%d = filterSharedBuffer1[(warpID %% %d) * %d + laneID + %d * %d];\n", i, verticalRepeat, outputChannelPerWarp * channelGroupSize, i - 1, DCU_WARP_SIZE);
			if (i == filterOperand || i % 5 == 0) {
				puts("");
			}
		}
		puts("    // Compute and Accumulate result in Result Registers");
		for (int i = 1; i <= inputOperand; ++i) {
			for (int j = 1; j <= filterOperand; ++j) {
				printf("    input%dfilter%d += inputOperand%d * filterOperand%d;\n", i, j, i, j);
				if (j == filterOperand || j % 5 == 0) {
					puts("");
				}
			}
		}
		puts("    __syncthreads();");
	}
	else {
		puts("    // load next group of Cnum channels");
		printf("    blockLoadInputStartIdx += %d * %d * %d;\n", inputSize, inputSize, channelGroupSize);
		for (int i = 0; i < inputTemp; ++i) {
			if (i == inputTemp - 1 && inputTempReminder != 0) {
				printf("    if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
				printf("        inputTemp%d = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i + 1, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
				puts("    }");
			}
			else {
				printf("    inputTemp%d = input[blockLoadInputStartIdx + (threadIdx.x + %d * %d) / %d * %d + ((threadIdx.x + %d * %d) %% %d) / %d * %d + (threadIdx.x + %d * %d) %% %d];\n", i + 1, i, threadPerBlock, blockLayer, inputLayer, i, threadPerBlock, blockLayer, outputWidthPerWarp * times, inputSize, i, threadPerBlock, outputWidthPerWarp * times);
			}
		}
		puts("");
		printf("    blockLoadFilterStartIdx += %d;\n", channelGroupSize);
		for (int i = 0; i < filterTemp; ++i) {
			if (i == filterTemp - 1 && filterTempReminder != 0) {
				printf("    if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
				printf("        filterTemp%d = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i + 1, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
				puts("    }");
			}
			else {
				printf("    filterTemp%d = filter[blockLoadFilterStartIdx + ((threadIdx.x + %d * %d) / %d) * %d + ((threadIdx.x + %d * %d) %% %d)];\n", i + 1, i, threadPerBlock, channelGroupSize, inputChannel, i, threadPerBlock, channelGroupSize);
			}
		}
		puts("");
		puts("    // Copy operands from shared buffer 1 into Operands Registers");
		for (int i = 1; i <= inputOperand; ++i) {
			printf("    inputOperand%d = inputSharedBuffer1[((warpID / %d) %% %d) * %d + (laneID %% %d) * %d * %d + %d];\n", i, verticalRepeat, horizontalRepeat, outputWidthPerWarp, channelGroupSize, outputWidthPerWarp, horizontalRepeat, i - 1);
			if (i == inputOperand || i % 5 == 0) {
				puts("");
			}
		}
		for (int i = 1; i <= filterOperand; ++i) {
			printf("    filterOperand%d = filterSharedBuffer1[(warpID %% %d) * %d + laneID + %d * %d];\n", i, verticalRepeat, outputChannelPerWarp * channelGroupSize, i - 1, DCU_WARP_SIZE);
			if (i == filterOperand || i % 5 == 0) {
				puts("");
			}
		}
		puts("    // Compute and Accumulate result in Result Registers");
		for (int i = 1; i <= inputOperand; ++i) {
			for (int j = 1; j <= filterOperand; ++j) {
				printf("    input%dfilter%d += inputOperand%d * filterOperand%d;\n", i, j, i, j);
				if (j == filterOperand || j % 5 == 0) {
					puts("");
				}
			}
		}
		puts("    // Copy Temp Registers to shared buffer 2");
		for (int i = 0; i < inputTemp; ++i) {
			if (i == inputTemp - 1 && inputTempReminder != 0) {
				printf("    if(threadIdx.x < %d * %d * %d - %d * %d) {\n", horizontalRepeat, outputWidthPerWarp, channelGroupSize, i, threadPerBlock);
				printf("        inputSharedBuffer2[threadIdx.x + %d * %d] = inputTemp%d;\n", i, threadPerBlock, i + 1);
				puts("    }");
			}
			else {
				printf("    inputSharedBuffer2[threadIdx.x + %d * %d] = inputTemp%d;\n", i, threadPerBlock, i + 1);
			}
		}
		puts("");
		for (int i = 0; i < filterTemp; ++i) {
			if (i == filterTemp - 1 && filterTempReminder != 0) {
				printf("    if(threadIdx.x < %d * %d * %d - %d * %d) {\n", verticalRepeat, outputChannelPerWarp, channelGroupSize, i, threadPerBlock);
				printf("        filterSharedBuffer2[threadIdx.x + %d * %d] = filterTemp%d;\n", i, threadPerBlock, i + 1);
				puts("    }");
			}
			else {
				printf("    filterSharedBuffer2[threadIdx.x + %d * %d] = filterTemp%d;\n", i, threadPerBlock, i + 1);
			}
		}
		puts("    __syncthreads();");
		puts("");
		puts("    // Exchange shared buffer 1 and shared buffer 2 and repeat");
		puts("    // Copy operands from shared buffer 2 into Operands Registers");
		for (int i = 1; i <= inputOperand; ++i) {
			printf("    inputOperand%d = inputSharedBuffer2[((warpID / %d) %% %d) * %d + (laneID %% %d) * %d * %d + %d];\n", i, verticalRepeat, horizontalRepeat, outputWidthPerWarp, channelGroupSize, outputWidthPerWarp, horizontalRepeat, i - 1);
			if (i == inputOperand || i % 5 == 0) {
				puts("");
			}
		}
		for (int i = 1; i <= filterOperand; ++i) {
			printf("    filterOperand%d = filterSharedBuffer2[(warpID %% %d) * %d + laneID + %d * %d];\n", i, verticalRepeat, outputChannelPerWarp * channelGroupSize, i - 1, DCU_WARP_SIZE);
			if (i == filterOperand || i % 5 == 0) {
				puts("");
			}
		}
		puts("    // Compute and Accumulate result in Result Registers");
		for (int i = 1; i <= inputOperand; ++i) {
			for (int j = 1; j <= filterOperand; ++j) {
				printf("    input%dfilter%d += inputOperand%d * filterOperand%d;\n", i, j, i, j);
				if (j == filterOperand || j % 5 == 0) {
					puts("");
				}
			}
		}
		puts("    __syncthreads();");
	}
	puts("    // For loop ends here");
	puts("");
	puts("    // Parallel Reduction to accumulate result across threads");
	puts("    // Cnum threads from one group");
	puts("    #pragma unroll");
	printf("    for (int offset = (%d >> 1); offset > 0; offset >>= 1) {\n", channelGroupSize);
	for (int i = 1; i <= inputOperand; ++i) {
		for (int j = 1; j <= filterOperand; ++j) {
			printf("        input%dfilter%d += __shfl_down(input%dfilter%d, offset, %d);\n", i, j, i, j, channelGroupSize);
			if (j == filterOperand || j % 5 == 0) {
				if (i != inputOperand || j != filterOperand) {
					puts("");
				}
			}
		}
	}
	puts("    }");
	puts("");
	puts("    // Store output");
	int outputPicSize = inputSize * inputSize * outputChannel;
	int outputPerBlockOffset = inputLayer * outputChannelPerWarp * verticalRepeat;
	printf("    int blockWriteOutputStartIdx = blockIdx.x / %d * %d + (blockIdx.x %% %d) / %d * %d + (blockIdx.x %% %d) / %d * %d + (blockIdx.x %% %d) * %d;\n", inputBlockPerBatch, outputPicSize, inputBlockPerBatch, inputBlockPerY, inputSize * (horizontalRepeat / times), inputBlockPerY, inputRepeat, outputWidthPerWarp * times, inputRepeat, outputPerBlockOffset);
	puts("");
	printf("    if(laneID %% %d == 0) {\n", channelGroupSize);
	for (int i = 1; i <= filterOperand; i++) {
		for (int j = 1; j <= inputOperand; j++) {
			printf("        output[blockWriteOutputStartIdx + %d * %d * outputHeight * outputWidth + (warpID / %d) / %d * outputWidth + ((warpID / %d) %% %d) * %d + (warpID %% %d) * %d + (laneID / %d) * outputHeight * outputWidth + %d] = input%dfilter%d;\n", i - 1, DCU_WARP_SIZE / channelGroupSize, verticalRepeat, times, verticalRepeat, times, outputWidthPerWarp, verticalRepeat, outputChannelPerWarp * inputSize * inputSize, channelGroupSize, j - 1, j, i);
			if (j == inputOperand && i != filterOperand) {
				puts("");
			}
		}
	}
	puts("    }");
	puts("}");
}

int main(int argc,char *argv[]) {
	init(argv);
	print_header();
	//print_comment();
	print_kernel();
	return 0;
}

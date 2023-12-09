/*
Pointwise Convolution Kernel
InputBatch_32_Input_7x7_InChannel_160_OutChannel_960

Grid:
    gridDim.x = (32 * 960 * 7 * 7) / (7 * 7 * 96);
Block:
    blockDim.x = 64 * 7;

warpNumPerBlock = 7
outputWidthPerWarp = 7
outputChannelPerWarp = 96
channelGroupSize = 2
horizontalRepeat = 7
verticalRepeat = 1

One thread block contains 7 warps, 7 * 64 = 448 threads.
Each thread block is responsible for generating 7 * 7 * 96 output data.
Each warp is responsible for generating 7 * 96 output data.
*/

__global__ __launch_bounds__(1024) void InputBatch_32_Input_7x7_InChannel_160_OutChannel_960(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[7 * 7 * 2];
    __shared__ float inputSharedBuffer2[7 * 7 * 2];

    __shared__ float filterSharedBuffer1[1 * 96 * 2];
    __shared__ float filterSharedBuffer2[1 * 96 * 2];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 10 * 7840 + (blockIdx.x % 10) / 10 * 49 + (blockIdx.x % 10) / 10 * 7;
    if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
        inputSharedBuffer1[threadIdx.x + 0 * 448] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 49 * 49 + ((threadIdx.x + 0 * 448) % 49) / 7 * 7 + (threadIdx.x + 0 * 448) % 7];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 10) * 15360;
    if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
        filterSharedBuffer1[threadIdx.x + 0 * 448] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 2) * 160 + ((threadIdx.x + 0 * 448) % 2)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 2) / (2 * 2); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 2;
        if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 49 * 49 + ((threadIdx.x + 0 * 448) % 49) / 7 * 7 + (threadIdx.x + 0 * 448) % 7];
        }

        blockLoadFilterStartIdx += 2;
        if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 2) * 160 + ((threadIdx.x + 0 * 448) % 2)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 192 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 1) * 192 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 1) * 192 + laneID + 2 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
            inputSharedBuffer2[threadIdx.x + 0 * 448] = inputTemp1;
        }

        if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
            filterSharedBuffer2[threadIdx.x + 0 * 448] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 2;
        if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 49 * 49 + ((threadIdx.x + 0 * 448) % 49) / 7 * 7 + (threadIdx.x + 0 * 448) % 7];
        }

        blockLoadFilterStartIdx += 2;
        if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 2) * 160 + ((threadIdx.x + 0 * 448) % 2)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 6];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 192 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 1) * 192 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 1) * 192 + laneID + 2 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
            inputSharedBuffer1[threadIdx.x + 0 * 448] = inputTemp1;
        }

        if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
            filterSharedBuffer1[threadIdx.x + 0 * 448] = filterTemp1;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 7 * 7 * 2;
    if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 49 * 49 + ((threadIdx.x + 0 * 448) % 49) / 7 * 7 + (threadIdx.x + 0 * 448) % 7];
    }

    blockLoadFilterStartIdx += 2;
    if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 2) * 160 + ((threadIdx.x + 0 * 448) % 2)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 6];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 192 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 1) * 192 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 1) * 192 + laneID + 2 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 7 * 7 * 2 - 0 * 448) {
        inputSharedBuffer2[threadIdx.x + 0 * 448] = inputTemp1;
    }

    if(threadIdx.x < 1 * 96 * 2 - 0 * 448) {
        filterSharedBuffer2[threadIdx.x + 0 * 448] = filterTemp1;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 1) % 7) * 7 + (laneID % 2) * 7 * 7 + 6];

    filterOperand1 = filterSharedBuffer2[(warpID % 1) * 192 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer2[(warpID % 1) * 192 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer2[(warpID % 1) * 192 + laneID + 2 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (2 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 2);
        input1filter2 += __shfl_down(input1filter2, offset, 2);
        input1filter3 += __shfl_down(input1filter3, offset, 2);

        input2filter1 += __shfl_down(input2filter1, offset, 2);
        input2filter2 += __shfl_down(input2filter2, offset, 2);
        input2filter3 += __shfl_down(input2filter3, offset, 2);

        input3filter1 += __shfl_down(input3filter1, offset, 2);
        input3filter2 += __shfl_down(input3filter2, offset, 2);
        input3filter3 += __shfl_down(input3filter3, offset, 2);

        input4filter1 += __shfl_down(input4filter1, offset, 2);
        input4filter2 += __shfl_down(input4filter2, offset, 2);
        input4filter3 += __shfl_down(input4filter3, offset, 2);

        input5filter1 += __shfl_down(input5filter1, offset, 2);
        input5filter2 += __shfl_down(input5filter2, offset, 2);
        input5filter3 += __shfl_down(input5filter3, offset, 2);

        input6filter1 += __shfl_down(input6filter1, offset, 2);
        input6filter2 += __shfl_down(input6filter2, offset, 2);
        input6filter3 += __shfl_down(input6filter3, offset, 2);

        input7filter1 += __shfl_down(input7filter1, offset, 2);
        input7filter2 += __shfl_down(input7filter2, offset, 2);
        input7filter3 += __shfl_down(input7filter3, offset, 2);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 10 * 47040 + (blockIdx.x % 10) / 10 * 49 + (blockIdx.x % 10) / 10 * 7 + (blockIdx.x % 10) * 4704;

    if(laneID % 2 == 0) {
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 6] = input7filter2;

        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 7 + (warpID % 1) * 4704 + (laneID / 2) * outputHeight * outputWidth + 6] = input7filter3;
    }
}

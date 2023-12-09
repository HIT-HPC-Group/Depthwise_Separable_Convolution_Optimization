/*
Pointwise Convolution Kernel
InputBatch_1_Input_56x56_InChannel_144_OutChannel_24

Grid:
    gridDim.x = (1 * 24 * 56 * 56) / (7 * 4 * 24);
Block:
    blockDim.x = 64 * 7;

warpNumPerBlock = 7
outputWidthPerWarp = 4
outputChannelPerWarp = 24
channelGroupSize = 16
horizontalRepeat = 7
verticalRepeat = 1

One thread block contains 7 warps, 7 * 64 = 448 threads.
Each thread block is responsible for generating 7 * 4 * 24 output data.
Each warp is responsible for generating 4 * 24 output data.
*/
template <typename scalar_t>
__global__ __launch_bounds__(1024) void InputBatch_1_Input_56x56_InChannel_144_OutChannel_24(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[7 * 4 * 16];
    __shared__ float inputSharedBuffer2[7 * 4 * 16];

    __shared__ float filterSharedBuffer1[1 * 24 * 16];
    __shared__ float filterSharedBuffer2[1 * 24 * 16];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0;
    float input1filter6 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 112 * 451584 + (blockIdx.x % 112) / 14 * 392 + (blockIdx.x % 14) / 1 * 4;
    inputSharedBuffer1[threadIdx.x + 0 * 448] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 28 * 3136 + ((threadIdx.x + 0 * 448) % 28) / 4 * 56 + (threadIdx.x + 0 * 448) % 4];

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 1) * 3456;
    if(threadIdx.x < 1 * 24 * 16 - 0 * 448) {
        filterSharedBuffer1[threadIdx.x + 0 * 448] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 16) * 144 + ((threadIdx.x + 0 * 448) % 16)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 16) / (2 * 16); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 56 * 56 * 16;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 28 * 3136 + ((threadIdx.x + 0 * 448) % 28) / 4 * 56 + (threadIdx.x + 0 * 448) % 4];

        blockLoadFilterStartIdx += 16;
        if(threadIdx.x < 1 * 24 * 16 - 0 * 448) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 16) * 144 + ((threadIdx.x + 0 * 448) % 16)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 3];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 5 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;

        // Copy Temp Registers to shared buffer 2
        inputSharedBuffer2[threadIdx.x + 0 * 448] = inputTemp1;

        if(threadIdx.x < 1 * 24 * 16 - 0 * 448) {
            filterSharedBuffer2[threadIdx.x + 0 * 448] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 56 * 56 * 16;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 448) / 28 * 3136 + ((threadIdx.x + 0 * 448) % 28) / 4 * 56 + (threadIdx.x + 0 * 448) % 4];

        blockLoadFilterStartIdx += 16;
        if(threadIdx.x < 1 * 24 * 16 - 0 * 448) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 448) / 16) * 144 + ((threadIdx.x + 0 * 448) % 16)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 3];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 384 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 1) * 384 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 1) * 384 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 1) * 384 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 1) * 384 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer2[(warpID % 1) * 384 + laneID + 5 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;

        // Copy Temp Registers to shared buffer 1
        inputSharedBuffer1[threadIdx.x + 0 * 448] = inputTemp1;

        if(threadIdx.x < 1 * 24 * 16 - 0 * 448) {
            filterSharedBuffer1[threadIdx.x + 0 * 448] = filterTemp1;
        }
        __syncthreads();
    }
    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 7) * 4 + (laneID % 16) * 4 * 7 + 3];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer1[(warpID % 1) * 384 + laneID + 5 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input1filter6 += inputOperand1 * filterOperand6;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input2filter6 += inputOperand2 * filterOperand6;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input3filter6 += inputOperand3 * filterOperand6;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input4filter6 += inputOperand4 * filterOperand6;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (16 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 16);
        input1filter2 += __shfl_down(input1filter2, offset, 16);
        input1filter3 += __shfl_down(input1filter3, offset, 16);
        input1filter4 += __shfl_down(input1filter4, offset, 16);
        input1filter5 += __shfl_down(input1filter5, offset, 16);

        input1filter6 += __shfl_down(input1filter6, offset, 16);

        input2filter1 += __shfl_down(input2filter1, offset, 16);
        input2filter2 += __shfl_down(input2filter2, offset, 16);
        input2filter3 += __shfl_down(input2filter3, offset, 16);
        input2filter4 += __shfl_down(input2filter4, offset, 16);
        input2filter5 += __shfl_down(input2filter5, offset, 16);

        input2filter6 += __shfl_down(input2filter6, offset, 16);

        input3filter1 += __shfl_down(input3filter1, offset, 16);
        input3filter2 += __shfl_down(input3filter2, offset, 16);
        input3filter3 += __shfl_down(input3filter3, offset, 16);
        input3filter4 += __shfl_down(input3filter4, offset, 16);
        input3filter5 += __shfl_down(input3filter5, offset, 16);

        input3filter6 += __shfl_down(input3filter6, offset, 16);

        input4filter1 += __shfl_down(input4filter1, offset, 16);
        input4filter2 += __shfl_down(input4filter2, offset, 16);
        input4filter3 += __shfl_down(input4filter3, offset, 16);
        input4filter4 += __shfl_down(input4filter4, offset, 16);
        input4filter5 += __shfl_down(input4filter5, offset, 16);

        input4filter6 += __shfl_down(input4filter6, offset, 16);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 112 * 75264 + (blockIdx.x % 112) / 14 * 392 + (blockIdx.x % 14) / 1 * 4 + (blockIdx.x % 1) * 75264;

    if(laneID % 16 == 0) {
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 3] = input4filter1;

        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 3] = input4filter2;

        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 3] = input4filter3;

        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 3] = input4filter4;

        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 3] = input4filter5;

        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 4 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 75264 + (laneID / 16) * outputHeight * outputWidth + 3] = input4filter6;
    }
}

/*
Pointwise Convolution Kernel
InputBatch_32_Input_56x56_InChannel_24_OutChannel_144

Grid:
    gridDim.x = (32 * 144 * 56 * 56) / (4 * 4 * 144);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 4
outputChannelPerWarp = 144
channelGroupSize = 4
horizontalRepeat = 4
verticalRepeat = 1

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 4 * 144 output data.
Each warp is responsible for generating 4 * 144 output data.
*/
template <typename scalar_t>
__global__ void InputBatch_32_Input_56x56_InChannel_24_OutChannel_144(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[4 * 4 * 4];
    __shared__ float inputSharedBuffer2[4 * 4 * 4];

    __shared__ float filterSharedBuffer1[1 * 144 * 4];
    __shared__ float filterSharedBuffer2[1 * 144 * 4];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0, filterTemp2 = 0, filterTemp3 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0, filterOperand7 = 0, filterOperand8 = 0, filterOperand9 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0;
    float input1filter6 = 0, input1filter7 = 0, input1filter8 = 0, input1filter9 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0, input2filter7 = 0, input2filter8 = 0, input2filter9 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0, input3filter7 = 0, input3filter8 = 0, input3filter9 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0, input4filter7 = 0, input4filter8 = 0, input4filter9 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 196 * 75264 + (blockIdx.x % 196) / 14 * 224 + (blockIdx.x % 14) / 1 * 4;
    if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 3136 + ((threadIdx.x + 0 * 256) % 16) / 4 * 56 + (threadIdx.x + 0 * 256) % 4];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 1) * 3456;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 24 + ((threadIdx.x + 0 * 256) % 4)];
    filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 24 + ((threadIdx.x + 1 * 256) % 4)];
    if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
        filterSharedBuffer1[threadIdx.x + 2 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 4) * 24 + ((threadIdx.x + 2 * 256) % 4)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 4) / (2 * 4); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 56 * 56 * 4;
        if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 3136 + ((threadIdx.x + 0 * 256) % 16) / 4 * 56 + (threadIdx.x + 0 * 256) % 4];
        }

        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 24 + ((threadIdx.x + 0 * 256) % 4)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 24 + ((threadIdx.x + 1 * 256) % 4)];
        if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
            filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 4) * 24 + ((threadIdx.x + 2 * 256) % 4)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 3];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 7 * 64];
        filterOperand9 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 8 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
            filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 56 * 56 * 4;
        if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 3136 + ((threadIdx.x + 0 * 256) % 16) / 4 * 56 + (threadIdx.x + 0 * 256) % 4];
        }

        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 24 + ((threadIdx.x + 0 * 256) % 4)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 24 + ((threadIdx.x + 1 * 256) % 4)];
        if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
            filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 4) * 24 + ((threadIdx.x + 2 * 256) % 4)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 3];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 7 * 64];
        filterOperand9 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 8 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;
        input1filter9 += inputOperand1 * filterOperand9;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
            filterSharedBuffer1[threadIdx.x + 2 * 256] = filterTemp3;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 56 * 56 * 4;
    if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 16 * 3136 + ((threadIdx.x + 0 * 256) % 16) / 4 * 56 + (threadIdx.x + 0 * 256) % 4];
    }

    blockLoadFilterStartIdx += 4;
    filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 24 + ((threadIdx.x + 0 * 256) % 4)];
    filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 24 + ((threadIdx.x + 1 * 256) % 4)];
    if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 4) * 24 + ((threadIdx.x + 2 * 256) % 4)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 3];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 7 * 64];
    filterOperand9 = filterSharedBuffer1[(warpID % 1) * 576 + laneID + 8 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input1filter6 += inputOperand1 * filterOperand6;
    input1filter7 += inputOperand1 * filterOperand7;
    input1filter8 += inputOperand1 * filterOperand8;
    input1filter9 += inputOperand1 * filterOperand9;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input2filter6 += inputOperand2 * filterOperand6;
    input2filter7 += inputOperand2 * filterOperand7;
    input2filter8 += inputOperand2 * filterOperand8;
    input2filter9 += inputOperand2 * filterOperand9;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input3filter6 += inputOperand3 * filterOperand6;
    input3filter7 += inputOperand3 * filterOperand7;
    input3filter8 += inputOperand3 * filterOperand8;
    input3filter9 += inputOperand3 * filterOperand9;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input4filter6 += inputOperand4 * filterOperand6;
    input4filter7 += inputOperand4 * filterOperand7;
    input4filter8 += inputOperand4 * filterOperand8;
    input4filter9 += inputOperand4 * filterOperand9;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 4 * 4 * 4 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
    if(threadIdx.x < 1 * 144 * 4 - 2 * 256) {
        filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 4 + (laneID % 4) * 4 * 4 + 3];

    filterOperand1 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 7 * 64];
    filterOperand9 = filterSharedBuffer2[(warpID % 1) * 576 + laneID + 8 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input1filter6 += inputOperand1 * filterOperand6;
    input1filter7 += inputOperand1 * filterOperand7;
    input1filter8 += inputOperand1 * filterOperand8;
    input1filter9 += inputOperand1 * filterOperand9;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input2filter6 += inputOperand2 * filterOperand6;
    input2filter7 += inputOperand2 * filterOperand7;
    input2filter8 += inputOperand2 * filterOperand8;
    input2filter9 += inputOperand2 * filterOperand9;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input3filter6 += inputOperand3 * filterOperand6;
    input3filter7 += inputOperand3 * filterOperand7;
    input3filter8 += inputOperand3 * filterOperand8;
    input3filter9 += inputOperand3 * filterOperand9;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input4filter6 += inputOperand4 * filterOperand6;
    input4filter7 += inputOperand4 * filterOperand7;
    input4filter8 += inputOperand4 * filterOperand8;
    input4filter9 += inputOperand4 * filterOperand9;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (4 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 4);
        input1filter2 += __shfl_down(input1filter2, offset, 4);
        input1filter3 += __shfl_down(input1filter3, offset, 4);
        input1filter4 += __shfl_down(input1filter4, offset, 4);
        input1filter5 += __shfl_down(input1filter5, offset, 4);

        input1filter6 += __shfl_down(input1filter6, offset, 4);
        input1filter7 += __shfl_down(input1filter7, offset, 4);
        input1filter8 += __shfl_down(input1filter8, offset, 4);
        input1filter9 += __shfl_down(input1filter9, offset, 4);

        input2filter1 += __shfl_down(input2filter1, offset, 4);
        input2filter2 += __shfl_down(input2filter2, offset, 4);
        input2filter3 += __shfl_down(input2filter3, offset, 4);
        input2filter4 += __shfl_down(input2filter4, offset, 4);
        input2filter5 += __shfl_down(input2filter5, offset, 4);

        input2filter6 += __shfl_down(input2filter6, offset, 4);
        input2filter7 += __shfl_down(input2filter7, offset, 4);
        input2filter8 += __shfl_down(input2filter8, offset, 4);
        input2filter9 += __shfl_down(input2filter9, offset, 4);

        input3filter1 += __shfl_down(input3filter1, offset, 4);
        input3filter2 += __shfl_down(input3filter2, offset, 4);
        input3filter3 += __shfl_down(input3filter3, offset, 4);
        input3filter4 += __shfl_down(input3filter4, offset, 4);
        input3filter5 += __shfl_down(input3filter5, offset, 4);

        input3filter6 += __shfl_down(input3filter6, offset, 4);
        input3filter7 += __shfl_down(input3filter7, offset, 4);
        input3filter8 += __shfl_down(input3filter8, offset, 4);
        input3filter9 += __shfl_down(input3filter9, offset, 4);

        input4filter1 += __shfl_down(input4filter1, offset, 4);
        input4filter2 += __shfl_down(input4filter2, offset, 4);
        input4filter3 += __shfl_down(input4filter3, offset, 4);
        input4filter4 += __shfl_down(input4filter4, offset, 4);
        input4filter5 += __shfl_down(input4filter5, offset, 4);

        input4filter6 += __shfl_down(input4filter6, offset, 4);
        input4filter7 += __shfl_down(input4filter7, offset, 4);
        input4filter8 += __shfl_down(input4filter8, offset, 4);
        input4filter9 += __shfl_down(input4filter9, offset, 4);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 196 * 451584 + (blockIdx.x % 196) / 14 * 224 + (blockIdx.x % 14) / 1 * 4 + (blockIdx.x % 1) * 451584;

    if(laneID % 4 == 0) {
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter1;

        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter2;

        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter3;

        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter4;

        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter5;

        output[blockWriteOutputStartIdx + 5 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter6;

        output[blockWriteOutputStartIdx + 6 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter7;

        output[blockWriteOutputStartIdx + 7 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter8;

        output[blockWriteOutputStartIdx + 8 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter9;
        output[blockWriteOutputStartIdx + 8 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter9;
        output[blockWriteOutputStartIdx + 8 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter9;
        output[blockWriteOutputStartIdx + 8 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 4 + (warpID % 1) * 451584 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter9;
    }
}

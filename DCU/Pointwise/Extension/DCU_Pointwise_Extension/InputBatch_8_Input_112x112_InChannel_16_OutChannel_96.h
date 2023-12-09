/*
Pointwise Convolution Kernel
InputBatch_8_Input_112x112_InChannel_16_OutChannel_96

Grid:
    gridDim.x = (8 * 96 * 112 * 112) / (4 * 8 * 96);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 8
outputChannelPerWarp = 96
channelGroupSize = 2
horizontalRepeat = 4
verticalRepeat = 1

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 8 * 96 output data.
Each warp is responsible for generating 8 * 96 output data.
*/
template <typename scalar_t>
__global__ void InputBatch_8_Input_112x112_InChannel_16_OutChannel_96(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[4 * 8 * 2];
    __shared__ float inputSharedBuffer2[4 * 8 * 2];

    __shared__ float filterSharedBuffer1[1 * 96 * 2];
    __shared__ float filterSharedBuffer2[1 * 96 * 2];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0, inputOperand8 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0;

    float input8filter1 = 0, input8filter2 = 0, input8filter3 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 392 * 200704 + (blockIdx.x % 392) / 14 * 448 + (blockIdx.x % 14) / 1 * 8;
    if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 32 * 12544 + ((threadIdx.x + 0 * 256) % 32) / 8 * 112 + (threadIdx.x + 0 * 256) % 8];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 1) * 1536;
    if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
        filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 2) * 16 + ((threadIdx.x + 0 * 256) % 2)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 2) / (2 * 2); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 112 * 112 * 2;
        if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 32 * 12544 + ((threadIdx.x + 0 * 256) % 32) / 8 * 112 + (threadIdx.x + 0 * 256) % 8];
        }

        blockLoadFilterStartIdx += 2;
        if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 2) * 16 + ((threadIdx.x + 0 * 256) % 2)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 6];
        inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 7];

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

        input8filter1 += inputOperand8 * filterOperand1;
        input8filter2 += inputOperand8 * filterOperand2;
        input8filter3 += inputOperand8 * filterOperand3;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
            filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 112 * 112 * 2;
        if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 32 * 12544 + ((threadIdx.x + 0 * 256) % 32) / 8 * 112 + (threadIdx.x + 0 * 256) % 8];
        }

        blockLoadFilterStartIdx += 2;
        if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 2) * 16 + ((threadIdx.x + 0 * 256) % 2)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 6];
        inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 7];

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

        input8filter1 += inputOperand8 * filterOperand1;
        input8filter2 += inputOperand8 * filterOperand2;
        input8filter3 += inputOperand8 * filterOperand3;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
            filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 112 * 112 * 2;
    if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 32 * 12544 + ((threadIdx.x + 0 * 256) % 32) / 8 * 112 + (threadIdx.x + 0 * 256) % 8];
    }

    blockLoadFilterStartIdx += 2;
    if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 2) * 16 + ((threadIdx.x + 0 * 256) % 2)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 6];
    inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 7];

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

    input8filter1 += inputOperand8 * filterOperand1;
    input8filter2 += inputOperand8 * filterOperand2;
    input8filter3 += inputOperand8 * filterOperand3;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 4 * 8 * 2 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    if(threadIdx.x < 1 * 96 * 2 - 0 * 256) {
        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 6];
    inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 8 + (laneID % 2) * 8 * 4 + 7];

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

    input8filter1 += inputOperand8 * filterOperand1;
    input8filter2 += inputOperand8 * filterOperand2;
    input8filter3 += inputOperand8 * filterOperand3;

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

        input8filter1 += __shfl_down(input8filter1, offset, 2);
        input8filter2 += __shfl_down(input8filter2, offset, 2);
        input8filter3 += __shfl_down(input8filter3, offset, 2);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 392 * 1204224 + (blockIdx.x % 392) / 14 * 448 + (blockIdx.x % 14) / 1 * 8 + (blockIdx.x % 1) * 1204224;

    if(laneID % 2 == 0) {
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 6] = input7filter1;
        output[blockWriteOutputStartIdx + 0 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 7] = input8filter1;

        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 6] = input7filter2;
        output[blockWriteOutputStartIdx + 1 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 7] = input8filter2;

        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 6] = input7filter3;
        output[blockWriteOutputStartIdx + 2 * 32 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 8 + (warpID % 1) * 1204224 + (laneID / 2) * outputHeight * outputWidth + 7] = input8filter3;
    }
}

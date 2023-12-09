/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_960_OutChannel_160

Grid:
    gridDim.x = (8 * 160 * 7 * 7) / (4 * 7 * 8);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 7
outputChannelPerWarp = 8
channelGroupSize = 64
horizontalRepeat = 1
verticalRepeat = 4

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 7 * 8 output data.
Each warp is responsible for generating 7 * 8 output data.
*/
template <typename scalar_t>
__global__ void InputBatch_8_Input_7x7_InChannel_960_OutChannel_160(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[1 * 7 * 64];
    __shared__ float inputSharedBuffer2[1 * 7 * 64];

    __shared__ float filterSharedBuffer1[4 * 8 * 64];
    __shared__ float filterSharedBuffer2[4 * 8 * 64];

    // to hold loaded operands temp.
    float inputTemp1 = 0, inputTemp2 = 0;
    float filterTemp1 = 0, filterTemp2 = 0, filterTemp3 = 0, filterTemp4 = 0, filterTemp5 = 0;
    float filterTemp6 = 0, filterTemp7 = 0, filterTemp8 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0, filterOperand7 = 0, filterOperand8 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0;
    float input1filter6 = 0, input1filter7 = 0, input1filter8 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0, input2filter7 = 0, input2filter8 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0, input3filter7 = 0, input3filter8 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0, input4filter7 = 0, input4filter8 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0, input5filter4 = 0, input5filter5 = 0;
    float input5filter6 = 0, input5filter7 = 0, input5filter8 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0, input6filter4 = 0, input6filter5 = 0;
    float input6filter6 = 0, input6filter7 = 0, input6filter8 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0, input7filter4 = 0, input7filter5 = 0;
    float input7filter6 = 0, input7filter7 = 0, input7filter8 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 35 * 47040 + (blockIdx.x % 35) / 5 * 7 + (blockIdx.x % 5) / 5 * 7;
    inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
    if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
        inputSharedBuffer1[threadIdx.x + 1 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 7 * 49 + ((threadIdx.x + 1 * 256) % 7) / 7 * 7 + (threadIdx.x + 1 * 256) % 7];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 5) * 30720;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 64) * 960 + ((threadIdx.x + 0 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 64) * 960 + ((threadIdx.x + 1 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 2 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 64) * 960 + ((threadIdx.x + 2 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 3 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 64) * 960 + ((threadIdx.x + 3 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 4 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 64) * 960 + ((threadIdx.x + 4 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 5 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 64) * 960 + ((threadIdx.x + 5 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 6 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 64) * 960 + ((threadIdx.x + 6 * 256) % 64)];
    filterSharedBuffer1[threadIdx.x + 7 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 64) * 960 + ((threadIdx.x + 7 * 256) % 64)];
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 64) / (2 * 64); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 64;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 7 * 49 + ((threadIdx.x + 1 * 256) % 7) / 7 * 7 + (threadIdx.x + 1 * 256) % 7];
        }

        blockLoadFilterStartIdx += 64;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 64) * 960 + ((threadIdx.x + 0 * 256) % 64)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 64) * 960 + ((threadIdx.x + 1 * 256) % 64)];
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 64) * 960 + ((threadIdx.x + 2 * 256) % 64)];
        filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 64) * 960 + ((threadIdx.x + 3 * 256) % 64)];
        filterTemp5 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 64) * 960 + ((threadIdx.x + 4 * 256) % 64)];
        filterTemp6 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 64) * 960 + ((threadIdx.x + 5 * 256) % 64)];
        filterTemp7 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 64) * 960 + ((threadIdx.x + 6 * 256) % 64)];
        filterTemp8 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 64) * 960 + ((threadIdx.x + 7 * 256) % 64)];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 6];

        filterOperand1 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 7 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;

        // Copy Temp Registers to shared buffer 2
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
        filterSharedBuffer2[threadIdx.x + 3 * 256] = filterTemp4;
        filterSharedBuffer2[threadIdx.x + 4 * 256] = filterTemp5;
        filterSharedBuffer2[threadIdx.x + 5 * 256] = filterTemp6;
        filterSharedBuffer2[threadIdx.x + 6 * 256] = filterTemp7;
        filterSharedBuffer2[threadIdx.x + 7 * 256] = filterTemp8;
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 64;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 7 * 49 + ((threadIdx.x + 1 * 256) % 7) / 7 * 7 + (threadIdx.x + 1 * 256) % 7];
        }

        blockLoadFilterStartIdx += 64;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 64) * 960 + ((threadIdx.x + 0 * 256) % 64)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 64) * 960 + ((threadIdx.x + 1 * 256) % 64)];
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 64) * 960 + ((threadIdx.x + 2 * 256) % 64)];
        filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 64) * 960 + ((threadIdx.x + 3 * 256) % 64)];
        filterTemp5 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 64) * 960 + ((threadIdx.x + 4 * 256) % 64)];
        filterTemp6 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 64) * 960 + ((threadIdx.x + 5 * 256) % 64)];
        filterTemp7 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 64) * 960 + ((threadIdx.x + 6 * 256) % 64)];
        filterTemp8 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 64) * 960 + ((threadIdx.x + 7 * 256) % 64)];

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 6];

        filterOperand1 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer2[(warpID % 4) * 512 + laneID + 7 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input1filter6 += inputOperand1 * filterOperand6;
        input1filter7 += inputOperand1 * filterOperand7;
        input1filter8 += inputOperand1 * filterOperand8;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;

        // Copy Temp Registers to shared buffer 1
        inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        if(threadIdx.x < 1 * 7 * 64 - 1 * 256) {
            inputSharedBuffer1[threadIdx.x + 1 * 256] = inputTemp2;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        filterSharedBuffer1[threadIdx.x + 2 * 256] = filterTemp3;
        filterSharedBuffer1[threadIdx.x + 3 * 256] = filterTemp4;
        filterSharedBuffer1[threadIdx.x + 4 * 256] = filterTemp5;
        filterSharedBuffer1[threadIdx.x + 5 * 256] = filterTemp6;
        filterSharedBuffer1[threadIdx.x + 6 * 256] = filterTemp7;
        filterSharedBuffer1[threadIdx.x + 7 * 256] = filterTemp8;
        __syncthreads();
    }
    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 64) * 7 * 1 + 6];

    filterOperand1 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer1[(warpID % 4) * 512 + laneID + 7 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input1filter6 += inputOperand1 * filterOperand6;
    input1filter7 += inputOperand1 * filterOperand7;
    input1filter8 += inputOperand1 * filterOperand8;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input2filter6 += inputOperand2 * filterOperand6;
    input2filter7 += inputOperand2 * filterOperand7;
    input2filter8 += inputOperand2 * filterOperand8;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input3filter6 += inputOperand3 * filterOperand6;
    input3filter7 += inputOperand3 * filterOperand7;
    input3filter8 += inputOperand3 * filterOperand8;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input4filter6 += inputOperand4 * filterOperand6;
    input4filter7 += inputOperand4 * filterOperand7;
    input4filter8 += inputOperand4 * filterOperand8;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;
    input5filter4 += inputOperand5 * filterOperand4;
    input5filter5 += inputOperand5 * filterOperand5;

    input5filter6 += inputOperand5 * filterOperand6;
    input5filter7 += inputOperand5 * filterOperand7;
    input5filter8 += inputOperand5 * filterOperand8;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;
    input6filter4 += inputOperand6 * filterOperand4;
    input6filter5 += inputOperand6 * filterOperand5;

    input6filter6 += inputOperand6 * filterOperand6;
    input6filter7 += inputOperand6 * filterOperand7;
    input6filter8 += inputOperand6 * filterOperand8;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;
    input7filter4 += inputOperand7 * filterOperand4;
    input7filter5 += inputOperand7 * filterOperand5;

    input7filter6 += inputOperand7 * filterOperand6;
    input7filter7 += inputOperand7 * filterOperand7;
    input7filter8 += inputOperand7 * filterOperand8;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (64 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 64);
        input1filter2 += __shfl_down(input1filter2, offset, 64);
        input1filter3 += __shfl_down(input1filter3, offset, 64);
        input1filter4 += __shfl_down(input1filter4, offset, 64);
        input1filter5 += __shfl_down(input1filter5, offset, 64);

        input1filter6 += __shfl_down(input1filter6, offset, 64);
        input1filter7 += __shfl_down(input1filter7, offset, 64);
        input1filter8 += __shfl_down(input1filter8, offset, 64);

        input2filter1 += __shfl_down(input2filter1, offset, 64);
        input2filter2 += __shfl_down(input2filter2, offset, 64);
        input2filter3 += __shfl_down(input2filter3, offset, 64);
        input2filter4 += __shfl_down(input2filter4, offset, 64);
        input2filter5 += __shfl_down(input2filter5, offset, 64);

        input2filter6 += __shfl_down(input2filter6, offset, 64);
        input2filter7 += __shfl_down(input2filter7, offset, 64);
        input2filter8 += __shfl_down(input2filter8, offset, 64);

        input3filter1 += __shfl_down(input3filter1, offset, 64);
        input3filter2 += __shfl_down(input3filter2, offset, 64);
        input3filter3 += __shfl_down(input3filter3, offset, 64);
        input3filter4 += __shfl_down(input3filter4, offset, 64);
        input3filter5 += __shfl_down(input3filter5, offset, 64);

        input3filter6 += __shfl_down(input3filter6, offset, 64);
        input3filter7 += __shfl_down(input3filter7, offset, 64);
        input3filter8 += __shfl_down(input3filter8, offset, 64);

        input4filter1 += __shfl_down(input4filter1, offset, 64);
        input4filter2 += __shfl_down(input4filter2, offset, 64);
        input4filter3 += __shfl_down(input4filter3, offset, 64);
        input4filter4 += __shfl_down(input4filter4, offset, 64);
        input4filter5 += __shfl_down(input4filter5, offset, 64);

        input4filter6 += __shfl_down(input4filter6, offset, 64);
        input4filter7 += __shfl_down(input4filter7, offset, 64);
        input4filter8 += __shfl_down(input4filter8, offset, 64);

        input5filter1 += __shfl_down(input5filter1, offset, 64);
        input5filter2 += __shfl_down(input5filter2, offset, 64);
        input5filter3 += __shfl_down(input5filter3, offset, 64);
        input5filter4 += __shfl_down(input5filter4, offset, 64);
        input5filter5 += __shfl_down(input5filter5, offset, 64);

        input5filter6 += __shfl_down(input5filter6, offset, 64);
        input5filter7 += __shfl_down(input5filter7, offset, 64);
        input5filter8 += __shfl_down(input5filter8, offset, 64);

        input6filter1 += __shfl_down(input6filter1, offset, 64);
        input6filter2 += __shfl_down(input6filter2, offset, 64);
        input6filter3 += __shfl_down(input6filter3, offset, 64);
        input6filter4 += __shfl_down(input6filter4, offset, 64);
        input6filter5 += __shfl_down(input6filter5, offset, 64);

        input6filter6 += __shfl_down(input6filter6, offset, 64);
        input6filter7 += __shfl_down(input6filter7, offset, 64);
        input6filter8 += __shfl_down(input6filter8, offset, 64);

        input7filter1 += __shfl_down(input7filter1, offset, 64);
        input7filter2 += __shfl_down(input7filter2, offset, 64);
        input7filter3 += __shfl_down(input7filter3, offset, 64);
        input7filter4 += __shfl_down(input7filter4, offset, 64);
        input7filter5 += __shfl_down(input7filter5, offset, 64);

        input7filter6 += __shfl_down(input7filter6, offset, 64);
        input7filter7 += __shfl_down(input7filter7, offset, 64);
        input7filter8 += __shfl_down(input7filter8, offset, 64);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 35 * 7840 + (blockIdx.x % 35) / 5 * 7 + (blockIdx.x % 5) / 5 * 7 + (blockIdx.x % 5) * 1568;

    if(laneID % 64 == 0) {
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter2;

        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter3;

        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter4;

        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter5;

        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter6;
        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter6;
        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter6;
        output[blockWriteOutputStartIdx + 5 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter6;

        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter7;
        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter7;
        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter7;
        output[blockWriteOutputStartIdx + 6 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter7;

        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 3] = input4filter8;
        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 4] = input5filter8;
        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 5] = input6filter8;
        output[blockWriteOutputStartIdx + 7 * 1 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 392 + (laneID / 64) * outputHeight * outputWidth + 6] = input7filter8;
    }
}

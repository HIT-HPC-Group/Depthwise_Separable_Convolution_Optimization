/*
Pointwise Convolution Kernel
InputBatch_128_Input_14x14_InChannel_384_OutChannel_96

Grid:
    gridDim.x = (128 * 96 * 14 * 14) / (4 * 14 * 24);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 14
outputChannelPerWarp = 24
channelGroupSize = 8
horizontalRepeat = 2
verticalRepeat = 2

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 14 * 24 output data.
Each warp is responsible for generating 14 * 24 output data.
*/

__global__ void InputBatch_128_Input_14x14_InChannel_384_OutChannel_96(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[2 * 14 * 8];
    __shared__ float inputSharedBuffer2[2 * 14 * 8];

    __shared__ float filterSharedBuffer1[2 * 24 * 8];
    __shared__ float filterSharedBuffer2[2 * 24 * 8];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0, filterTemp2 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0, inputOperand8 = 0, inputOperand9 = 0, inputOperand10 = 0;
    float inputOperand11 = 0, inputOperand12 = 0, inputOperand13 = 0, inputOperand14 = 0;
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

    float input9filter1 = 0, input9filter2 = 0, input9filter3 = 0;

    float input10filter1 = 0, input10filter2 = 0, input10filter3 = 0;

    float input11filter1 = 0, input11filter2 = 0, input11filter3 = 0;

    float input12filter1 = 0, input12filter2 = 0, input12filter3 = 0;

    float input13filter1 = 0, input13filter2 = 0, input13filter3 = 0;

    float input14filter1 = 0, input14filter2 = 0, input14filter3 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 14 * 75264 + (blockIdx.x % 14) / 2 * 28 + (blockIdx.x % 2) / 2 * 14;
    if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 2) * 18432;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 384 + ((threadIdx.x + 0 * 256) % 8)];
    if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 384 + ((threadIdx.x + 1 * 256) % 8)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 8) / (2 * 8); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 8;
        if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
        }

        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 384 + ((threadIdx.x + 0 * 256) % 8)];
        if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
            filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 384 + ((threadIdx.x + 1 * 256) % 8)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 6];
        inputOperand8 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 7];
        inputOperand9 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 8];
        inputOperand10 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 9];

        inputOperand11 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 10];
        inputOperand12 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 11];
        inputOperand13 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 12];
        inputOperand14 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 13];

        filterOperand1 = filterSharedBuffer1[(warpID % 2) * 192 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 2) * 192 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 2) * 192 + laneID + 2 * 64];

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

        input9filter1 += inputOperand9 * filterOperand1;
        input9filter2 += inputOperand9 * filterOperand2;
        input9filter3 += inputOperand9 * filterOperand3;

        input10filter1 += inputOperand10 * filterOperand1;
        input10filter2 += inputOperand10 * filterOperand2;
        input10filter3 += inputOperand10 * filterOperand3;

        input11filter1 += inputOperand11 * filterOperand1;
        input11filter2 += inputOperand11 * filterOperand2;
        input11filter3 += inputOperand11 * filterOperand3;

        input12filter1 += inputOperand12 * filterOperand1;
        input12filter2 += inputOperand12 * filterOperand2;
        input12filter3 += inputOperand12 * filterOperand3;

        input13filter1 += inputOperand13 * filterOperand1;
        input13filter2 += inputOperand13 * filterOperand2;
        input13filter3 += inputOperand13 * filterOperand3;

        input14filter1 += inputOperand14 * filterOperand1;
        input14filter2 += inputOperand14 * filterOperand2;
        input14filter3 += inputOperand14 * filterOperand3;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
            filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 8;
        if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
        }

        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 384 + ((threadIdx.x + 0 * 256) % 8)];
        if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
            filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 384 + ((threadIdx.x + 1 * 256) % 8)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 6];
        inputOperand8 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 7];
        inputOperand9 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 8];
        inputOperand10 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 9];

        inputOperand11 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 10];
        inputOperand12 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 11];
        inputOperand13 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 12];
        inputOperand14 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 13];

        filterOperand1 = filterSharedBuffer2[(warpID % 2) * 192 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 2) * 192 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 2) * 192 + laneID + 2 * 64];

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

        input9filter1 += inputOperand9 * filterOperand1;
        input9filter2 += inputOperand9 * filterOperand2;
        input9filter3 += inputOperand9 * filterOperand3;

        input10filter1 += inputOperand10 * filterOperand1;
        input10filter2 += inputOperand10 * filterOperand2;
        input10filter3 += inputOperand10 * filterOperand3;

        input11filter1 += inputOperand11 * filterOperand1;
        input11filter2 += inputOperand11 * filterOperand2;
        input11filter3 += inputOperand11 * filterOperand3;

        input12filter1 += inputOperand12 * filterOperand1;
        input12filter2 += inputOperand12 * filterOperand2;
        input12filter3 += inputOperand12 * filterOperand3;

        input13filter1 += inputOperand13 * filterOperand1;
        input13filter2 += inputOperand13 * filterOperand2;
        input13filter3 += inputOperand13 * filterOperand3;

        input14filter1 += inputOperand14 * filterOperand1;
        input14filter2 += inputOperand14 * filterOperand2;
        input14filter3 += inputOperand14 * filterOperand3;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
            filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 14 * 14 * 8;
    if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
    }

    blockLoadFilterStartIdx += 8;
    filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 384 + ((threadIdx.x + 0 * 256) % 8)];
    if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 384 + ((threadIdx.x + 1 * 256) % 8)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 6];
    inputOperand8 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 7];
    inputOperand9 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 8];
    inputOperand10 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 9];

    inputOperand11 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 10];
    inputOperand12 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 11];
    inputOperand13 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 12];
    inputOperand14 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 13];

    filterOperand1 = filterSharedBuffer1[(warpID % 2) * 192 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 2) * 192 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 2) * 192 + laneID + 2 * 64];

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

    input9filter1 += inputOperand9 * filterOperand1;
    input9filter2 += inputOperand9 * filterOperand2;
    input9filter3 += inputOperand9 * filterOperand3;

    input10filter1 += inputOperand10 * filterOperand1;
    input10filter2 += inputOperand10 * filterOperand2;
    input10filter3 += inputOperand10 * filterOperand3;

    input11filter1 += inputOperand11 * filterOperand1;
    input11filter2 += inputOperand11 * filterOperand2;
    input11filter3 += inputOperand11 * filterOperand3;

    input12filter1 += inputOperand12 * filterOperand1;
    input12filter2 += inputOperand12 * filterOperand2;
    input12filter3 += inputOperand12 * filterOperand3;

    input13filter1 += inputOperand13 * filterOperand1;
    input13filter2 += inputOperand13 * filterOperand2;
    input13filter3 += inputOperand13 * filterOperand3;

    input14filter1 += inputOperand14 * filterOperand1;
    input14filter2 += inputOperand14 * filterOperand2;
    input14filter3 += inputOperand14 * filterOperand3;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 2 * 14 * 8 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    if(threadIdx.x < 2 * 24 * 8 - 1 * 256) {
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 6];
    inputOperand8 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 7];
    inputOperand9 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 8];
    inputOperand10 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 9];

    inputOperand11 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 10];
    inputOperand12 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 11];
    inputOperand13 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 12];
    inputOperand14 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 8) * 14 * 2 + 13];

    filterOperand1 = filterSharedBuffer2[(warpID % 2) * 192 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer2[(warpID % 2) * 192 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer2[(warpID % 2) * 192 + laneID + 2 * 64];

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

    input9filter1 += inputOperand9 * filterOperand1;
    input9filter2 += inputOperand9 * filterOperand2;
    input9filter3 += inputOperand9 * filterOperand3;

    input10filter1 += inputOperand10 * filterOperand1;
    input10filter2 += inputOperand10 * filterOperand2;
    input10filter3 += inputOperand10 * filterOperand3;

    input11filter1 += inputOperand11 * filterOperand1;
    input11filter2 += inputOperand11 * filterOperand2;
    input11filter3 += inputOperand11 * filterOperand3;

    input12filter1 += inputOperand12 * filterOperand1;
    input12filter2 += inputOperand12 * filterOperand2;
    input12filter3 += inputOperand12 * filterOperand3;

    input13filter1 += inputOperand13 * filterOperand1;
    input13filter2 += inputOperand13 * filterOperand2;
    input13filter3 += inputOperand13 * filterOperand3;

    input14filter1 += inputOperand14 * filterOperand1;
    input14filter2 += inputOperand14 * filterOperand2;
    input14filter3 += inputOperand14 * filterOperand3;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (8 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 8);
        input1filter2 += __shfl_down(input1filter2, offset, 8);
        input1filter3 += __shfl_down(input1filter3, offset, 8);

        input2filter1 += __shfl_down(input2filter1, offset, 8);
        input2filter2 += __shfl_down(input2filter2, offset, 8);
        input2filter3 += __shfl_down(input2filter3, offset, 8);

        input3filter1 += __shfl_down(input3filter1, offset, 8);
        input3filter2 += __shfl_down(input3filter2, offset, 8);
        input3filter3 += __shfl_down(input3filter3, offset, 8);

        input4filter1 += __shfl_down(input4filter1, offset, 8);
        input4filter2 += __shfl_down(input4filter2, offset, 8);
        input4filter3 += __shfl_down(input4filter3, offset, 8);

        input5filter1 += __shfl_down(input5filter1, offset, 8);
        input5filter2 += __shfl_down(input5filter2, offset, 8);
        input5filter3 += __shfl_down(input5filter3, offset, 8);

        input6filter1 += __shfl_down(input6filter1, offset, 8);
        input6filter2 += __shfl_down(input6filter2, offset, 8);
        input6filter3 += __shfl_down(input6filter3, offset, 8);

        input7filter1 += __shfl_down(input7filter1, offset, 8);
        input7filter2 += __shfl_down(input7filter2, offset, 8);
        input7filter3 += __shfl_down(input7filter3, offset, 8);

        input8filter1 += __shfl_down(input8filter1, offset, 8);
        input8filter2 += __shfl_down(input8filter2, offset, 8);
        input8filter3 += __shfl_down(input8filter3, offset, 8);

        input9filter1 += __shfl_down(input9filter1, offset, 8);
        input9filter2 += __shfl_down(input9filter2, offset, 8);
        input9filter3 += __shfl_down(input9filter3, offset, 8);

        input10filter1 += __shfl_down(input10filter1, offset, 8);
        input10filter2 += __shfl_down(input10filter2, offset, 8);
        input10filter3 += __shfl_down(input10filter3, offset, 8);

        input11filter1 += __shfl_down(input11filter1, offset, 8);
        input11filter2 += __shfl_down(input11filter2, offset, 8);
        input11filter3 += __shfl_down(input11filter3, offset, 8);

        input12filter1 += __shfl_down(input12filter1, offset, 8);
        input12filter2 += __shfl_down(input12filter2, offset, 8);
        input12filter3 += __shfl_down(input12filter3, offset, 8);

        input13filter1 += __shfl_down(input13filter1, offset, 8);
        input13filter2 += __shfl_down(input13filter2, offset, 8);
        input13filter3 += __shfl_down(input13filter3, offset, 8);

        input14filter1 += __shfl_down(input14filter1, offset, 8);
        input14filter2 += __shfl_down(input14filter2, offset, 8);
        input14filter3 += __shfl_down(input14filter3, offset, 8);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 14 * 18816 + (blockIdx.x % 14) / 2 * 28 + (blockIdx.x % 2) / 2 * 14 + (blockIdx.x % 2) * 9408;

    if(laneID % 8 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 7] = input8filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 8] = input9filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 9] = input10filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 10] = input11filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 11] = input12filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 12] = input13filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 13] = input14filter1;

        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 7] = input8filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 8] = input9filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 9] = input10filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 10] = input11filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 11] = input12filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 12] = input13filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 13] = input14filter2;

        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 7] = input8filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 8] = input9filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 9] = input10filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 10] = input11filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 11] = input12filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 12] = input13filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 4704 + (laneID / 8) * outputHeight * outputWidth + 13] = input14filter3;
    }
}

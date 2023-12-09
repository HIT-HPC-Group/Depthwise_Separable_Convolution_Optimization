/*
Pointwise Convolution Kernel
InputBatch_32_Input_28x28_InChannel_40_OutChannel_240

Grid:
    gridDim.x = (32 * 240 * 28 * 28) / (4 * 14 * 80);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 14
outputChannelPerWarp = 80
channelGroupSize = 4
horizontalRepeat = 4
verticalRepeat = 1

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 14 * 80 output data.
Each warp is responsible for generating 14 * 80 output data.
*/

__global__ void InputBatch_32_Input_28x28_InChannel_40_OutChannel_240(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[4 * 14 * 4];
    __shared__ float inputSharedBuffer2[4 * 14 * 4];

    __shared__ float filterSharedBuffer1[1 * 80 * 4];
    __shared__ float filterSharedBuffer2[1 * 80 * 4];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0, filterTemp2 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0, inputOperand8 = 0, inputOperand9 = 0, inputOperand10 = 0;
    float inputOperand11 = 0, inputOperand12 = 0, inputOperand13 = 0, inputOperand14 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0, input5filter4 = 0, input5filter5 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0, input6filter4 = 0, input6filter5 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0, input7filter4 = 0, input7filter5 = 0;

    float input8filter1 = 0, input8filter2 = 0, input8filter3 = 0, input8filter4 = 0, input8filter5 = 0;

    float input9filter1 = 0, input9filter2 = 0, input9filter3 = 0, input9filter4 = 0, input9filter5 = 0;

    float input10filter1 = 0, input10filter2 = 0, input10filter3 = 0, input10filter4 = 0, input10filter5 = 0;

    float input11filter1 = 0, input11filter2 = 0, input11filter3 = 0, input11filter4 = 0, input11filter5 = 0;

    float input12filter1 = 0, input12filter2 = 0, input12filter3 = 0, input12filter4 = 0, input12filter5 = 0;

    float input13filter1 = 0, input13filter2 = 0, input13filter3 = 0, input13filter4 = 0, input13filter5 = 0;

    float input14filter1 = 0, input14filter2 = 0, input14filter3 = 0, input14filter4 = 0, input14filter5 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 42 * 31360 + (blockIdx.x % 42) / 6 * 112 + (blockIdx.x % 6) / 3 * 14;
    if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 784 + ((threadIdx.x + 0 * 256) % 56) / 14 * 28 + (threadIdx.x + 0 * 256) % 14];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 3) * 3200;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 40 + ((threadIdx.x + 0 * 256) % 4)];
    if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 40 + ((threadIdx.x + 1 * 256) % 4)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 4) / (2 * 4); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 28 * 28 * 4;
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 784 + ((threadIdx.x + 0 * 256) % 56) / 14 * 28 + (threadIdx.x + 0 * 256) % 14];
        }

        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 40 + ((threadIdx.x + 0 * 256) % 4)];
        if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
            filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 40 + ((threadIdx.x + 1 * 256) % 4)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
        inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
        inputOperand9 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
        inputOperand10 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

        inputOperand11 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
        inputOperand12 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
        inputOperand13 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
        inputOperand14 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 4 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input8filter1 += inputOperand8 * filterOperand1;
        input8filter2 += inputOperand8 * filterOperand2;
        input8filter3 += inputOperand8 * filterOperand3;
        input8filter4 += inputOperand8 * filterOperand4;
        input8filter5 += inputOperand8 * filterOperand5;

        input9filter1 += inputOperand9 * filterOperand1;
        input9filter2 += inputOperand9 * filterOperand2;
        input9filter3 += inputOperand9 * filterOperand3;
        input9filter4 += inputOperand9 * filterOperand4;
        input9filter5 += inputOperand9 * filterOperand5;

        input10filter1 += inputOperand10 * filterOperand1;
        input10filter2 += inputOperand10 * filterOperand2;
        input10filter3 += inputOperand10 * filterOperand3;
        input10filter4 += inputOperand10 * filterOperand4;
        input10filter5 += inputOperand10 * filterOperand5;

        input11filter1 += inputOperand11 * filterOperand1;
        input11filter2 += inputOperand11 * filterOperand2;
        input11filter3 += inputOperand11 * filterOperand3;
        input11filter4 += inputOperand11 * filterOperand4;
        input11filter5 += inputOperand11 * filterOperand5;

        input12filter1 += inputOperand12 * filterOperand1;
        input12filter2 += inputOperand12 * filterOperand2;
        input12filter3 += inputOperand12 * filterOperand3;
        input12filter4 += inputOperand12 * filterOperand4;
        input12filter5 += inputOperand12 * filterOperand5;

        input13filter1 += inputOperand13 * filterOperand1;
        input13filter2 += inputOperand13 * filterOperand2;
        input13filter3 += inputOperand13 * filterOperand3;
        input13filter4 += inputOperand13 * filterOperand4;
        input13filter5 += inputOperand13 * filterOperand5;

        input14filter1 += inputOperand14 * filterOperand1;
        input14filter2 += inputOperand14 * filterOperand2;
        input14filter3 += inputOperand14 * filterOperand3;
        input14filter4 += inputOperand14 * filterOperand4;
        input14filter5 += inputOperand14 * filterOperand5;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
            filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 28 * 28 * 4;
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 784 + ((threadIdx.x + 0 * 256) % 56) / 14 * 28 + (threadIdx.x + 0 * 256) % 14];
        }

        blockLoadFilterStartIdx += 4;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 40 + ((threadIdx.x + 0 * 256) % 4)];
        if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
            filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 40 + ((threadIdx.x + 1 * 256) % 4)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
        inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
        inputOperand9 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
        inputOperand10 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

        inputOperand11 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
        inputOperand12 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
        inputOperand13 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
        inputOperand14 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 4 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;
        input1filter2 += inputOperand1 * filterOperand2;
        input1filter3 += inputOperand1 * filterOperand3;
        input1filter4 += inputOperand1 * filterOperand4;
        input1filter5 += inputOperand1 * filterOperand5;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input8filter1 += inputOperand8 * filterOperand1;
        input8filter2 += inputOperand8 * filterOperand2;
        input8filter3 += inputOperand8 * filterOperand3;
        input8filter4 += inputOperand8 * filterOperand4;
        input8filter5 += inputOperand8 * filterOperand5;

        input9filter1 += inputOperand9 * filterOperand1;
        input9filter2 += inputOperand9 * filterOperand2;
        input9filter3 += inputOperand9 * filterOperand3;
        input9filter4 += inputOperand9 * filterOperand4;
        input9filter5 += inputOperand9 * filterOperand5;

        input10filter1 += inputOperand10 * filterOperand1;
        input10filter2 += inputOperand10 * filterOperand2;
        input10filter3 += inputOperand10 * filterOperand3;
        input10filter4 += inputOperand10 * filterOperand4;
        input10filter5 += inputOperand10 * filterOperand5;

        input11filter1 += inputOperand11 * filterOperand1;
        input11filter2 += inputOperand11 * filterOperand2;
        input11filter3 += inputOperand11 * filterOperand3;
        input11filter4 += inputOperand11 * filterOperand4;
        input11filter5 += inputOperand11 * filterOperand5;

        input12filter1 += inputOperand12 * filterOperand1;
        input12filter2 += inputOperand12 * filterOperand2;
        input12filter3 += inputOperand12 * filterOperand3;
        input12filter4 += inputOperand12 * filterOperand4;
        input12filter5 += inputOperand12 * filterOperand5;

        input13filter1 += inputOperand13 * filterOperand1;
        input13filter2 += inputOperand13 * filterOperand2;
        input13filter3 += inputOperand13 * filterOperand3;
        input13filter4 += inputOperand13 * filterOperand4;
        input13filter5 += inputOperand13 * filterOperand5;

        input14filter1 += inputOperand14 * filterOperand1;
        input14filter2 += inputOperand14 * filterOperand2;
        input14filter3 += inputOperand14 * filterOperand3;
        input14filter4 += inputOperand14 * filterOperand4;
        input14filter5 += inputOperand14 * filterOperand5;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
            filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 28 * 28 * 4;
    if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 56 * 784 + ((threadIdx.x + 0 * 256) % 56) / 14 * 28 + (threadIdx.x + 0 * 256) % 14];
    }

    blockLoadFilterStartIdx += 4;
    filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 40 + ((threadIdx.x + 0 * 256) % 4)];
    if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 4) * 40 + ((threadIdx.x + 1 * 256) % 4)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
    inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
    inputOperand9 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
    inputOperand10 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

    inputOperand11 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
    inputOperand12 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
    inputOperand13 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
    inputOperand14 = inputSharedBuffer1[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 1) * 320 + laneID + 4 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;
    input5filter4 += inputOperand5 * filterOperand4;
    input5filter5 += inputOperand5 * filterOperand5;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;
    input6filter4 += inputOperand6 * filterOperand4;
    input6filter5 += inputOperand6 * filterOperand5;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;
    input7filter4 += inputOperand7 * filterOperand4;
    input7filter5 += inputOperand7 * filterOperand5;

    input8filter1 += inputOperand8 * filterOperand1;
    input8filter2 += inputOperand8 * filterOperand2;
    input8filter3 += inputOperand8 * filterOperand3;
    input8filter4 += inputOperand8 * filterOperand4;
    input8filter5 += inputOperand8 * filterOperand5;

    input9filter1 += inputOperand9 * filterOperand1;
    input9filter2 += inputOperand9 * filterOperand2;
    input9filter3 += inputOperand9 * filterOperand3;
    input9filter4 += inputOperand9 * filterOperand4;
    input9filter5 += inputOperand9 * filterOperand5;

    input10filter1 += inputOperand10 * filterOperand1;
    input10filter2 += inputOperand10 * filterOperand2;
    input10filter3 += inputOperand10 * filterOperand3;
    input10filter4 += inputOperand10 * filterOperand4;
    input10filter5 += inputOperand10 * filterOperand5;

    input11filter1 += inputOperand11 * filterOperand1;
    input11filter2 += inputOperand11 * filterOperand2;
    input11filter3 += inputOperand11 * filterOperand3;
    input11filter4 += inputOperand11 * filterOperand4;
    input11filter5 += inputOperand11 * filterOperand5;

    input12filter1 += inputOperand12 * filterOperand1;
    input12filter2 += inputOperand12 * filterOperand2;
    input12filter3 += inputOperand12 * filterOperand3;
    input12filter4 += inputOperand12 * filterOperand4;
    input12filter5 += inputOperand12 * filterOperand5;

    input13filter1 += inputOperand13 * filterOperand1;
    input13filter2 += inputOperand13 * filterOperand2;
    input13filter3 += inputOperand13 * filterOperand3;
    input13filter4 += inputOperand13 * filterOperand4;
    input13filter5 += inputOperand13 * filterOperand5;

    input14filter1 += inputOperand14 * filterOperand1;
    input14filter2 += inputOperand14 * filterOperand2;
    input14filter3 += inputOperand14 * filterOperand3;
    input14filter4 += inputOperand14 * filterOperand4;
    input14filter5 += inputOperand14 * filterOperand5;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 4 * 14 * 4 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    if(threadIdx.x < 1 * 80 * 4 - 1 * 256) {
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 6];
    inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 7];
    inputOperand9 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 8];
    inputOperand10 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 9];

    inputOperand11 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 10];
    inputOperand12 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 11];
    inputOperand13 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 12];
    inputOperand14 = inputSharedBuffer2[((warpID / 1) % 4) * 14 + (laneID % 4) * 14 * 4 + 13];

    filterOperand1 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer2[(warpID % 1) * 320 + laneID + 4 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;
    input1filter2 += inputOperand1 * filterOperand2;
    input1filter3 += inputOperand1 * filterOperand3;
    input1filter4 += inputOperand1 * filterOperand4;
    input1filter5 += inputOperand1 * filterOperand5;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;
    input5filter4 += inputOperand5 * filterOperand4;
    input5filter5 += inputOperand5 * filterOperand5;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;
    input6filter4 += inputOperand6 * filterOperand4;
    input6filter5 += inputOperand6 * filterOperand5;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;
    input7filter4 += inputOperand7 * filterOperand4;
    input7filter5 += inputOperand7 * filterOperand5;

    input8filter1 += inputOperand8 * filterOperand1;
    input8filter2 += inputOperand8 * filterOperand2;
    input8filter3 += inputOperand8 * filterOperand3;
    input8filter4 += inputOperand8 * filterOperand4;
    input8filter5 += inputOperand8 * filterOperand5;

    input9filter1 += inputOperand9 * filterOperand1;
    input9filter2 += inputOperand9 * filterOperand2;
    input9filter3 += inputOperand9 * filterOperand3;
    input9filter4 += inputOperand9 * filterOperand4;
    input9filter5 += inputOperand9 * filterOperand5;

    input10filter1 += inputOperand10 * filterOperand1;
    input10filter2 += inputOperand10 * filterOperand2;
    input10filter3 += inputOperand10 * filterOperand3;
    input10filter4 += inputOperand10 * filterOperand4;
    input10filter5 += inputOperand10 * filterOperand5;

    input11filter1 += inputOperand11 * filterOperand1;
    input11filter2 += inputOperand11 * filterOperand2;
    input11filter3 += inputOperand11 * filterOperand3;
    input11filter4 += inputOperand11 * filterOperand4;
    input11filter5 += inputOperand11 * filterOperand5;

    input12filter1 += inputOperand12 * filterOperand1;
    input12filter2 += inputOperand12 * filterOperand2;
    input12filter3 += inputOperand12 * filterOperand3;
    input12filter4 += inputOperand12 * filterOperand4;
    input12filter5 += inputOperand12 * filterOperand5;

    input13filter1 += inputOperand13 * filterOperand1;
    input13filter2 += inputOperand13 * filterOperand2;
    input13filter3 += inputOperand13 * filterOperand3;
    input13filter4 += inputOperand13 * filterOperand4;
    input13filter5 += inputOperand13 * filterOperand5;

    input14filter1 += inputOperand14 * filterOperand1;
    input14filter2 += inputOperand14 * filterOperand2;
    input14filter3 += inputOperand14 * filterOperand3;
    input14filter4 += inputOperand14 * filterOperand4;
    input14filter5 += inputOperand14 * filterOperand5;

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

        input2filter1 += __shfl_down(input2filter1, offset, 4);
        input2filter2 += __shfl_down(input2filter2, offset, 4);
        input2filter3 += __shfl_down(input2filter3, offset, 4);
        input2filter4 += __shfl_down(input2filter4, offset, 4);
        input2filter5 += __shfl_down(input2filter5, offset, 4);

        input3filter1 += __shfl_down(input3filter1, offset, 4);
        input3filter2 += __shfl_down(input3filter2, offset, 4);
        input3filter3 += __shfl_down(input3filter3, offset, 4);
        input3filter4 += __shfl_down(input3filter4, offset, 4);
        input3filter5 += __shfl_down(input3filter5, offset, 4);

        input4filter1 += __shfl_down(input4filter1, offset, 4);
        input4filter2 += __shfl_down(input4filter2, offset, 4);
        input4filter3 += __shfl_down(input4filter3, offset, 4);
        input4filter4 += __shfl_down(input4filter4, offset, 4);
        input4filter5 += __shfl_down(input4filter5, offset, 4);

        input5filter1 += __shfl_down(input5filter1, offset, 4);
        input5filter2 += __shfl_down(input5filter2, offset, 4);
        input5filter3 += __shfl_down(input5filter3, offset, 4);
        input5filter4 += __shfl_down(input5filter4, offset, 4);
        input5filter5 += __shfl_down(input5filter5, offset, 4);

        input6filter1 += __shfl_down(input6filter1, offset, 4);
        input6filter2 += __shfl_down(input6filter2, offset, 4);
        input6filter3 += __shfl_down(input6filter3, offset, 4);
        input6filter4 += __shfl_down(input6filter4, offset, 4);
        input6filter5 += __shfl_down(input6filter5, offset, 4);

        input7filter1 += __shfl_down(input7filter1, offset, 4);
        input7filter2 += __shfl_down(input7filter2, offset, 4);
        input7filter3 += __shfl_down(input7filter3, offset, 4);
        input7filter4 += __shfl_down(input7filter4, offset, 4);
        input7filter5 += __shfl_down(input7filter5, offset, 4);

        input8filter1 += __shfl_down(input8filter1, offset, 4);
        input8filter2 += __shfl_down(input8filter2, offset, 4);
        input8filter3 += __shfl_down(input8filter3, offset, 4);
        input8filter4 += __shfl_down(input8filter4, offset, 4);
        input8filter5 += __shfl_down(input8filter5, offset, 4);

        input9filter1 += __shfl_down(input9filter1, offset, 4);
        input9filter2 += __shfl_down(input9filter2, offset, 4);
        input9filter3 += __shfl_down(input9filter3, offset, 4);
        input9filter4 += __shfl_down(input9filter4, offset, 4);
        input9filter5 += __shfl_down(input9filter5, offset, 4);

        input10filter1 += __shfl_down(input10filter1, offset, 4);
        input10filter2 += __shfl_down(input10filter2, offset, 4);
        input10filter3 += __shfl_down(input10filter3, offset, 4);
        input10filter4 += __shfl_down(input10filter4, offset, 4);
        input10filter5 += __shfl_down(input10filter5, offset, 4);

        input11filter1 += __shfl_down(input11filter1, offset, 4);
        input11filter2 += __shfl_down(input11filter2, offset, 4);
        input11filter3 += __shfl_down(input11filter3, offset, 4);
        input11filter4 += __shfl_down(input11filter4, offset, 4);
        input11filter5 += __shfl_down(input11filter5, offset, 4);

        input12filter1 += __shfl_down(input12filter1, offset, 4);
        input12filter2 += __shfl_down(input12filter2, offset, 4);
        input12filter3 += __shfl_down(input12filter3, offset, 4);
        input12filter4 += __shfl_down(input12filter4, offset, 4);
        input12filter5 += __shfl_down(input12filter5, offset, 4);

        input13filter1 += __shfl_down(input13filter1, offset, 4);
        input13filter2 += __shfl_down(input13filter2, offset, 4);
        input13filter3 += __shfl_down(input13filter3, offset, 4);
        input13filter4 += __shfl_down(input13filter4, offset, 4);
        input13filter5 += __shfl_down(input13filter5, offset, 4);

        input14filter1 += __shfl_down(input14filter1, offset, 4);
        input14filter2 += __shfl_down(input14filter2, offset, 4);
        input14filter3 += __shfl_down(input14filter3, offset, 4);
        input14filter4 += __shfl_down(input14filter4, offset, 4);
        input14filter5 += __shfl_down(input14filter5, offset, 4);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 42 * 188160 + (blockIdx.x % 42) / 6 * 112 + (blockIdx.x % 6) / 3 * 14 + (blockIdx.x % 3) * 62720;

    if(laneID % 4 == 0) {
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter1;

        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter2;
        output[blockWriteOutputStartIdx + 1 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter2;

        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter3;
        output[blockWriteOutputStartIdx + 2 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter3;

        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter4;
        output[blockWriteOutputStartIdx + 3 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter4;

        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter5;
        output[blockWriteOutputStartIdx + 4 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 14 + (warpID % 1) * 62720 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter5;
    }
}

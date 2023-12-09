/*
Pointwise Convolution Kernel
InputBatch_8_Input_14x14_InChannel_576_OutChannel_96

Grid:
    gridDim.x = (8 * 96 * 14 * 14) / (4 * 14 * 16);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 14
outputChannelPerWarp = 16
channelGroupSize = 32
horizontalRepeat = 2
verticalRepeat = 2

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 14 * 16 output data.
Each warp is responsible for generating 14 * 16 output data.
*/

__global__ void InputBatch_8_Input_14x14_InChannel_576_OutChannel_96(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[2 * 14 * 32];
    __shared__ float inputSharedBuffer2[2 * 14 * 32];

    __shared__ float filterSharedBuffer1[2 * 16 * 32];
    __shared__ float filterSharedBuffer2[2 * 16 * 32];

    // to hold loaded operands temp.
    float inputTemp1 = 0, inputTemp2 = 0, inputTemp3 = 0, inputTemp4 = 0;
    float filterTemp1 = 0, filterTemp2 = 0, filterTemp3 = 0, filterTemp4 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0, inputOperand8 = 0, inputOperand9 = 0, inputOperand10 = 0;
    float inputOperand11 = 0, inputOperand12 = 0, inputOperand13 = 0, inputOperand14 = 0;
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

    float input8filter1 = 0, input8filter2 = 0, input8filter3 = 0, input8filter4 = 0, input8filter5 = 0;
    float input8filter6 = 0, input8filter7 = 0, input8filter8 = 0;

    float input9filter1 = 0, input9filter2 = 0, input9filter3 = 0, input9filter4 = 0, input9filter5 = 0;
    float input9filter6 = 0, input9filter7 = 0, input9filter8 = 0;

    float input10filter1 = 0, input10filter2 = 0, input10filter3 = 0, input10filter4 = 0, input10filter5 = 0;
    float input10filter6 = 0, input10filter7 = 0, input10filter8 = 0;

    float input11filter1 = 0, input11filter2 = 0, input11filter3 = 0, input11filter4 = 0, input11filter5 = 0;
    float input11filter6 = 0, input11filter7 = 0, input11filter8 = 0;

    float input12filter1 = 0, input12filter2 = 0, input12filter3 = 0, input12filter4 = 0, input12filter5 = 0;
    float input12filter6 = 0, input12filter7 = 0, input12filter8 = 0;

    float input13filter1 = 0, input13filter2 = 0, input13filter3 = 0, input13filter4 = 0, input13filter5 = 0;
    float input13filter6 = 0, input13filter7 = 0, input13filter8 = 0;

    float input14filter1 = 0, input14filter2 = 0, input14filter3 = 0, input14filter4 = 0, input14filter5 = 0;
    float input14filter6 = 0, input14filter7 = 0, input14filter8 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 21 * 112896 + (blockIdx.x % 21) / 3 * 28 + (blockIdx.x % 3) / 3 * 14;
    inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
    inputSharedBuffer1[threadIdx.x + 1 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 28 * 196 + ((threadIdx.x + 1 * 256) % 28) / 14 * 14 + (threadIdx.x + 1 * 256) % 14];
    inputSharedBuffer1[threadIdx.x + 2 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 2 * 256) / 28 * 196 + ((threadIdx.x + 2 * 256) % 28) / 14 * 14 + (threadIdx.x + 2 * 256) % 14];
    if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
        inputSharedBuffer1[threadIdx.x + 3 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 3 * 256) / 28 * 196 + ((threadIdx.x + 3 * 256) % 28) / 14 * 14 + (threadIdx.x + 3 * 256) % 14];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 3) * 18432;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 576 + ((threadIdx.x + 0 * 256) % 32)];
    filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 32) * 576 + ((threadIdx.x + 1 * 256) % 32)];
    filterSharedBuffer1[threadIdx.x + 2 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 32) * 576 + ((threadIdx.x + 2 * 256) % 32)];
    filterSharedBuffer1[threadIdx.x + 3 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 32) * 576 + ((threadIdx.x + 3 * 256) % 32)];
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 32) / (2 * 32); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 32;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
        inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 28 * 196 + ((threadIdx.x + 1 * 256) % 28) / 14 * 14 + (threadIdx.x + 1 * 256) % 14];
        inputTemp3 = input[blockLoadInputStartIdx + (threadIdx.x + 2 * 256) / 28 * 196 + ((threadIdx.x + 2 * 256) % 28) / 14 * 14 + (threadIdx.x + 2 * 256) % 14];
        if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
            inputTemp4 = input[blockLoadInputStartIdx + (threadIdx.x + 3 * 256) / 28 * 196 + ((threadIdx.x + 3 * 256) % 28) / 14 * 14 + (threadIdx.x + 3 * 256) % 14];
        }

        blockLoadFilterStartIdx += 32;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 576 + ((threadIdx.x + 0 * 256) % 32)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 32) * 576 + ((threadIdx.x + 1 * 256) % 32)];
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 32) * 576 + ((threadIdx.x + 2 * 256) % 32)];
        filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 32) * 576 + ((threadIdx.x + 3 * 256) % 32)];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 6];
        inputOperand8 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 7];
        inputOperand9 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 8];
        inputOperand10 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 9];

        inputOperand11 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 10];
        inputOperand12 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 11];
        inputOperand13 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 12];
        inputOperand14 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 13];

        filterOperand1 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 7 * 64];

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

        input8filter1 += inputOperand8 * filterOperand1;
        input8filter2 += inputOperand8 * filterOperand2;
        input8filter3 += inputOperand8 * filterOperand3;
        input8filter4 += inputOperand8 * filterOperand4;
        input8filter5 += inputOperand8 * filterOperand5;

        input8filter6 += inputOperand8 * filterOperand6;
        input8filter7 += inputOperand8 * filterOperand7;
        input8filter8 += inputOperand8 * filterOperand8;

        input9filter1 += inputOperand9 * filterOperand1;
        input9filter2 += inputOperand9 * filterOperand2;
        input9filter3 += inputOperand9 * filterOperand3;
        input9filter4 += inputOperand9 * filterOperand4;
        input9filter5 += inputOperand9 * filterOperand5;

        input9filter6 += inputOperand9 * filterOperand6;
        input9filter7 += inputOperand9 * filterOperand7;
        input9filter8 += inputOperand9 * filterOperand8;

        input10filter1 += inputOperand10 * filterOperand1;
        input10filter2 += inputOperand10 * filterOperand2;
        input10filter3 += inputOperand10 * filterOperand3;
        input10filter4 += inputOperand10 * filterOperand4;
        input10filter5 += inputOperand10 * filterOperand5;

        input10filter6 += inputOperand10 * filterOperand6;
        input10filter7 += inputOperand10 * filterOperand7;
        input10filter8 += inputOperand10 * filterOperand8;

        input11filter1 += inputOperand11 * filterOperand1;
        input11filter2 += inputOperand11 * filterOperand2;
        input11filter3 += inputOperand11 * filterOperand3;
        input11filter4 += inputOperand11 * filterOperand4;
        input11filter5 += inputOperand11 * filterOperand5;

        input11filter6 += inputOperand11 * filterOperand6;
        input11filter7 += inputOperand11 * filterOperand7;
        input11filter8 += inputOperand11 * filterOperand8;

        input12filter1 += inputOperand12 * filterOperand1;
        input12filter2 += inputOperand12 * filterOperand2;
        input12filter3 += inputOperand12 * filterOperand3;
        input12filter4 += inputOperand12 * filterOperand4;
        input12filter5 += inputOperand12 * filterOperand5;

        input12filter6 += inputOperand12 * filterOperand6;
        input12filter7 += inputOperand12 * filterOperand7;
        input12filter8 += inputOperand12 * filterOperand8;

        input13filter1 += inputOperand13 * filterOperand1;
        input13filter2 += inputOperand13 * filterOperand2;
        input13filter3 += inputOperand13 * filterOperand3;
        input13filter4 += inputOperand13 * filterOperand4;
        input13filter5 += inputOperand13 * filterOperand5;

        input13filter6 += inputOperand13 * filterOperand6;
        input13filter7 += inputOperand13 * filterOperand7;
        input13filter8 += inputOperand13 * filterOperand8;

        input14filter1 += inputOperand14 * filterOperand1;
        input14filter2 += inputOperand14 * filterOperand2;
        input14filter3 += inputOperand14 * filterOperand3;
        input14filter4 += inputOperand14 * filterOperand4;
        input14filter5 += inputOperand14 * filterOperand5;

        input14filter6 += inputOperand14 * filterOperand6;
        input14filter7 += inputOperand14 * filterOperand7;
        input14filter8 += inputOperand14 * filterOperand8;

        // Copy Temp Registers to shared buffer 2
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
        inputSharedBuffer2[threadIdx.x + 2 * 256] = inputTemp3;
        if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
            inputSharedBuffer2[threadIdx.x + 3 * 256] = inputTemp4;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
        filterSharedBuffer2[threadIdx.x + 3 * 256] = filterTemp4;
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 14 * 14 * 32;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
        inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 28 * 196 + ((threadIdx.x + 1 * 256) % 28) / 14 * 14 + (threadIdx.x + 1 * 256) % 14];
        inputTemp3 = input[blockLoadInputStartIdx + (threadIdx.x + 2 * 256) / 28 * 196 + ((threadIdx.x + 2 * 256) % 28) / 14 * 14 + (threadIdx.x + 2 * 256) % 14];
        if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
            inputTemp4 = input[blockLoadInputStartIdx + (threadIdx.x + 3 * 256) / 28 * 196 + ((threadIdx.x + 3 * 256) % 28) / 14 * 14 + (threadIdx.x + 3 * 256) % 14];
        }

        blockLoadFilterStartIdx += 32;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 576 + ((threadIdx.x + 0 * 256) % 32)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 32) * 576 + ((threadIdx.x + 1 * 256) % 32)];
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 32) * 576 + ((threadIdx.x + 2 * 256) % 32)];
        filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 32) * 576 + ((threadIdx.x + 3 * 256) % 32)];

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 6];
        inputOperand8 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 7];
        inputOperand9 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 8];
        inputOperand10 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 9];

        inputOperand11 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 10];
        inputOperand12 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 11];
        inputOperand13 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 12];
        inputOperand14 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 13];

        filterOperand1 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 7 * 64];

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

        input8filter1 += inputOperand8 * filterOperand1;
        input8filter2 += inputOperand8 * filterOperand2;
        input8filter3 += inputOperand8 * filterOperand3;
        input8filter4 += inputOperand8 * filterOperand4;
        input8filter5 += inputOperand8 * filterOperand5;

        input8filter6 += inputOperand8 * filterOperand6;
        input8filter7 += inputOperand8 * filterOperand7;
        input8filter8 += inputOperand8 * filterOperand8;

        input9filter1 += inputOperand9 * filterOperand1;
        input9filter2 += inputOperand9 * filterOperand2;
        input9filter3 += inputOperand9 * filterOperand3;
        input9filter4 += inputOperand9 * filterOperand4;
        input9filter5 += inputOperand9 * filterOperand5;

        input9filter6 += inputOperand9 * filterOperand6;
        input9filter7 += inputOperand9 * filterOperand7;
        input9filter8 += inputOperand9 * filterOperand8;

        input10filter1 += inputOperand10 * filterOperand1;
        input10filter2 += inputOperand10 * filterOperand2;
        input10filter3 += inputOperand10 * filterOperand3;
        input10filter4 += inputOperand10 * filterOperand4;
        input10filter5 += inputOperand10 * filterOperand5;

        input10filter6 += inputOperand10 * filterOperand6;
        input10filter7 += inputOperand10 * filterOperand7;
        input10filter8 += inputOperand10 * filterOperand8;

        input11filter1 += inputOperand11 * filterOperand1;
        input11filter2 += inputOperand11 * filterOperand2;
        input11filter3 += inputOperand11 * filterOperand3;
        input11filter4 += inputOperand11 * filterOperand4;
        input11filter5 += inputOperand11 * filterOperand5;

        input11filter6 += inputOperand11 * filterOperand6;
        input11filter7 += inputOperand11 * filterOperand7;
        input11filter8 += inputOperand11 * filterOperand8;

        input12filter1 += inputOperand12 * filterOperand1;
        input12filter2 += inputOperand12 * filterOperand2;
        input12filter3 += inputOperand12 * filterOperand3;
        input12filter4 += inputOperand12 * filterOperand4;
        input12filter5 += inputOperand12 * filterOperand5;

        input12filter6 += inputOperand12 * filterOperand6;
        input12filter7 += inputOperand12 * filterOperand7;
        input12filter8 += inputOperand12 * filterOperand8;

        input13filter1 += inputOperand13 * filterOperand1;
        input13filter2 += inputOperand13 * filterOperand2;
        input13filter3 += inputOperand13 * filterOperand3;
        input13filter4 += inputOperand13 * filterOperand4;
        input13filter5 += inputOperand13 * filterOperand5;

        input13filter6 += inputOperand13 * filterOperand6;
        input13filter7 += inputOperand13 * filterOperand7;
        input13filter8 += inputOperand13 * filterOperand8;

        input14filter1 += inputOperand14 * filterOperand1;
        input14filter2 += inputOperand14 * filterOperand2;
        input14filter3 += inputOperand14 * filterOperand3;
        input14filter4 += inputOperand14 * filterOperand4;
        input14filter5 += inputOperand14 * filterOperand5;

        input14filter6 += inputOperand14 * filterOperand6;
        input14filter7 += inputOperand14 * filterOperand7;
        input14filter8 += inputOperand14 * filterOperand8;

        // Copy Temp Registers to shared buffer 1
        inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        inputSharedBuffer1[threadIdx.x + 1 * 256] = inputTemp2;
        inputSharedBuffer1[threadIdx.x + 2 * 256] = inputTemp3;
        if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
            inputSharedBuffer1[threadIdx.x + 3 * 256] = inputTemp4;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        filterSharedBuffer1[threadIdx.x + 2 * 256] = filterTemp3;
        filterSharedBuffer1[threadIdx.x + 3 * 256] = filterTemp4;
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 14 * 14 * 32;
    inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 28 * 196 + ((threadIdx.x + 0 * 256) % 28) / 14 * 14 + (threadIdx.x + 0 * 256) % 14];
    inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 28 * 196 + ((threadIdx.x + 1 * 256) % 28) / 14 * 14 + (threadIdx.x + 1 * 256) % 14];
    inputTemp3 = input[blockLoadInputStartIdx + (threadIdx.x + 2 * 256) / 28 * 196 + ((threadIdx.x + 2 * 256) % 28) / 14 * 14 + (threadIdx.x + 2 * 256) % 14];
    if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
        inputTemp4 = input[blockLoadInputStartIdx + (threadIdx.x + 3 * 256) / 28 * 196 + ((threadIdx.x + 3 * 256) % 28) / 14 * 14 + (threadIdx.x + 3 * 256) % 14];
    }

    blockLoadFilterStartIdx += 32;
    filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 32) * 576 + ((threadIdx.x + 0 * 256) % 32)];
    filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 32) * 576 + ((threadIdx.x + 1 * 256) % 32)];
    filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 32) * 576 + ((threadIdx.x + 2 * 256) % 32)];
    filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 32) * 576 + ((threadIdx.x + 3 * 256) % 32)];

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 6];
    inputOperand8 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 7];
    inputOperand9 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 8];
    inputOperand10 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 9];

    inputOperand11 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 10];
    inputOperand12 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 11];
    inputOperand13 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 12];
    inputOperand14 = inputSharedBuffer1[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 13];

    filterOperand1 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer1[(warpID % 2) * 512 + laneID + 7 * 64];

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

    input8filter1 += inputOperand8 * filterOperand1;
    input8filter2 += inputOperand8 * filterOperand2;
    input8filter3 += inputOperand8 * filterOperand3;
    input8filter4 += inputOperand8 * filterOperand4;
    input8filter5 += inputOperand8 * filterOperand5;

    input8filter6 += inputOperand8 * filterOperand6;
    input8filter7 += inputOperand8 * filterOperand7;
    input8filter8 += inputOperand8 * filterOperand8;

    input9filter1 += inputOperand9 * filterOperand1;
    input9filter2 += inputOperand9 * filterOperand2;
    input9filter3 += inputOperand9 * filterOperand3;
    input9filter4 += inputOperand9 * filterOperand4;
    input9filter5 += inputOperand9 * filterOperand5;

    input9filter6 += inputOperand9 * filterOperand6;
    input9filter7 += inputOperand9 * filterOperand7;
    input9filter8 += inputOperand9 * filterOperand8;

    input10filter1 += inputOperand10 * filterOperand1;
    input10filter2 += inputOperand10 * filterOperand2;
    input10filter3 += inputOperand10 * filterOperand3;
    input10filter4 += inputOperand10 * filterOperand4;
    input10filter5 += inputOperand10 * filterOperand5;

    input10filter6 += inputOperand10 * filterOperand6;
    input10filter7 += inputOperand10 * filterOperand7;
    input10filter8 += inputOperand10 * filterOperand8;

    input11filter1 += inputOperand11 * filterOperand1;
    input11filter2 += inputOperand11 * filterOperand2;
    input11filter3 += inputOperand11 * filterOperand3;
    input11filter4 += inputOperand11 * filterOperand4;
    input11filter5 += inputOperand11 * filterOperand5;

    input11filter6 += inputOperand11 * filterOperand6;
    input11filter7 += inputOperand11 * filterOperand7;
    input11filter8 += inputOperand11 * filterOperand8;

    input12filter1 += inputOperand12 * filterOperand1;
    input12filter2 += inputOperand12 * filterOperand2;
    input12filter3 += inputOperand12 * filterOperand3;
    input12filter4 += inputOperand12 * filterOperand4;
    input12filter5 += inputOperand12 * filterOperand5;

    input12filter6 += inputOperand12 * filterOperand6;
    input12filter7 += inputOperand12 * filterOperand7;
    input12filter8 += inputOperand12 * filterOperand8;

    input13filter1 += inputOperand13 * filterOperand1;
    input13filter2 += inputOperand13 * filterOperand2;
    input13filter3 += inputOperand13 * filterOperand3;
    input13filter4 += inputOperand13 * filterOperand4;
    input13filter5 += inputOperand13 * filterOperand5;

    input13filter6 += inputOperand13 * filterOperand6;
    input13filter7 += inputOperand13 * filterOperand7;
    input13filter8 += inputOperand13 * filterOperand8;

    input14filter1 += inputOperand14 * filterOperand1;
    input14filter2 += inputOperand14 * filterOperand2;
    input14filter3 += inputOperand14 * filterOperand3;
    input14filter4 += inputOperand14 * filterOperand4;
    input14filter5 += inputOperand14 * filterOperand5;

    input14filter6 += inputOperand14 * filterOperand6;
    input14filter7 += inputOperand14 * filterOperand7;
    input14filter8 += inputOperand14 * filterOperand8;

    // Copy Temp Registers to shared buffer 2
    inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
    inputSharedBuffer2[threadIdx.x + 2 * 256] = inputTemp3;
    if(threadIdx.x < 2 * 14 * 32 - 3 * 256) {
        inputSharedBuffer2[threadIdx.x + 3 * 256] = inputTemp4;
    }

    filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
    filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
    filterSharedBuffer2[threadIdx.x + 3 * 256] = filterTemp4;
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 6];
    inputOperand8 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 7];
    inputOperand9 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 8];
    inputOperand10 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 9];

    inputOperand11 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 10];
    inputOperand12 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 11];
    inputOperand13 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 12];
    inputOperand14 = inputSharedBuffer2[((warpID / 2) % 2) * 14 + (laneID % 32) * 14 * 2 + 13];

    filterOperand1 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer2[(warpID % 2) * 512 + laneID + 7 * 64];

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

    input8filter1 += inputOperand8 * filterOperand1;
    input8filter2 += inputOperand8 * filterOperand2;
    input8filter3 += inputOperand8 * filterOperand3;
    input8filter4 += inputOperand8 * filterOperand4;
    input8filter5 += inputOperand8 * filterOperand5;

    input8filter6 += inputOperand8 * filterOperand6;
    input8filter7 += inputOperand8 * filterOperand7;
    input8filter8 += inputOperand8 * filterOperand8;

    input9filter1 += inputOperand9 * filterOperand1;
    input9filter2 += inputOperand9 * filterOperand2;
    input9filter3 += inputOperand9 * filterOperand3;
    input9filter4 += inputOperand9 * filterOperand4;
    input9filter5 += inputOperand9 * filterOperand5;

    input9filter6 += inputOperand9 * filterOperand6;
    input9filter7 += inputOperand9 * filterOperand7;
    input9filter8 += inputOperand9 * filterOperand8;

    input10filter1 += inputOperand10 * filterOperand1;
    input10filter2 += inputOperand10 * filterOperand2;
    input10filter3 += inputOperand10 * filterOperand3;
    input10filter4 += inputOperand10 * filterOperand4;
    input10filter5 += inputOperand10 * filterOperand5;

    input10filter6 += inputOperand10 * filterOperand6;
    input10filter7 += inputOperand10 * filterOperand7;
    input10filter8 += inputOperand10 * filterOperand8;

    input11filter1 += inputOperand11 * filterOperand1;
    input11filter2 += inputOperand11 * filterOperand2;
    input11filter3 += inputOperand11 * filterOperand3;
    input11filter4 += inputOperand11 * filterOperand4;
    input11filter5 += inputOperand11 * filterOperand5;

    input11filter6 += inputOperand11 * filterOperand6;
    input11filter7 += inputOperand11 * filterOperand7;
    input11filter8 += inputOperand11 * filterOperand8;

    input12filter1 += inputOperand12 * filterOperand1;
    input12filter2 += inputOperand12 * filterOperand2;
    input12filter3 += inputOperand12 * filterOperand3;
    input12filter4 += inputOperand12 * filterOperand4;
    input12filter5 += inputOperand12 * filterOperand5;

    input12filter6 += inputOperand12 * filterOperand6;
    input12filter7 += inputOperand12 * filterOperand7;
    input12filter8 += inputOperand12 * filterOperand8;

    input13filter1 += inputOperand13 * filterOperand1;
    input13filter2 += inputOperand13 * filterOperand2;
    input13filter3 += inputOperand13 * filterOperand3;
    input13filter4 += inputOperand13 * filterOperand4;
    input13filter5 += inputOperand13 * filterOperand5;

    input13filter6 += inputOperand13 * filterOperand6;
    input13filter7 += inputOperand13 * filterOperand7;
    input13filter8 += inputOperand13 * filterOperand8;

    input14filter1 += inputOperand14 * filterOperand1;
    input14filter2 += inputOperand14 * filterOperand2;
    input14filter3 += inputOperand14 * filterOperand3;
    input14filter4 += inputOperand14 * filterOperand4;
    input14filter5 += inputOperand14 * filterOperand5;

    input14filter6 += inputOperand14 * filterOperand6;
    input14filter7 += inputOperand14 * filterOperand7;
    input14filter8 += inputOperand14 * filterOperand8;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 32);
        input1filter2 += __shfl_down(input1filter2, offset, 32);
        input1filter3 += __shfl_down(input1filter3, offset, 32);
        input1filter4 += __shfl_down(input1filter4, offset, 32);
        input1filter5 += __shfl_down(input1filter5, offset, 32);

        input1filter6 += __shfl_down(input1filter6, offset, 32);
        input1filter7 += __shfl_down(input1filter7, offset, 32);
        input1filter8 += __shfl_down(input1filter8, offset, 32);

        input2filter1 += __shfl_down(input2filter1, offset, 32);
        input2filter2 += __shfl_down(input2filter2, offset, 32);
        input2filter3 += __shfl_down(input2filter3, offset, 32);
        input2filter4 += __shfl_down(input2filter4, offset, 32);
        input2filter5 += __shfl_down(input2filter5, offset, 32);

        input2filter6 += __shfl_down(input2filter6, offset, 32);
        input2filter7 += __shfl_down(input2filter7, offset, 32);
        input2filter8 += __shfl_down(input2filter8, offset, 32);

        input3filter1 += __shfl_down(input3filter1, offset, 32);
        input3filter2 += __shfl_down(input3filter2, offset, 32);
        input3filter3 += __shfl_down(input3filter3, offset, 32);
        input3filter4 += __shfl_down(input3filter4, offset, 32);
        input3filter5 += __shfl_down(input3filter5, offset, 32);

        input3filter6 += __shfl_down(input3filter6, offset, 32);
        input3filter7 += __shfl_down(input3filter7, offset, 32);
        input3filter8 += __shfl_down(input3filter8, offset, 32);

        input4filter1 += __shfl_down(input4filter1, offset, 32);
        input4filter2 += __shfl_down(input4filter2, offset, 32);
        input4filter3 += __shfl_down(input4filter3, offset, 32);
        input4filter4 += __shfl_down(input4filter4, offset, 32);
        input4filter5 += __shfl_down(input4filter5, offset, 32);

        input4filter6 += __shfl_down(input4filter6, offset, 32);
        input4filter7 += __shfl_down(input4filter7, offset, 32);
        input4filter8 += __shfl_down(input4filter8, offset, 32);

        input5filter1 += __shfl_down(input5filter1, offset, 32);
        input5filter2 += __shfl_down(input5filter2, offset, 32);
        input5filter3 += __shfl_down(input5filter3, offset, 32);
        input5filter4 += __shfl_down(input5filter4, offset, 32);
        input5filter5 += __shfl_down(input5filter5, offset, 32);

        input5filter6 += __shfl_down(input5filter6, offset, 32);
        input5filter7 += __shfl_down(input5filter7, offset, 32);
        input5filter8 += __shfl_down(input5filter8, offset, 32);

        input6filter1 += __shfl_down(input6filter1, offset, 32);
        input6filter2 += __shfl_down(input6filter2, offset, 32);
        input6filter3 += __shfl_down(input6filter3, offset, 32);
        input6filter4 += __shfl_down(input6filter4, offset, 32);
        input6filter5 += __shfl_down(input6filter5, offset, 32);

        input6filter6 += __shfl_down(input6filter6, offset, 32);
        input6filter7 += __shfl_down(input6filter7, offset, 32);
        input6filter8 += __shfl_down(input6filter8, offset, 32);

        input7filter1 += __shfl_down(input7filter1, offset, 32);
        input7filter2 += __shfl_down(input7filter2, offset, 32);
        input7filter3 += __shfl_down(input7filter3, offset, 32);
        input7filter4 += __shfl_down(input7filter4, offset, 32);
        input7filter5 += __shfl_down(input7filter5, offset, 32);

        input7filter6 += __shfl_down(input7filter6, offset, 32);
        input7filter7 += __shfl_down(input7filter7, offset, 32);
        input7filter8 += __shfl_down(input7filter8, offset, 32);

        input8filter1 += __shfl_down(input8filter1, offset, 32);
        input8filter2 += __shfl_down(input8filter2, offset, 32);
        input8filter3 += __shfl_down(input8filter3, offset, 32);
        input8filter4 += __shfl_down(input8filter4, offset, 32);
        input8filter5 += __shfl_down(input8filter5, offset, 32);

        input8filter6 += __shfl_down(input8filter6, offset, 32);
        input8filter7 += __shfl_down(input8filter7, offset, 32);
        input8filter8 += __shfl_down(input8filter8, offset, 32);

        input9filter1 += __shfl_down(input9filter1, offset, 32);
        input9filter2 += __shfl_down(input9filter2, offset, 32);
        input9filter3 += __shfl_down(input9filter3, offset, 32);
        input9filter4 += __shfl_down(input9filter4, offset, 32);
        input9filter5 += __shfl_down(input9filter5, offset, 32);

        input9filter6 += __shfl_down(input9filter6, offset, 32);
        input9filter7 += __shfl_down(input9filter7, offset, 32);
        input9filter8 += __shfl_down(input9filter8, offset, 32);

        input10filter1 += __shfl_down(input10filter1, offset, 32);
        input10filter2 += __shfl_down(input10filter2, offset, 32);
        input10filter3 += __shfl_down(input10filter3, offset, 32);
        input10filter4 += __shfl_down(input10filter4, offset, 32);
        input10filter5 += __shfl_down(input10filter5, offset, 32);

        input10filter6 += __shfl_down(input10filter6, offset, 32);
        input10filter7 += __shfl_down(input10filter7, offset, 32);
        input10filter8 += __shfl_down(input10filter8, offset, 32);

        input11filter1 += __shfl_down(input11filter1, offset, 32);
        input11filter2 += __shfl_down(input11filter2, offset, 32);
        input11filter3 += __shfl_down(input11filter3, offset, 32);
        input11filter4 += __shfl_down(input11filter4, offset, 32);
        input11filter5 += __shfl_down(input11filter5, offset, 32);

        input11filter6 += __shfl_down(input11filter6, offset, 32);
        input11filter7 += __shfl_down(input11filter7, offset, 32);
        input11filter8 += __shfl_down(input11filter8, offset, 32);

        input12filter1 += __shfl_down(input12filter1, offset, 32);
        input12filter2 += __shfl_down(input12filter2, offset, 32);
        input12filter3 += __shfl_down(input12filter3, offset, 32);
        input12filter4 += __shfl_down(input12filter4, offset, 32);
        input12filter5 += __shfl_down(input12filter5, offset, 32);

        input12filter6 += __shfl_down(input12filter6, offset, 32);
        input12filter7 += __shfl_down(input12filter7, offset, 32);
        input12filter8 += __shfl_down(input12filter8, offset, 32);

        input13filter1 += __shfl_down(input13filter1, offset, 32);
        input13filter2 += __shfl_down(input13filter2, offset, 32);
        input13filter3 += __shfl_down(input13filter3, offset, 32);
        input13filter4 += __shfl_down(input13filter4, offset, 32);
        input13filter5 += __shfl_down(input13filter5, offset, 32);

        input13filter6 += __shfl_down(input13filter6, offset, 32);
        input13filter7 += __shfl_down(input13filter7, offset, 32);
        input13filter8 += __shfl_down(input13filter8, offset, 32);

        input14filter1 += __shfl_down(input14filter1, offset, 32);
        input14filter2 += __shfl_down(input14filter2, offset, 32);
        input14filter3 += __shfl_down(input14filter3, offset, 32);
        input14filter4 += __shfl_down(input14filter4, offset, 32);
        input14filter5 += __shfl_down(input14filter5, offset, 32);

        input14filter6 += __shfl_down(input14filter6, offset, 32);
        input14filter7 += __shfl_down(input14filter7, offset, 32);
        input14filter8 += __shfl_down(input14filter8, offset, 32);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 21 * 18816 + (blockIdx.x % 21) / 3 * 28 + (blockIdx.x % 3) / 3 * 14 + (blockIdx.x % 3) * 6272;

    if(laneID % 32 == 0) {
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter1;
        output[blockWriteOutputStartIdx + 0 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter1;

        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter2;
        output[blockWriteOutputStartIdx + 1 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter2;

        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter3;
        output[blockWriteOutputStartIdx + 2 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter3;

        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter4;
        output[blockWriteOutputStartIdx + 3 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter4;

        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter5;
        output[blockWriteOutputStartIdx + 4 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter5;

        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter6;
        output[blockWriteOutputStartIdx + 5 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter6;

        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter7;
        output[blockWriteOutputStartIdx + 6 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter7;

        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 3] = input4filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 4] = input5filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 5] = input6filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 6] = input7filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 7] = input8filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 8] = input9filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 9] = input10filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 10] = input11filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 11] = input12filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 12] = input13filter8;
        output[blockWriteOutputStartIdx + 7 * 2 * outputHeight * outputWidth + (warpID / 2) / 1 * outputWidth + ((warpID / 2) % 1) * 14 + (warpID % 2) * 3136 + (laneID / 32) * outputHeight * outputWidth + 13] = input14filter8;
    }
}

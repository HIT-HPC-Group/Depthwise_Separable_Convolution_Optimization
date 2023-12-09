/*
Pointwise Convolution Kernel
InputBatch_8_Input_7x7_InChannel_160_OutChannel_960

Grid:
    gridDim.x = (8 * 960 * 7 * 7) / (4 * 7 * 120);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 7
outputChannelPerWarp = 120
channelGroupSize = 8
horizontalRepeat = 1
verticalRepeat = 4

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 7 * 120 output data.
Each warp is responsible for generating 7 * 120 output data.
*/

__global__ void InputBatch_8_Input_7x7_InChannel_160_OutChannel_960(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[1 * 7 * 8];
    __shared__ float inputSharedBuffer2[1 * 7 * 8];

    __shared__ float filterSharedBuffer1[4 * 120 * 8];
    __shared__ float filterSharedBuffer2[4 * 120 * 8];

    // to hold loaded operands temp.
    float inputTemp1 = 0;
    float filterTemp1 = 0, filterTemp2 = 0, filterTemp3 = 0, filterTemp4 = 0, filterTemp5 = 0;
    float filterTemp6 = 0, filterTemp7 = 0, filterTemp8 = 0, filterTemp9 = 0, filterTemp10 = 0;
    float filterTemp11 = 0, filterTemp12 = 0, filterTemp13 = 0, filterTemp14 = 0, filterTemp15 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0;
    float filterOperand1 = 0, filterOperand2 = 0, filterOperand3 = 0, filterOperand4 = 0, filterOperand5 = 0;
    float filterOperand6 = 0, filterOperand7 = 0, filterOperand8 = 0, filterOperand9 = 0, filterOperand10 = 0;
    float filterOperand11 = 0, filterOperand12 = 0, filterOperand13 = 0, filterOperand14 = 0, filterOperand15 = 0;

    // to hold intermediate result
    float input1filter1 = 0, input1filter2 = 0, input1filter3 = 0, input1filter4 = 0, input1filter5 = 0;
    float input1filter6 = 0, input1filter7 = 0, input1filter8 = 0, input1filter9 = 0, input1filter10 = 0;
    float input1filter11 = 0, input1filter12 = 0, input1filter13 = 0, input1filter14 = 0, input1filter15 = 0;

    float input2filter1 = 0, input2filter2 = 0, input2filter3 = 0, input2filter4 = 0, input2filter5 = 0;
    float input2filter6 = 0, input2filter7 = 0, input2filter8 = 0, input2filter9 = 0, input2filter10 = 0;
    float input2filter11 = 0, input2filter12 = 0, input2filter13 = 0, input2filter14 = 0, input2filter15 = 0;

    float input3filter1 = 0, input3filter2 = 0, input3filter3 = 0, input3filter4 = 0, input3filter5 = 0;
    float input3filter6 = 0, input3filter7 = 0, input3filter8 = 0, input3filter9 = 0, input3filter10 = 0;
    float input3filter11 = 0, input3filter12 = 0, input3filter13 = 0, input3filter14 = 0, input3filter15 = 0;

    float input4filter1 = 0, input4filter2 = 0, input4filter3 = 0, input4filter4 = 0, input4filter5 = 0;
    float input4filter6 = 0, input4filter7 = 0, input4filter8 = 0, input4filter9 = 0, input4filter10 = 0;
    float input4filter11 = 0, input4filter12 = 0, input4filter13 = 0, input4filter14 = 0, input4filter15 = 0;

    float input5filter1 = 0, input5filter2 = 0, input5filter3 = 0, input5filter4 = 0, input5filter5 = 0;
    float input5filter6 = 0, input5filter7 = 0, input5filter8 = 0, input5filter9 = 0, input5filter10 = 0;
    float input5filter11 = 0, input5filter12 = 0, input5filter13 = 0, input5filter14 = 0, input5filter15 = 0;

    float input6filter1 = 0, input6filter2 = 0, input6filter3 = 0, input6filter4 = 0, input6filter5 = 0;
    float input6filter6 = 0, input6filter7 = 0, input6filter8 = 0, input6filter9 = 0, input6filter10 = 0;
    float input6filter11 = 0, input6filter12 = 0, input6filter13 = 0, input6filter14 = 0, input6filter15 = 0;

    float input7filter1 = 0, input7filter2 = 0, input7filter3 = 0, input7filter4 = 0, input7filter5 = 0;
    float input7filter6 = 0, input7filter7 = 0, input7filter8 = 0, input7filter9 = 0, input7filter10 = 0;
    float input7filter11 = 0, input7filter12 = 0, input7filter13 = 0, input7filter14 = 0, input7filter15 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 14 * 7840 + (blockIdx.x % 14) / 2 * 7 + (blockIdx.x % 2) / 2 * 7;
    if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
        inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 2) * 76800;
    filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 160 + ((threadIdx.x + 0 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 1 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 160 + ((threadIdx.x + 1 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 2 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 8) * 160 + ((threadIdx.x + 2 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 3 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 8) * 160 + ((threadIdx.x + 3 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 4 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 8) * 160 + ((threadIdx.x + 4 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 5 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 8) * 160 + ((threadIdx.x + 5 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 6 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 8) * 160 + ((threadIdx.x + 6 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 7 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 8) * 160 + ((threadIdx.x + 7 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 8 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 8 * 256) / 8) * 160 + ((threadIdx.x + 8 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 9 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 9 * 256) / 8) * 160 + ((threadIdx.x + 9 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 10 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 10 * 256) / 8) * 160 + ((threadIdx.x + 10 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 11 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 11 * 256) / 8) * 160 + ((threadIdx.x + 11 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 12 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 12 * 256) / 8) * 160 + ((threadIdx.x + 12 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 13 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 13 * 256) / 8) * 160 + ((threadIdx.x + 13 * 256) % 8)];
    filterSharedBuffer1[threadIdx.x + 14 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 14 * 256) / 8) * 160 + ((threadIdx.x + 14 * 256) % 8)];
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 8) / (2 * 8); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 8;
        if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
        }

        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 160 + ((threadIdx.x + 0 * 256) % 8)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 160 + ((threadIdx.x + 1 * 256) % 8)];
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 8) * 160 + ((threadIdx.x + 2 * 256) % 8)];
        filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 8) * 160 + ((threadIdx.x + 3 * 256) % 8)];
        filterTemp5 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 8) * 160 + ((threadIdx.x + 4 * 256) % 8)];
        filterTemp6 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 8) * 160 + ((threadIdx.x + 5 * 256) % 8)];
        filterTemp7 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 8) * 160 + ((threadIdx.x + 6 * 256) % 8)];
        filterTemp8 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 8) * 160 + ((threadIdx.x + 7 * 256) % 8)];
        filterTemp9 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 8 * 256) / 8) * 160 + ((threadIdx.x + 8 * 256) % 8)];
        filterTemp10 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 9 * 256) / 8) * 160 + ((threadIdx.x + 9 * 256) % 8)];
        filterTemp11 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 10 * 256) / 8) * 160 + ((threadIdx.x + 10 * 256) % 8)];
        filterTemp12 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 11 * 256) / 8) * 160 + ((threadIdx.x + 11 * 256) % 8)];
        filterTemp13 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 12 * 256) / 8) * 160 + ((threadIdx.x + 12 * 256) % 8)];
        filterTemp14 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 13 * 256) / 8) * 160 + ((threadIdx.x + 13 * 256) % 8)];
        filterTemp15 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 14 * 256) / 8) * 160 + ((threadIdx.x + 14 * 256) % 8)];

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 6];

        filterOperand1 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 7 * 64];
        filterOperand9 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 8 * 64];
        filterOperand10 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 9 * 64];

        filterOperand11 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 10 * 64];
        filterOperand12 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 11 * 64];
        filterOperand13 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 12 * 64];
        filterOperand14 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 13 * 64];
        filterOperand15 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 14 * 64];

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
        input1filter10 += inputOperand1 * filterOperand10;

        input1filter11 += inputOperand1 * filterOperand11;
        input1filter12 += inputOperand1 * filterOperand12;
        input1filter13 += inputOperand1 * filterOperand13;
        input1filter14 += inputOperand1 * filterOperand14;
        input1filter15 += inputOperand1 * filterOperand15;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input2filter11 += inputOperand2 * filterOperand11;
        input2filter12 += inputOperand2 * filterOperand12;
        input2filter13 += inputOperand2 * filterOperand13;
        input2filter14 += inputOperand2 * filterOperand14;
        input2filter15 += inputOperand2 * filterOperand15;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input3filter11 += inputOperand3 * filterOperand11;
        input3filter12 += inputOperand3 * filterOperand12;
        input3filter13 += inputOperand3 * filterOperand13;
        input3filter14 += inputOperand3 * filterOperand14;
        input3filter15 += inputOperand3 * filterOperand15;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input4filter11 += inputOperand4 * filterOperand11;
        input4filter12 += inputOperand4 * filterOperand12;
        input4filter13 += inputOperand4 * filterOperand13;
        input4filter14 += inputOperand4 * filterOperand14;
        input4filter15 += inputOperand4 * filterOperand15;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input5filter11 += inputOperand5 * filterOperand11;
        input5filter12 += inputOperand5 * filterOperand12;
        input5filter13 += inputOperand5 * filterOperand13;
        input5filter14 += inputOperand5 * filterOperand14;
        input5filter15 += inputOperand5 * filterOperand15;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input6filter11 += inputOperand6 * filterOperand11;
        input6filter12 += inputOperand6 * filterOperand12;
        input6filter13 += inputOperand6 * filterOperand13;
        input6filter14 += inputOperand6 * filterOperand14;
        input6filter15 += inputOperand6 * filterOperand15;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        input7filter11 += inputOperand7 * filterOperand11;
        input7filter12 += inputOperand7 * filterOperand12;
        input7filter13 += inputOperand7 * filterOperand13;
        input7filter14 += inputOperand7 * filterOperand14;
        input7filter15 += inputOperand7 * filterOperand15;

        // Copy Temp Registers to shared buffer 2
        if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
            inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
        filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
        filterSharedBuffer2[threadIdx.x + 3 * 256] = filterTemp4;
        filterSharedBuffer2[threadIdx.x + 4 * 256] = filterTemp5;
        filterSharedBuffer2[threadIdx.x + 5 * 256] = filterTemp6;
        filterSharedBuffer2[threadIdx.x + 6 * 256] = filterTemp7;
        filterSharedBuffer2[threadIdx.x + 7 * 256] = filterTemp8;
        filterSharedBuffer2[threadIdx.x + 8 * 256] = filterTemp9;
        filterSharedBuffer2[threadIdx.x + 9 * 256] = filterTemp10;
        filterSharedBuffer2[threadIdx.x + 10 * 256] = filterTemp11;
        filterSharedBuffer2[threadIdx.x + 11 * 256] = filterTemp12;
        filterSharedBuffer2[threadIdx.x + 12 * 256] = filterTemp13;
        filterSharedBuffer2[threadIdx.x + 13 * 256] = filterTemp14;
        filterSharedBuffer2[threadIdx.x + 14 * 256] = filterTemp15;
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 7 * 7 * 8;
        if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
            inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
        }

        blockLoadFilterStartIdx += 8;
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 160 + ((threadIdx.x + 0 * 256) % 8)];
        filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 160 + ((threadIdx.x + 1 * 256) % 8)];
        filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 8) * 160 + ((threadIdx.x + 2 * 256) % 8)];
        filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 8) * 160 + ((threadIdx.x + 3 * 256) % 8)];
        filterTemp5 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 8) * 160 + ((threadIdx.x + 4 * 256) % 8)];
        filterTemp6 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 8) * 160 + ((threadIdx.x + 5 * 256) % 8)];
        filterTemp7 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 8) * 160 + ((threadIdx.x + 6 * 256) % 8)];
        filterTemp8 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 8) * 160 + ((threadIdx.x + 7 * 256) % 8)];
        filterTemp9 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 8 * 256) / 8) * 160 + ((threadIdx.x + 8 * 256) % 8)];
        filterTemp10 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 9 * 256) / 8) * 160 + ((threadIdx.x + 9 * 256) % 8)];
        filterTemp11 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 10 * 256) / 8) * 160 + ((threadIdx.x + 10 * 256) % 8)];
        filterTemp12 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 11 * 256) / 8) * 160 + ((threadIdx.x + 11 * 256) % 8)];
        filterTemp13 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 12 * 256) / 8) * 160 + ((threadIdx.x + 12 * 256) % 8)];
        filterTemp14 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 13 * 256) / 8) * 160 + ((threadIdx.x + 13 * 256) % 8)];
        filterTemp15 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 14 * 256) / 8) * 160 + ((threadIdx.x + 14 * 256) % 8)];

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 6];

        filterOperand1 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 0 * 64];
        filterOperand2 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 1 * 64];
        filterOperand3 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 2 * 64];
        filterOperand4 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 3 * 64];
        filterOperand5 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 4 * 64];

        filterOperand6 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 5 * 64];
        filterOperand7 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 6 * 64];
        filterOperand8 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 7 * 64];
        filterOperand9 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 8 * 64];
        filterOperand10 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 9 * 64];

        filterOperand11 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 10 * 64];
        filterOperand12 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 11 * 64];
        filterOperand13 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 12 * 64];
        filterOperand14 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 13 * 64];
        filterOperand15 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 14 * 64];

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
        input1filter10 += inputOperand1 * filterOperand10;

        input1filter11 += inputOperand1 * filterOperand11;
        input1filter12 += inputOperand1 * filterOperand12;
        input1filter13 += inputOperand1 * filterOperand13;
        input1filter14 += inputOperand1 * filterOperand14;
        input1filter15 += inputOperand1 * filterOperand15;

        input2filter1 += inputOperand2 * filterOperand1;
        input2filter2 += inputOperand2 * filterOperand2;
        input2filter3 += inputOperand2 * filterOperand3;
        input2filter4 += inputOperand2 * filterOperand4;
        input2filter5 += inputOperand2 * filterOperand5;

        input2filter6 += inputOperand2 * filterOperand6;
        input2filter7 += inputOperand2 * filterOperand7;
        input2filter8 += inputOperand2 * filterOperand8;
        input2filter9 += inputOperand2 * filterOperand9;
        input2filter10 += inputOperand2 * filterOperand10;

        input2filter11 += inputOperand2 * filterOperand11;
        input2filter12 += inputOperand2 * filterOperand12;
        input2filter13 += inputOperand2 * filterOperand13;
        input2filter14 += inputOperand2 * filterOperand14;
        input2filter15 += inputOperand2 * filterOperand15;

        input3filter1 += inputOperand3 * filterOperand1;
        input3filter2 += inputOperand3 * filterOperand2;
        input3filter3 += inputOperand3 * filterOperand3;
        input3filter4 += inputOperand3 * filterOperand4;
        input3filter5 += inputOperand3 * filterOperand5;

        input3filter6 += inputOperand3 * filterOperand6;
        input3filter7 += inputOperand3 * filterOperand7;
        input3filter8 += inputOperand3 * filterOperand8;
        input3filter9 += inputOperand3 * filterOperand9;
        input3filter10 += inputOperand3 * filterOperand10;

        input3filter11 += inputOperand3 * filterOperand11;
        input3filter12 += inputOperand3 * filterOperand12;
        input3filter13 += inputOperand3 * filterOperand13;
        input3filter14 += inputOperand3 * filterOperand14;
        input3filter15 += inputOperand3 * filterOperand15;

        input4filter1 += inputOperand4 * filterOperand1;
        input4filter2 += inputOperand4 * filterOperand2;
        input4filter3 += inputOperand4 * filterOperand3;
        input4filter4 += inputOperand4 * filterOperand4;
        input4filter5 += inputOperand4 * filterOperand5;

        input4filter6 += inputOperand4 * filterOperand6;
        input4filter7 += inputOperand4 * filterOperand7;
        input4filter8 += inputOperand4 * filterOperand8;
        input4filter9 += inputOperand4 * filterOperand9;
        input4filter10 += inputOperand4 * filterOperand10;

        input4filter11 += inputOperand4 * filterOperand11;
        input4filter12 += inputOperand4 * filterOperand12;
        input4filter13 += inputOperand4 * filterOperand13;
        input4filter14 += inputOperand4 * filterOperand14;
        input4filter15 += inputOperand4 * filterOperand15;

        input5filter1 += inputOperand5 * filterOperand1;
        input5filter2 += inputOperand5 * filterOperand2;
        input5filter3 += inputOperand5 * filterOperand3;
        input5filter4 += inputOperand5 * filterOperand4;
        input5filter5 += inputOperand5 * filterOperand5;

        input5filter6 += inputOperand5 * filterOperand6;
        input5filter7 += inputOperand5 * filterOperand7;
        input5filter8 += inputOperand5 * filterOperand8;
        input5filter9 += inputOperand5 * filterOperand9;
        input5filter10 += inputOperand5 * filterOperand10;

        input5filter11 += inputOperand5 * filterOperand11;
        input5filter12 += inputOperand5 * filterOperand12;
        input5filter13 += inputOperand5 * filterOperand13;
        input5filter14 += inputOperand5 * filterOperand14;
        input5filter15 += inputOperand5 * filterOperand15;

        input6filter1 += inputOperand6 * filterOperand1;
        input6filter2 += inputOperand6 * filterOperand2;
        input6filter3 += inputOperand6 * filterOperand3;
        input6filter4 += inputOperand6 * filterOperand4;
        input6filter5 += inputOperand6 * filterOperand5;

        input6filter6 += inputOperand6 * filterOperand6;
        input6filter7 += inputOperand6 * filterOperand7;
        input6filter8 += inputOperand6 * filterOperand8;
        input6filter9 += inputOperand6 * filterOperand9;
        input6filter10 += inputOperand6 * filterOperand10;

        input6filter11 += inputOperand6 * filterOperand11;
        input6filter12 += inputOperand6 * filterOperand12;
        input6filter13 += inputOperand6 * filterOperand13;
        input6filter14 += inputOperand6 * filterOperand14;
        input6filter15 += inputOperand6 * filterOperand15;

        input7filter1 += inputOperand7 * filterOperand1;
        input7filter2 += inputOperand7 * filterOperand2;
        input7filter3 += inputOperand7 * filterOperand3;
        input7filter4 += inputOperand7 * filterOperand4;
        input7filter5 += inputOperand7 * filterOperand5;

        input7filter6 += inputOperand7 * filterOperand6;
        input7filter7 += inputOperand7 * filterOperand7;
        input7filter8 += inputOperand7 * filterOperand8;
        input7filter9 += inputOperand7 * filterOperand9;
        input7filter10 += inputOperand7 * filterOperand10;

        input7filter11 += inputOperand7 * filterOperand11;
        input7filter12 += inputOperand7 * filterOperand12;
        input7filter13 += inputOperand7 * filterOperand13;
        input7filter14 += inputOperand7 * filterOperand14;
        input7filter15 += inputOperand7 * filterOperand15;

        // Copy Temp Registers to shared buffer 1
        if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
            inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        }

        filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        filterSharedBuffer1[threadIdx.x + 1 * 256] = filterTemp2;
        filterSharedBuffer1[threadIdx.x + 2 * 256] = filterTemp3;
        filterSharedBuffer1[threadIdx.x + 3 * 256] = filterTemp4;
        filterSharedBuffer1[threadIdx.x + 4 * 256] = filterTemp5;
        filterSharedBuffer1[threadIdx.x + 5 * 256] = filterTemp6;
        filterSharedBuffer1[threadIdx.x + 6 * 256] = filterTemp7;
        filterSharedBuffer1[threadIdx.x + 7 * 256] = filterTemp8;
        filterSharedBuffer1[threadIdx.x + 8 * 256] = filterTemp9;
        filterSharedBuffer1[threadIdx.x + 9 * 256] = filterTemp10;
        filterSharedBuffer1[threadIdx.x + 10 * 256] = filterTemp11;
        filterSharedBuffer1[threadIdx.x + 11 * 256] = filterTemp12;
        filterSharedBuffer1[threadIdx.x + 12 * 256] = filterTemp13;
        filterSharedBuffer1[threadIdx.x + 13 * 256] = filterTemp14;
        filterSharedBuffer1[threadIdx.x + 14 * 256] = filterTemp15;
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 7 * 7 * 8;
    if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 7 * 49 + ((threadIdx.x + 0 * 256) % 7) / 7 * 7 + (threadIdx.x + 0 * 256) % 7];
    }

    blockLoadFilterStartIdx += 8;
    filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 8) * 160 + ((threadIdx.x + 0 * 256) % 8)];
    filterTemp2 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 1 * 256) / 8) * 160 + ((threadIdx.x + 1 * 256) % 8)];
    filterTemp3 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 2 * 256) / 8) * 160 + ((threadIdx.x + 2 * 256) % 8)];
    filterTemp4 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 3 * 256) / 8) * 160 + ((threadIdx.x + 3 * 256) % 8)];
    filterTemp5 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 4 * 256) / 8) * 160 + ((threadIdx.x + 4 * 256) % 8)];
    filterTemp6 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 5 * 256) / 8) * 160 + ((threadIdx.x + 5 * 256) % 8)];
    filterTemp7 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 6 * 256) / 8) * 160 + ((threadIdx.x + 6 * 256) % 8)];
    filterTemp8 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 7 * 256) / 8) * 160 + ((threadIdx.x + 7 * 256) % 8)];
    filterTemp9 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 8 * 256) / 8) * 160 + ((threadIdx.x + 8 * 256) % 8)];
    filterTemp10 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 9 * 256) / 8) * 160 + ((threadIdx.x + 9 * 256) % 8)];
    filterTemp11 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 10 * 256) / 8) * 160 + ((threadIdx.x + 10 * 256) % 8)];
    filterTemp12 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 11 * 256) / 8) * 160 + ((threadIdx.x + 11 * 256) % 8)];
    filterTemp13 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 12 * 256) / 8) * 160 + ((threadIdx.x + 12 * 256) % 8)];
    filterTemp14 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 13 * 256) / 8) * 160 + ((threadIdx.x + 13 * 256) % 8)];
    filterTemp15 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 14 * 256) / 8) * 160 + ((threadIdx.x + 14 * 256) % 8)];

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 6];

    filterOperand1 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 7 * 64];
    filterOperand9 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 8 * 64];
    filterOperand10 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 9 * 64];

    filterOperand11 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 10 * 64];
    filterOperand12 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 11 * 64];
    filterOperand13 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 12 * 64];
    filterOperand14 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 13 * 64];
    filterOperand15 = filterSharedBuffer1[(warpID % 4) * 960 + laneID + 14 * 64];

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
    input1filter10 += inputOperand1 * filterOperand10;

    input1filter11 += inputOperand1 * filterOperand11;
    input1filter12 += inputOperand1 * filterOperand12;
    input1filter13 += inputOperand1 * filterOperand13;
    input1filter14 += inputOperand1 * filterOperand14;
    input1filter15 += inputOperand1 * filterOperand15;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input2filter6 += inputOperand2 * filterOperand6;
    input2filter7 += inputOperand2 * filterOperand7;
    input2filter8 += inputOperand2 * filterOperand8;
    input2filter9 += inputOperand2 * filterOperand9;
    input2filter10 += inputOperand2 * filterOperand10;

    input2filter11 += inputOperand2 * filterOperand11;
    input2filter12 += inputOperand2 * filterOperand12;
    input2filter13 += inputOperand2 * filterOperand13;
    input2filter14 += inputOperand2 * filterOperand14;
    input2filter15 += inputOperand2 * filterOperand15;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input3filter6 += inputOperand3 * filterOperand6;
    input3filter7 += inputOperand3 * filterOperand7;
    input3filter8 += inputOperand3 * filterOperand8;
    input3filter9 += inputOperand3 * filterOperand9;
    input3filter10 += inputOperand3 * filterOperand10;

    input3filter11 += inputOperand3 * filterOperand11;
    input3filter12 += inputOperand3 * filterOperand12;
    input3filter13 += inputOperand3 * filterOperand13;
    input3filter14 += inputOperand3 * filterOperand14;
    input3filter15 += inputOperand3 * filterOperand15;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input4filter6 += inputOperand4 * filterOperand6;
    input4filter7 += inputOperand4 * filterOperand7;
    input4filter8 += inputOperand4 * filterOperand8;
    input4filter9 += inputOperand4 * filterOperand9;
    input4filter10 += inputOperand4 * filterOperand10;

    input4filter11 += inputOperand4 * filterOperand11;
    input4filter12 += inputOperand4 * filterOperand12;
    input4filter13 += inputOperand4 * filterOperand13;
    input4filter14 += inputOperand4 * filterOperand14;
    input4filter15 += inputOperand4 * filterOperand15;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;
    input5filter4 += inputOperand5 * filterOperand4;
    input5filter5 += inputOperand5 * filterOperand5;

    input5filter6 += inputOperand5 * filterOperand6;
    input5filter7 += inputOperand5 * filterOperand7;
    input5filter8 += inputOperand5 * filterOperand8;
    input5filter9 += inputOperand5 * filterOperand9;
    input5filter10 += inputOperand5 * filterOperand10;

    input5filter11 += inputOperand5 * filterOperand11;
    input5filter12 += inputOperand5 * filterOperand12;
    input5filter13 += inputOperand5 * filterOperand13;
    input5filter14 += inputOperand5 * filterOperand14;
    input5filter15 += inputOperand5 * filterOperand15;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;
    input6filter4 += inputOperand6 * filterOperand4;
    input6filter5 += inputOperand6 * filterOperand5;

    input6filter6 += inputOperand6 * filterOperand6;
    input6filter7 += inputOperand6 * filterOperand7;
    input6filter8 += inputOperand6 * filterOperand8;
    input6filter9 += inputOperand6 * filterOperand9;
    input6filter10 += inputOperand6 * filterOperand10;

    input6filter11 += inputOperand6 * filterOperand11;
    input6filter12 += inputOperand6 * filterOperand12;
    input6filter13 += inputOperand6 * filterOperand13;
    input6filter14 += inputOperand6 * filterOperand14;
    input6filter15 += inputOperand6 * filterOperand15;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;
    input7filter4 += inputOperand7 * filterOperand4;
    input7filter5 += inputOperand7 * filterOperand5;

    input7filter6 += inputOperand7 * filterOperand6;
    input7filter7 += inputOperand7 * filterOperand7;
    input7filter8 += inputOperand7 * filterOperand8;
    input7filter9 += inputOperand7 * filterOperand9;
    input7filter10 += inputOperand7 * filterOperand10;

    input7filter11 += inputOperand7 * filterOperand11;
    input7filter12 += inputOperand7 * filterOperand12;
    input7filter13 += inputOperand7 * filterOperand13;
    input7filter14 += inputOperand7 * filterOperand14;
    input7filter15 += inputOperand7 * filterOperand15;

    // Copy Temp Registers to shared buffer 2
    if(threadIdx.x < 1 * 7 * 8 - 0 * 256) {
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    }

    filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    filterSharedBuffer2[threadIdx.x + 1 * 256] = filterTemp2;
    filterSharedBuffer2[threadIdx.x + 2 * 256] = filterTemp3;
    filterSharedBuffer2[threadIdx.x + 3 * 256] = filterTemp4;
    filterSharedBuffer2[threadIdx.x + 4 * 256] = filterTemp5;
    filterSharedBuffer2[threadIdx.x + 5 * 256] = filterTemp6;
    filterSharedBuffer2[threadIdx.x + 6 * 256] = filterTemp7;
    filterSharedBuffer2[threadIdx.x + 7 * 256] = filterTemp8;
    filterSharedBuffer2[threadIdx.x + 8 * 256] = filterTemp9;
    filterSharedBuffer2[threadIdx.x + 9 * 256] = filterTemp10;
    filterSharedBuffer2[threadIdx.x + 10 * 256] = filterTemp11;
    filterSharedBuffer2[threadIdx.x + 11 * 256] = filterTemp12;
    filterSharedBuffer2[threadIdx.x + 12 * 256] = filterTemp13;
    filterSharedBuffer2[threadIdx.x + 13 * 256] = filterTemp14;
    filterSharedBuffer2[threadIdx.x + 14 * 256] = filterTemp15;
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 4) % 1) * 7 + (laneID % 8) * 7 * 1 + 6];

    filterOperand1 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 0 * 64];
    filterOperand2 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 1 * 64];
    filterOperand3 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 2 * 64];
    filterOperand4 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 3 * 64];
    filterOperand5 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 4 * 64];

    filterOperand6 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 5 * 64];
    filterOperand7 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 6 * 64];
    filterOperand8 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 7 * 64];
    filterOperand9 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 8 * 64];
    filterOperand10 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 9 * 64];

    filterOperand11 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 10 * 64];
    filterOperand12 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 11 * 64];
    filterOperand13 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 12 * 64];
    filterOperand14 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 13 * 64];
    filterOperand15 = filterSharedBuffer2[(warpID % 4) * 960 + laneID + 14 * 64];

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
    input1filter10 += inputOperand1 * filterOperand10;

    input1filter11 += inputOperand1 * filterOperand11;
    input1filter12 += inputOperand1 * filterOperand12;
    input1filter13 += inputOperand1 * filterOperand13;
    input1filter14 += inputOperand1 * filterOperand14;
    input1filter15 += inputOperand1 * filterOperand15;

    input2filter1 += inputOperand2 * filterOperand1;
    input2filter2 += inputOperand2 * filterOperand2;
    input2filter3 += inputOperand2 * filterOperand3;
    input2filter4 += inputOperand2 * filterOperand4;
    input2filter5 += inputOperand2 * filterOperand5;

    input2filter6 += inputOperand2 * filterOperand6;
    input2filter7 += inputOperand2 * filterOperand7;
    input2filter8 += inputOperand2 * filterOperand8;
    input2filter9 += inputOperand2 * filterOperand9;
    input2filter10 += inputOperand2 * filterOperand10;

    input2filter11 += inputOperand2 * filterOperand11;
    input2filter12 += inputOperand2 * filterOperand12;
    input2filter13 += inputOperand2 * filterOperand13;
    input2filter14 += inputOperand2 * filterOperand14;
    input2filter15 += inputOperand2 * filterOperand15;

    input3filter1 += inputOperand3 * filterOperand1;
    input3filter2 += inputOperand3 * filterOperand2;
    input3filter3 += inputOperand3 * filterOperand3;
    input3filter4 += inputOperand3 * filterOperand4;
    input3filter5 += inputOperand3 * filterOperand5;

    input3filter6 += inputOperand3 * filterOperand6;
    input3filter7 += inputOperand3 * filterOperand7;
    input3filter8 += inputOperand3 * filterOperand8;
    input3filter9 += inputOperand3 * filterOperand9;
    input3filter10 += inputOperand3 * filterOperand10;

    input3filter11 += inputOperand3 * filterOperand11;
    input3filter12 += inputOperand3 * filterOperand12;
    input3filter13 += inputOperand3 * filterOperand13;
    input3filter14 += inputOperand3 * filterOperand14;
    input3filter15 += inputOperand3 * filterOperand15;

    input4filter1 += inputOperand4 * filterOperand1;
    input4filter2 += inputOperand4 * filterOperand2;
    input4filter3 += inputOperand4 * filterOperand3;
    input4filter4 += inputOperand4 * filterOperand4;
    input4filter5 += inputOperand4 * filterOperand5;

    input4filter6 += inputOperand4 * filterOperand6;
    input4filter7 += inputOperand4 * filterOperand7;
    input4filter8 += inputOperand4 * filterOperand8;
    input4filter9 += inputOperand4 * filterOperand9;
    input4filter10 += inputOperand4 * filterOperand10;

    input4filter11 += inputOperand4 * filterOperand11;
    input4filter12 += inputOperand4 * filterOperand12;
    input4filter13 += inputOperand4 * filterOperand13;
    input4filter14 += inputOperand4 * filterOperand14;
    input4filter15 += inputOperand4 * filterOperand15;

    input5filter1 += inputOperand5 * filterOperand1;
    input5filter2 += inputOperand5 * filterOperand2;
    input5filter3 += inputOperand5 * filterOperand3;
    input5filter4 += inputOperand5 * filterOperand4;
    input5filter5 += inputOperand5 * filterOperand5;

    input5filter6 += inputOperand5 * filterOperand6;
    input5filter7 += inputOperand5 * filterOperand7;
    input5filter8 += inputOperand5 * filterOperand8;
    input5filter9 += inputOperand5 * filterOperand9;
    input5filter10 += inputOperand5 * filterOperand10;

    input5filter11 += inputOperand5 * filterOperand11;
    input5filter12 += inputOperand5 * filterOperand12;
    input5filter13 += inputOperand5 * filterOperand13;
    input5filter14 += inputOperand5 * filterOperand14;
    input5filter15 += inputOperand5 * filterOperand15;

    input6filter1 += inputOperand6 * filterOperand1;
    input6filter2 += inputOperand6 * filterOperand2;
    input6filter3 += inputOperand6 * filterOperand3;
    input6filter4 += inputOperand6 * filterOperand4;
    input6filter5 += inputOperand6 * filterOperand5;

    input6filter6 += inputOperand6 * filterOperand6;
    input6filter7 += inputOperand6 * filterOperand7;
    input6filter8 += inputOperand6 * filterOperand8;
    input6filter9 += inputOperand6 * filterOperand9;
    input6filter10 += inputOperand6 * filterOperand10;

    input6filter11 += inputOperand6 * filterOperand11;
    input6filter12 += inputOperand6 * filterOperand12;
    input6filter13 += inputOperand6 * filterOperand13;
    input6filter14 += inputOperand6 * filterOperand14;
    input6filter15 += inputOperand6 * filterOperand15;

    input7filter1 += inputOperand7 * filterOperand1;
    input7filter2 += inputOperand7 * filterOperand2;
    input7filter3 += inputOperand7 * filterOperand3;
    input7filter4 += inputOperand7 * filterOperand4;
    input7filter5 += inputOperand7 * filterOperand5;

    input7filter6 += inputOperand7 * filterOperand6;
    input7filter7 += inputOperand7 * filterOperand7;
    input7filter8 += inputOperand7 * filterOperand8;
    input7filter9 += inputOperand7 * filterOperand9;
    input7filter10 += inputOperand7 * filterOperand10;

    input7filter11 += inputOperand7 * filterOperand11;
    input7filter12 += inputOperand7 * filterOperand12;
    input7filter13 += inputOperand7 * filterOperand13;
    input7filter14 += inputOperand7 * filterOperand14;
    input7filter15 += inputOperand7 * filterOperand15;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (8 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 8);
        input1filter2 += __shfl_down(input1filter2, offset, 8);
        input1filter3 += __shfl_down(input1filter3, offset, 8);
        input1filter4 += __shfl_down(input1filter4, offset, 8);
        input1filter5 += __shfl_down(input1filter5, offset, 8);

        input1filter6 += __shfl_down(input1filter6, offset, 8);
        input1filter7 += __shfl_down(input1filter7, offset, 8);
        input1filter8 += __shfl_down(input1filter8, offset, 8);
        input1filter9 += __shfl_down(input1filter9, offset, 8);
        input1filter10 += __shfl_down(input1filter10, offset, 8);

        input1filter11 += __shfl_down(input1filter11, offset, 8);
        input1filter12 += __shfl_down(input1filter12, offset, 8);
        input1filter13 += __shfl_down(input1filter13, offset, 8);
        input1filter14 += __shfl_down(input1filter14, offset, 8);
        input1filter15 += __shfl_down(input1filter15, offset, 8);

        input2filter1 += __shfl_down(input2filter1, offset, 8);
        input2filter2 += __shfl_down(input2filter2, offset, 8);
        input2filter3 += __shfl_down(input2filter3, offset, 8);
        input2filter4 += __shfl_down(input2filter4, offset, 8);
        input2filter5 += __shfl_down(input2filter5, offset, 8);

        input2filter6 += __shfl_down(input2filter6, offset, 8);
        input2filter7 += __shfl_down(input2filter7, offset, 8);
        input2filter8 += __shfl_down(input2filter8, offset, 8);
        input2filter9 += __shfl_down(input2filter9, offset, 8);
        input2filter10 += __shfl_down(input2filter10, offset, 8);

        input2filter11 += __shfl_down(input2filter11, offset, 8);
        input2filter12 += __shfl_down(input2filter12, offset, 8);
        input2filter13 += __shfl_down(input2filter13, offset, 8);
        input2filter14 += __shfl_down(input2filter14, offset, 8);
        input2filter15 += __shfl_down(input2filter15, offset, 8);

        input3filter1 += __shfl_down(input3filter1, offset, 8);
        input3filter2 += __shfl_down(input3filter2, offset, 8);
        input3filter3 += __shfl_down(input3filter3, offset, 8);
        input3filter4 += __shfl_down(input3filter4, offset, 8);
        input3filter5 += __shfl_down(input3filter5, offset, 8);

        input3filter6 += __shfl_down(input3filter6, offset, 8);
        input3filter7 += __shfl_down(input3filter7, offset, 8);
        input3filter8 += __shfl_down(input3filter8, offset, 8);
        input3filter9 += __shfl_down(input3filter9, offset, 8);
        input3filter10 += __shfl_down(input3filter10, offset, 8);

        input3filter11 += __shfl_down(input3filter11, offset, 8);
        input3filter12 += __shfl_down(input3filter12, offset, 8);
        input3filter13 += __shfl_down(input3filter13, offset, 8);
        input3filter14 += __shfl_down(input3filter14, offset, 8);
        input3filter15 += __shfl_down(input3filter15, offset, 8);

        input4filter1 += __shfl_down(input4filter1, offset, 8);
        input4filter2 += __shfl_down(input4filter2, offset, 8);
        input4filter3 += __shfl_down(input4filter3, offset, 8);
        input4filter4 += __shfl_down(input4filter4, offset, 8);
        input4filter5 += __shfl_down(input4filter5, offset, 8);

        input4filter6 += __shfl_down(input4filter6, offset, 8);
        input4filter7 += __shfl_down(input4filter7, offset, 8);
        input4filter8 += __shfl_down(input4filter8, offset, 8);
        input4filter9 += __shfl_down(input4filter9, offset, 8);
        input4filter10 += __shfl_down(input4filter10, offset, 8);

        input4filter11 += __shfl_down(input4filter11, offset, 8);
        input4filter12 += __shfl_down(input4filter12, offset, 8);
        input4filter13 += __shfl_down(input4filter13, offset, 8);
        input4filter14 += __shfl_down(input4filter14, offset, 8);
        input4filter15 += __shfl_down(input4filter15, offset, 8);

        input5filter1 += __shfl_down(input5filter1, offset, 8);
        input5filter2 += __shfl_down(input5filter2, offset, 8);
        input5filter3 += __shfl_down(input5filter3, offset, 8);
        input5filter4 += __shfl_down(input5filter4, offset, 8);
        input5filter5 += __shfl_down(input5filter5, offset, 8);

        input5filter6 += __shfl_down(input5filter6, offset, 8);
        input5filter7 += __shfl_down(input5filter7, offset, 8);
        input5filter8 += __shfl_down(input5filter8, offset, 8);
        input5filter9 += __shfl_down(input5filter9, offset, 8);
        input5filter10 += __shfl_down(input5filter10, offset, 8);

        input5filter11 += __shfl_down(input5filter11, offset, 8);
        input5filter12 += __shfl_down(input5filter12, offset, 8);
        input5filter13 += __shfl_down(input5filter13, offset, 8);
        input5filter14 += __shfl_down(input5filter14, offset, 8);
        input5filter15 += __shfl_down(input5filter15, offset, 8);

        input6filter1 += __shfl_down(input6filter1, offset, 8);
        input6filter2 += __shfl_down(input6filter2, offset, 8);
        input6filter3 += __shfl_down(input6filter3, offset, 8);
        input6filter4 += __shfl_down(input6filter4, offset, 8);
        input6filter5 += __shfl_down(input6filter5, offset, 8);

        input6filter6 += __shfl_down(input6filter6, offset, 8);
        input6filter7 += __shfl_down(input6filter7, offset, 8);
        input6filter8 += __shfl_down(input6filter8, offset, 8);
        input6filter9 += __shfl_down(input6filter9, offset, 8);
        input6filter10 += __shfl_down(input6filter10, offset, 8);

        input6filter11 += __shfl_down(input6filter11, offset, 8);
        input6filter12 += __shfl_down(input6filter12, offset, 8);
        input6filter13 += __shfl_down(input6filter13, offset, 8);
        input6filter14 += __shfl_down(input6filter14, offset, 8);
        input6filter15 += __shfl_down(input6filter15, offset, 8);

        input7filter1 += __shfl_down(input7filter1, offset, 8);
        input7filter2 += __shfl_down(input7filter2, offset, 8);
        input7filter3 += __shfl_down(input7filter3, offset, 8);
        input7filter4 += __shfl_down(input7filter4, offset, 8);
        input7filter5 += __shfl_down(input7filter5, offset, 8);

        input7filter6 += __shfl_down(input7filter6, offset, 8);
        input7filter7 += __shfl_down(input7filter7, offset, 8);
        input7filter8 += __shfl_down(input7filter8, offset, 8);
        input7filter9 += __shfl_down(input7filter9, offset, 8);
        input7filter10 += __shfl_down(input7filter10, offset, 8);

        input7filter11 += __shfl_down(input7filter11, offset, 8);
        input7filter12 += __shfl_down(input7filter12, offset, 8);
        input7filter13 += __shfl_down(input7filter13, offset, 8);
        input7filter14 += __shfl_down(input7filter14, offset, 8);
        input7filter15 += __shfl_down(input7filter15, offset, 8);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 14 * 47040 + (blockIdx.x % 14) / 2 * 7 + (blockIdx.x % 2) / 2 * 7 + (blockIdx.x % 2) * 23520;

    if(laneID % 8 == 0) {
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter1;

        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter2;
        output[blockWriteOutputStartIdx + 1 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter2;

        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter3;
        output[blockWriteOutputStartIdx + 2 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter3;

        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter4;
        output[blockWriteOutputStartIdx + 3 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter4;

        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter5;
        output[blockWriteOutputStartIdx + 4 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter5;

        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter6;
        output[blockWriteOutputStartIdx + 5 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter6;

        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter7;
        output[blockWriteOutputStartIdx + 6 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter7;

        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter8;
        output[blockWriteOutputStartIdx + 7 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter8;

        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter9;
        output[blockWriteOutputStartIdx + 8 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter9;

        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter10;
        output[blockWriteOutputStartIdx + 9 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter10;

        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter11;
        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter11;
        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter11;
        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter11;
        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter11;
        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter11;
        output[blockWriteOutputStartIdx + 10 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter11;

        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter12;
        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter12;
        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter12;
        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter12;
        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter12;
        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter12;
        output[blockWriteOutputStartIdx + 11 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter12;

        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter13;
        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter13;
        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter13;
        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter13;
        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter13;
        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter13;
        output[blockWriteOutputStartIdx + 12 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter13;

        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter14;
        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter14;
        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter14;
        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter14;
        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter14;
        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter14;
        output[blockWriteOutputStartIdx + 13 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter14;

        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 0] = input1filter15;
        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 1] = input2filter15;
        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 2] = input3filter15;
        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 3] = input4filter15;
        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 4] = input5filter15;
        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 5] = input6filter15;
        output[blockWriteOutputStartIdx + 14 * 8 * outputHeight * outputWidth + (warpID / 4) / 1 * outputWidth + ((warpID / 4) % 1) * 7 + (warpID % 4) * 5880 + (laneID / 8) * outputHeight * outputWidth + 6] = input7filter15;
    }
}

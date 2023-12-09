/*
Pointwise Convolution Kernel
InputBatch_128_Input_112x112_InChannel_32_OutChannel_16

Grid:
    gridDim.x = (128 * 16 * 112 * 112) / (4 * 28 * 16);
Block:
    blockDim.x = 64 * 4;

warpNumPerBlock = 4
outputWidthPerWarp = 28
outputChannelPerWarp = 16
channelGroupSize = 4
horizontalRepeat = 4
verticalRepeat = 1

One thread block contains 4 warps, 4 * 64 = 256 threads.
Each thread block is responsible for generating 4 * 28 * 16 output data.
Each warp is responsible for generating 28 * 16 output data.
*/

__global__ void InputBatch_128_Input_112x112_InChannel_32_OutChannel_16(const float* input, const float* filter, float* output,
    int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
    int filterOutChannel, int filterInChannel, int filterHeight, int filterWidth,
    int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth) {

    __shared__ float inputSharedBuffer1[4 * 28 * 4];
    __shared__ float inputSharedBuffer2[4 * 28 * 4];

    __shared__ float filterSharedBuffer1[1 * 16 * 4];
    __shared__ float filterSharedBuffer2[1 * 16 * 4];

    // to hold loaded operands temp.
    float inputTemp1 = 0, inputTemp2 = 0;
    float filterTemp1 = 0;

    // to hold operands
    float inputOperand1 = 0, inputOperand2 = 0, inputOperand3 = 0, inputOperand4 = 0, inputOperand5 = 0;
    float inputOperand6 = 0, inputOperand7 = 0, inputOperand8 = 0, inputOperand9 = 0, inputOperand10 = 0;
    float inputOperand11 = 0, inputOperand12 = 0, inputOperand13 = 0, inputOperand14 = 0, inputOperand15 = 0;
    float inputOperand16 = 0, inputOperand17 = 0, inputOperand18 = 0, inputOperand19 = 0, inputOperand20 = 0;
    float inputOperand21 = 0, inputOperand22 = 0, inputOperand23 = 0, inputOperand24 = 0, inputOperand25 = 0;
    float inputOperand26 = 0, inputOperand27 = 0, inputOperand28 = 0;
    float filterOperand1 = 0;

    // to hold intermediate result
    float input1filter1 = 0;

    float input2filter1 = 0;

    float input3filter1 = 0;

    float input4filter1 = 0;

    float input5filter1 = 0;

    float input6filter1 = 0;

    float input7filter1 = 0;

    float input8filter1 = 0;

    float input9filter1 = 0;

    float input10filter1 = 0;

    float input11filter1 = 0;

    float input12filter1 = 0;

    float input13filter1 = 0;

    float input14filter1 = 0;

    float input15filter1 = 0;

    float input16filter1 = 0;

    float input17filter1 = 0;

    float input18filter1 = 0;

    float input19filter1 = 0;

    float input20filter1 = 0;

    float input21filter1 = 0;

    float input22filter1 = 0;

    float input23filter1 = 0;

    float input24filter1 = 0;

    float input25filter1 = 0;

    float input26filter1 = 0;

    float input27filter1 = 0;

    float input28filter1 = 0;

    int warpID = threadIdx.x / 64;
    int laneID = threadIdx.x % 64;

    // input
    int blockLoadInputStartIdx = blockIdx.x / 112 * 401408 + (blockIdx.x % 112) / 4 * 448 + (blockIdx.x % 4) / 1 * 28;
    inputSharedBuffer1[threadIdx.x + 0 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 112 * 12544 + ((threadIdx.x + 0 * 256) % 112) / 28 * 112 + (threadIdx.x + 0 * 256) % 28];
    if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
        inputSharedBuffer1[threadIdx.x + 1 * 256] = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 112 * 12544 + ((threadIdx.x + 1 * 256) % 112) / 28 * 112 + (threadIdx.x + 1 * 256) % 28];
    }

    // filter
    int blockLoadFilterStartIdx = (blockIdx.x % 1) * 512;
    if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
        filterSharedBuffer1[threadIdx.x + 0 * 256] = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
    }
    __syncthreads();


    // For loop begins
    for(int i = 0; i < (inputChannel - 4) / (2 * 4); i++) {
        // load next group of Cnum channels
        blockLoadInputStartIdx += 112 * 112 * 4;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 112 * 12544 + ((threadIdx.x + 0 * 256) % 112) / 28 * 112 + (threadIdx.x + 0 * 256) % 28];
        if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 112 * 12544 + ((threadIdx.x + 1 * 256) % 112) / 28 * 112 + (threadIdx.x + 1 * 256) % 28];
        }

        blockLoadFilterStartIdx += 4;
        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
        }

        // Copy operands from shared buffer 1 into Operands Registers
        inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 0];
        inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 1];
        inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 2];
        inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 3];
        inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 4];

        inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 5];
        inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 6];
        inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 7];
        inputOperand9 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 8];
        inputOperand10 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 9];

        inputOperand11 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 10];
        inputOperand12 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 11];
        inputOperand13 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 12];
        inputOperand14 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 13];
        inputOperand15 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 14];

        inputOperand16 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 15];
        inputOperand17 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 16];
        inputOperand18 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 17];
        inputOperand19 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 18];
        inputOperand20 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 19];

        inputOperand21 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 20];
        inputOperand22 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 21];
        inputOperand23 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 22];
        inputOperand24 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 23];
        inputOperand25 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 24];

        inputOperand26 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 25];
        inputOperand27 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 26];
        inputOperand28 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 27];

        filterOperand1 = filterSharedBuffer1[(warpID % 1) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        input5filter1 += inputOperand5 * filterOperand1;

        input6filter1 += inputOperand6 * filterOperand1;

        input7filter1 += inputOperand7 * filterOperand1;

        input8filter1 += inputOperand8 * filterOperand1;

        input9filter1 += inputOperand9 * filterOperand1;

        input10filter1 += inputOperand10 * filterOperand1;

        input11filter1 += inputOperand11 * filterOperand1;

        input12filter1 += inputOperand12 * filterOperand1;

        input13filter1 += inputOperand13 * filterOperand1;

        input14filter1 += inputOperand14 * filterOperand1;

        input15filter1 += inputOperand15 * filterOperand1;

        input16filter1 += inputOperand16 * filterOperand1;

        input17filter1 += inputOperand17 * filterOperand1;

        input18filter1 += inputOperand18 * filterOperand1;

        input19filter1 += inputOperand19 * filterOperand1;

        input20filter1 += inputOperand20 * filterOperand1;

        input21filter1 += inputOperand21 * filterOperand1;

        input22filter1 += inputOperand22 * filterOperand1;

        input23filter1 += inputOperand23 * filterOperand1;

        input24filter1 += inputOperand24 * filterOperand1;

        input25filter1 += inputOperand25 * filterOperand1;

        input26filter1 += inputOperand26 * filterOperand1;

        input27filter1 += inputOperand27 * filterOperand1;

        input28filter1 += inputOperand28 * filterOperand1;

        // Copy Temp Registers to shared buffer 2
        inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
        if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
            inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
        }

        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();

        // Exchange shared buffer 1 and shared buffer 2 and repeat
        // load next group of Cnum channels
        blockLoadInputStartIdx += 112 * 112 * 4;
        inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 112 * 12544 + ((threadIdx.x + 0 * 256) % 112) / 28 * 112 + (threadIdx.x + 0 * 256) % 28];
        if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
            inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 112 * 12544 + ((threadIdx.x + 1 * 256) % 112) / 28 * 112 + (threadIdx.x + 1 * 256) % 28];
        }

        blockLoadFilterStartIdx += 4;
        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
        }

        // Copy operands from shared buffer 2 into Operands Registers
        inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 0];
        inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 1];
        inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 2];
        inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 3];
        inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 4];

        inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 5];
        inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 6];
        inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 7];
        inputOperand9 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 8];
        inputOperand10 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 9];

        inputOperand11 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 10];
        inputOperand12 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 11];
        inputOperand13 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 12];
        inputOperand14 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 13];
        inputOperand15 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 14];

        inputOperand16 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 15];
        inputOperand17 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 16];
        inputOperand18 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 17];
        inputOperand19 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 18];
        inputOperand20 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 19];

        inputOperand21 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 20];
        inputOperand22 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 21];
        inputOperand23 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 22];
        inputOperand24 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 23];
        inputOperand25 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 24];

        inputOperand26 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 25];
        inputOperand27 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 26];
        inputOperand28 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 27];

        filterOperand1 = filterSharedBuffer2[(warpID % 1) * 64 + laneID + 0 * 64];

        // Compute and Accumulate result in Result Registers
        input1filter1 += inputOperand1 * filterOperand1;

        input2filter1 += inputOperand2 * filterOperand1;

        input3filter1 += inputOperand3 * filterOperand1;

        input4filter1 += inputOperand4 * filterOperand1;

        input5filter1 += inputOperand5 * filterOperand1;

        input6filter1 += inputOperand6 * filterOperand1;

        input7filter1 += inputOperand7 * filterOperand1;

        input8filter1 += inputOperand8 * filterOperand1;

        input9filter1 += inputOperand9 * filterOperand1;

        input10filter1 += inputOperand10 * filterOperand1;

        input11filter1 += inputOperand11 * filterOperand1;

        input12filter1 += inputOperand12 * filterOperand1;

        input13filter1 += inputOperand13 * filterOperand1;

        input14filter1 += inputOperand14 * filterOperand1;

        input15filter1 += inputOperand15 * filterOperand1;

        input16filter1 += inputOperand16 * filterOperand1;

        input17filter1 += inputOperand17 * filterOperand1;

        input18filter1 += inputOperand18 * filterOperand1;

        input19filter1 += inputOperand19 * filterOperand1;

        input20filter1 += inputOperand20 * filterOperand1;

        input21filter1 += inputOperand21 * filterOperand1;

        input22filter1 += inputOperand22 * filterOperand1;

        input23filter1 += inputOperand23 * filterOperand1;

        input24filter1 += inputOperand24 * filterOperand1;

        input25filter1 += inputOperand25 * filterOperand1;

        input26filter1 += inputOperand26 * filterOperand1;

        input27filter1 += inputOperand27 * filterOperand1;

        input28filter1 += inputOperand28 * filterOperand1;

        // Copy Temp Registers to shared buffer 1
        inputSharedBuffer1[threadIdx.x + 0 * 256] = inputTemp1;
        if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
            inputSharedBuffer1[threadIdx.x + 1 * 256] = inputTemp2;
        }

        if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
            filterSharedBuffer1[threadIdx.x + 0 * 256] = filterTemp1;
        }
        __syncthreads();
    }
    // load next group of Cnum channels
    blockLoadInputStartIdx += 112 * 112 * 4;
    inputTemp1 = input[blockLoadInputStartIdx + (threadIdx.x + 0 * 256) / 112 * 12544 + ((threadIdx.x + 0 * 256) % 112) / 28 * 112 + (threadIdx.x + 0 * 256) % 28];
    if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
        inputTemp2 = input[blockLoadInputStartIdx + (threadIdx.x + 1 * 256) / 112 * 12544 + ((threadIdx.x + 1 * 256) % 112) / 28 * 112 + (threadIdx.x + 1 * 256) % 28];
    }

    blockLoadFilterStartIdx += 4;
    if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
        filterTemp1 = filter[blockLoadFilterStartIdx + ((threadIdx.x + 0 * 256) / 4) * 32 + ((threadIdx.x + 0 * 256) % 4)];
    }

    // Copy operands from shared buffer 1 into Operands Registers
    inputOperand1 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 0];
    inputOperand2 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 1];
    inputOperand3 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 2];
    inputOperand4 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 3];
    inputOperand5 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 4];

    inputOperand6 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 5];
    inputOperand7 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 6];
    inputOperand8 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 7];
    inputOperand9 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 8];
    inputOperand10 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 9];

    inputOperand11 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 10];
    inputOperand12 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 11];
    inputOperand13 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 12];
    inputOperand14 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 13];
    inputOperand15 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 14];

    inputOperand16 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 15];
    inputOperand17 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 16];
    inputOperand18 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 17];
    inputOperand19 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 18];
    inputOperand20 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 19];

    inputOperand21 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 20];
    inputOperand22 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 21];
    inputOperand23 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 22];
    inputOperand24 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 23];
    inputOperand25 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 24];

    inputOperand26 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 25];
    inputOperand27 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 26];
    inputOperand28 = inputSharedBuffer1[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 27];

    filterOperand1 = filterSharedBuffer1[(warpID % 1) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    input5filter1 += inputOperand5 * filterOperand1;

    input6filter1 += inputOperand6 * filterOperand1;

    input7filter1 += inputOperand7 * filterOperand1;

    input8filter1 += inputOperand8 * filterOperand1;

    input9filter1 += inputOperand9 * filterOperand1;

    input10filter1 += inputOperand10 * filterOperand1;

    input11filter1 += inputOperand11 * filterOperand1;

    input12filter1 += inputOperand12 * filterOperand1;

    input13filter1 += inputOperand13 * filterOperand1;

    input14filter1 += inputOperand14 * filterOperand1;

    input15filter1 += inputOperand15 * filterOperand1;

    input16filter1 += inputOperand16 * filterOperand1;

    input17filter1 += inputOperand17 * filterOperand1;

    input18filter1 += inputOperand18 * filterOperand1;

    input19filter1 += inputOperand19 * filterOperand1;

    input20filter1 += inputOperand20 * filterOperand1;

    input21filter1 += inputOperand21 * filterOperand1;

    input22filter1 += inputOperand22 * filterOperand1;

    input23filter1 += inputOperand23 * filterOperand1;

    input24filter1 += inputOperand24 * filterOperand1;

    input25filter1 += inputOperand25 * filterOperand1;

    input26filter1 += inputOperand26 * filterOperand1;

    input27filter1 += inputOperand27 * filterOperand1;

    input28filter1 += inputOperand28 * filterOperand1;

    // Copy Temp Registers to shared buffer 2
    inputSharedBuffer2[threadIdx.x + 0 * 256] = inputTemp1;
    if(threadIdx.x < 4 * 28 * 4 - 1 * 256) {
        inputSharedBuffer2[threadIdx.x + 1 * 256] = inputTemp2;
    }

    if(threadIdx.x < 1 * 16 * 4 - 0 * 256) {
        filterSharedBuffer2[threadIdx.x + 0 * 256] = filterTemp1;
    }
    __syncthreads();

    // Exchange shared buffer 1 and shared buffer 2 and repeat
    // Copy operands from shared buffer 2 into Operands Registers
    inputOperand1 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 0];
    inputOperand2 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 1];
    inputOperand3 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 2];
    inputOperand4 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 3];
    inputOperand5 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 4];

    inputOperand6 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 5];
    inputOperand7 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 6];
    inputOperand8 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 7];
    inputOperand9 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 8];
    inputOperand10 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 9];

    inputOperand11 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 10];
    inputOperand12 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 11];
    inputOperand13 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 12];
    inputOperand14 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 13];
    inputOperand15 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 14];

    inputOperand16 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 15];
    inputOperand17 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 16];
    inputOperand18 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 17];
    inputOperand19 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 18];
    inputOperand20 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 19];

    inputOperand21 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 20];
    inputOperand22 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 21];
    inputOperand23 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 22];
    inputOperand24 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 23];
    inputOperand25 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 24];

    inputOperand26 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 25];
    inputOperand27 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 26];
    inputOperand28 = inputSharedBuffer2[((warpID / 1) % 4) * 28 + (laneID % 4) * 28 * 4 + 27];

    filterOperand1 = filterSharedBuffer2[(warpID % 1) * 64 + laneID + 0 * 64];

    // Compute and Accumulate result in Result Registers
    input1filter1 += inputOperand1 * filterOperand1;

    input2filter1 += inputOperand2 * filterOperand1;

    input3filter1 += inputOperand3 * filterOperand1;

    input4filter1 += inputOperand4 * filterOperand1;

    input5filter1 += inputOperand5 * filterOperand1;

    input6filter1 += inputOperand6 * filterOperand1;

    input7filter1 += inputOperand7 * filterOperand1;

    input8filter1 += inputOperand8 * filterOperand1;

    input9filter1 += inputOperand9 * filterOperand1;

    input10filter1 += inputOperand10 * filterOperand1;

    input11filter1 += inputOperand11 * filterOperand1;

    input12filter1 += inputOperand12 * filterOperand1;

    input13filter1 += inputOperand13 * filterOperand1;

    input14filter1 += inputOperand14 * filterOperand1;

    input15filter1 += inputOperand15 * filterOperand1;

    input16filter1 += inputOperand16 * filterOperand1;

    input17filter1 += inputOperand17 * filterOperand1;

    input18filter1 += inputOperand18 * filterOperand1;

    input19filter1 += inputOperand19 * filterOperand1;

    input20filter1 += inputOperand20 * filterOperand1;

    input21filter1 += inputOperand21 * filterOperand1;

    input22filter1 += inputOperand22 * filterOperand1;

    input23filter1 += inputOperand23 * filterOperand1;

    input24filter1 += inputOperand24 * filterOperand1;

    input25filter1 += inputOperand25 * filterOperand1;

    input26filter1 += inputOperand26 * filterOperand1;

    input27filter1 += inputOperand27 * filterOperand1;

    input28filter1 += inputOperand28 * filterOperand1;

    __syncthreads();
    // For loop ends here

    // Parallel Reduction to accumulate result across threads
    // Cnum threads from one group
    #pragma unroll
    for (int offset = (4 >> 1); offset > 0; offset >>= 1) {
        input1filter1 += __shfl_down(input1filter1, offset, 4);

        input2filter1 += __shfl_down(input2filter1, offset, 4);

        input3filter1 += __shfl_down(input3filter1, offset, 4);

        input4filter1 += __shfl_down(input4filter1, offset, 4);

        input5filter1 += __shfl_down(input5filter1, offset, 4);

        input6filter1 += __shfl_down(input6filter1, offset, 4);

        input7filter1 += __shfl_down(input7filter1, offset, 4);

        input8filter1 += __shfl_down(input8filter1, offset, 4);

        input9filter1 += __shfl_down(input9filter1, offset, 4);

        input10filter1 += __shfl_down(input10filter1, offset, 4);

        input11filter1 += __shfl_down(input11filter1, offset, 4);

        input12filter1 += __shfl_down(input12filter1, offset, 4);

        input13filter1 += __shfl_down(input13filter1, offset, 4);

        input14filter1 += __shfl_down(input14filter1, offset, 4);

        input15filter1 += __shfl_down(input15filter1, offset, 4);

        input16filter1 += __shfl_down(input16filter1, offset, 4);

        input17filter1 += __shfl_down(input17filter1, offset, 4);

        input18filter1 += __shfl_down(input18filter1, offset, 4);

        input19filter1 += __shfl_down(input19filter1, offset, 4);

        input20filter1 += __shfl_down(input20filter1, offset, 4);

        input21filter1 += __shfl_down(input21filter1, offset, 4);

        input22filter1 += __shfl_down(input22filter1, offset, 4);

        input23filter1 += __shfl_down(input23filter1, offset, 4);

        input24filter1 += __shfl_down(input24filter1, offset, 4);

        input25filter1 += __shfl_down(input25filter1, offset, 4);

        input26filter1 += __shfl_down(input26filter1, offset, 4);

        input27filter1 += __shfl_down(input27filter1, offset, 4);

        input28filter1 += __shfl_down(input28filter1, offset, 4);
    }

    // Store output
    int blockWriteOutputStartIdx = blockIdx.x / 112 * 200704 + (blockIdx.x % 112) / 4 * 448 + (blockIdx.x % 4) / 1 * 28 + (blockIdx.x % 1) * 200704;

    if(laneID % 4 == 0) {
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 0] = input1filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 1] = input2filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 2] = input3filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 3] = input4filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 4] = input5filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 5] = input6filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 6] = input7filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 7] = input8filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 8] = input9filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 9] = input10filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 10] = input11filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 11] = input12filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 12] = input13filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 13] = input14filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 14] = input15filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 15] = input16filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 16] = input17filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 17] = input18filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 18] = input19filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 19] = input20filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 20] = input21filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 21] = input22filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 22] = input23filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 23] = input24filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 24] = input25filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 25] = input26filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 26] = input27filter1;
        output[blockWriteOutputStartIdx + 0 * 16 * outputHeight * outputWidth + (warpID / 1) / 1 * outputWidth + ((warpID / 1) % 1) * 28 + (warpID % 1) * 200704 + (laneID / 4) * outputHeight * outputWidth + 27] = input28filter1;
    }
}

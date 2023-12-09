#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

const int TEST_PER_PARAM = 1;

struct Param
{
    int inputChannel, inputWidth, outputChannel, inputBatchSize;
    int warpPerBlock, outputChannelPerWarp, outputWidthPerWarp;
    int channelGroupSize, horizontalRepeat, verticalRepeat;

    float minT = 1e18, avgT = 0, variance = 0, avgCudnnT = 0;
    bool correct = true;

    string ToCSV()
    {
        char tmp[200];
        sprintf(tmp, "%d,%d,%d,%d,", inputWidth, inputChannel, outputChannel, inputBatchSize);
        string result = string(tmp) + KernelParamToString() + ",";
        if (correct)
        {
            sprintf(tmp, "%.4f,%.4f,%.4f,%.5f", avgCudnnT, avgT, minT, variance);
            result += string(tmp);
        }
        return result;
    }

    string KernelParamToString()
    {
        char tmp[200];
        sprintf(tmp, "%d/%d/%d/%d/%d/%d", warpPerBlock, outputChannelPerWarp, outputWidthPerWarp, channelGroupSize, horizontalRepeat, verticalRepeat);
        return string(tmp);
    }
};
vector<vector<Param>> params;
int paramsCnt;

void GetParams();
void GenerateKernel(const Param &param);
void ModifyPointwiseCpp(const Param &param);
void Run(const Param &param);
void CleanUp();

int main()
{
    cout << "Generating Parameters" << endl;
    system("python ./HardwareUtilizationParameter.py");
    GetParams();

    system("g++ CodeGenerator.cpp -o CodeGenerator -O2");

    int cnt = 0;
    ofstream result("result.csv");
    ofstream best("best.csv");
    result << "inputWidth, inputChannel, outputChannel, batchSize, param(warpPerBlock/outputChannelPerWarp/outputWidthPerWarp/channelGroupSize/horizontalRepeat/VerticalRepeat), avgCudnnTime, avgKernelTime, minKernelTime, kernelTimeVariance" << endl;
    best << "inputWidth, inputChannel, outputChannel, batchSize, param(warpPerBlock/outputChannelPerWarp/outputWidthPerWarp/channelGroupSize/horizontalRepeat/VerticalRepeat), avgCudnnTime, avgKernelTime, minKernelTime, kernelTimeVariance" << endl;
    for (int i = 0; i < params.size(); i++)
    {
        Param bestParam;
        float bestParamAvgT = 1e18;
        for (int j = 0; j < params[i].size(); j++)
        {
            printf("=============%d / %d==============\n", ++cnt, paramsCnt);
            GenerateKernel(params[i][j]);
            ModifyPointwiseCpp(params[i][j]);
            system("cd ./build && make");

            float minT = 1e18, avgT = 0, avgCudnnT = 0, var = 0;
            for (int k = 1; k <= TEST_PER_PARAM && params[i][j].correct; k++)
            {
                printf("Running Test %d / %d\n", k, TEST_PER_PARAM);
                Run(params[i][j]);

                // 检查结果
                ifstream tmp("output.txt");
                if (!tmp.is_open())
                {
                    params[i][j].correct = false;
                    break;
                }
                float kernel = 0, cudnn = 0;
                tmp >> kernel >> cudnn;
                minT = min(minT, kernel);
                avgT += kernel;
                avgCudnnT += cudnn;
                var += kernel * kernel;
                tmp.close();
                system("rm output.txt");
            }
            if (params[i][j].correct)
            {
                avgT /= TEST_PER_PARAM;
                avgCudnnT /= TEST_PER_PARAM;
                var = var / TEST_PER_PARAM - avgT * avgT;
                params[i][j].minT = minT;
                params[i][j].avgT = avgT;
                params[i][j].avgCudnnT = avgCudnnT;
                params[i][j].variance = var;
                if (avgT < bestParamAvgT)
                {
                    bestParamAvgT = avgT;
                    bestParam = params[i][j];
                }
            }
            result << params[i][j].ToCSV() << endl;
        }

        if (bestParamAvgT < 1e17)
            best << bestParam.ToCSV() << endl;
    }

    result.close();
    best.close();
    CleanUp();
    return 0;
}

void GetParams()
{
    ifstream input("./parameters.txt");
    string line;
    vector<Param> paramGroup;
    while (getline(input, line))
    {
        if (line[0] == '=')
        {
            params.push_back(paramGroup);
            paramGroup.clear();
            continue;
        }
        Param param;
        sscanf(line.c_str(), "%d%d%d%d%d%d%d%d%d%d", &param.inputChannel, &param.inputWidth, &param.outputChannel,
               &param.inputBatchSize, &param.warpPerBlock, &param.outputChannelPerWarp, &param.outputWidthPerWarp,
               &param.channelGroupSize, &param.horizontalRepeat, &param.verticalRepeat);
        paramGroup.emplace_back(param);
        paramsCnt++;
    }
    input.close();
}

void GenerateKernel(const Param &param)
{
    string command = "./CodeGenerator ";
    command += to_string(param.inputChannel) + " " + to_string(param.inputWidth) + " " + to_string(param.outputChannel) + " ";
    command += to_string(param.inputBatchSize) + " " + to_string(param.warpPerBlock) + " " + to_string(param.outputChannelPerWarp) + " ";
    command += to_string(param.outputWidthPerWarp) + " " + to_string(param.channelGroupSize) + " ";
    command += to_string(param.horizontalRepeat) + " " + to_string(param.verticalRepeat) + " mykernel.h";
    system(command.c_str());
}

void ModifyPointwiseCpp(const Param &param)
{
    ifstream input("DCU_Pointwise_Kernel.original.cpp");
    ofstream output("DCU_Pointwise_Kernel.cpp");
    string line;
    while (getline(input, line))
    {
        if (line.find("int warpNumPerBlock = ") != string::npos)
            line = "    int warpNumPerBlock = " + to_string(param.warpPerBlock) + ";";
        else if (line.find("int outputWidthPerWarp = ") != string::npos)
            line = "    int outputWidthPerWarp = " + to_string(param.outputWidthPerWarp) + ";";
        else if (line.find("int outputHeightPerWarp = ") != string::npos)
            line = "    int outputHeightPerWarp = " + to_string(param.outputChannelPerWarp) + ";";
        else if (line.find("InputBatch_8_Input_112x112_InChannel_32_OutChannel_16") != string::npos)
        {
            char tmp[200];
            sprintf(tmp, "        InputBatch_%d_Input_%dx%d_InChannel_%d_OutChannel_%d<<<gridSize, blockSize>>>(deviceInput, deviceFilter, deviceKernelOutput,",
                    param.inputBatchSize, param.inputWidth, param.inputWidth, param.inputChannel, param.outputChannel);
            line = string(tmp);
        }
        output << line << endl;
    }
    input.close();
    output.close();
}

void Run(const Param &param)
{
    /*
        Arguments:
        1. Input Batch Number
        2. Input Channel
        3. Input Height
        4. Output Channel
    */
    char command[200];
    sprintf(command, "./build/kernel %d %d %d %d", param.inputBatchSize, param.inputChannel, param.inputWidth, param.outputChannel);
    system(command);
}

void CleanUp()
{
}
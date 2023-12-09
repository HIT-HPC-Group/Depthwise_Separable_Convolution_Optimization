# 面向DCU的深度分离卷积优化项目

# 软件简介
    深度分离卷积（Depthwise Separable Convolution）将常见的卷积运算分解为”深度卷积“（Depthwise Convolution）与“点卷积”（Pointwise Convolution）以减少计算开销。这种方法已应用于多种深度学习神经网络模型之中，比如 MobileNet 系列和 EfficientNet 系列。它们在“大规模视觉识别挑战赛”（Large-Scale Recognition Competition，LSVRC）中取得了优异成绩，因此证明了深度分离卷积的训练有效性。
    该项目针对 MobileNet V2 和 EfficientNet B0 中的深度分离卷积层进行优化，并将优化后的卷积核函数（Kernel）通过PyTorch拓展（Extension）的方式封装，进而在PyTorch中调用。


# 源码（source）目录结构
        - source
        - Depthwise
	    - Kernel
            - Extension
                - DCU_Depthwise_Extension
                - Test_DCU_Depthwise_Extension
            - V100S_CUDNN
            - V100S_Depthwise_PyTorch
        - Pointwise
	    - Kernel
            - Extension
                - DCU_Pointwise_Extension
                - Test_DCU_Pointwise_Extension
            - V100S_CUDNN
            - V100S_Pointwise_PyTorch
            - AutomaticPointwiseHardwareSelector
        - Models
            - data
            - EfficientNetB0
            - MobileNetV2
            - V100S_EfficientNetB0
            - V100S_MobileNetV2

    详细展开每个文件夹后，可以看到以下文件：
    - source
        - Depthwise
	    - Kernel (该文件夹也用于测试)
	        - .h文件: 每个文件分别包含了不同的Depthwise卷积核函数
                - DCU_Depthwise_Kernel.cpp: 测试一个核函数的正确性、与MIOpen对比性能
                - TestDepthwiseKernel.py: 测试所有核函数，并记录测试结果
		- CMakeLists.txt: cmake编译文件
                - DCU_Depthwise_Kernel_Result.csv: 保存测试结果

	    - Extension
	        - DCU_Depthwise_Extension
		    - .h文件：每个文件分别包含了不同的Depthwise卷积核函数
		    - 有_hip后缀的文件：安装拓展时由hipify自动生成
                    - DCU_Depthwise_Kernel.hip：调用核函数的函数
		    - DCU_Depthwise.cpp: C++封装	调用核函数的函数
                    - setup.py: 搭建、编译、安装名为optimizedDepthwise的Pytorch拓展（已安装于容器中）

                - Test_DCU_Depthwise_Extension (该文件夹也用于测试)
		    - DepthwiseLayer.py: 使用Python封装的Depthwise卷积层，底层调用了Depthwise拓展
                    - OriginalLayer.py：使用Python封装的Depthwise卷积层，底层调用了PyTorch的nn.Conv2d卷积
		    - DCU_TestOptimizedDepthwiseExtension.py: 测试DepthwiseLayer的前向传递性能，与OriginalLayer对比
		    - DCU_TestOptimizedDephwiseExtension_ForwardBackward.py:测试DepthwiseLayer的前向传递+后向传递的性能，与OriginalLayer对比
		    - DCU_Depthwise_Extension_Result.csv：保存DepthwiseLayer与OriginalLayer的前向传递性能
		    - DCU_Depthwise_Extension_ForwardBackward_Result.csv: 保存DepthwiseLayer与OriginalLayer的前向传递+后向传递性能
            
                - V100S_CUDNN（该文件夹用于工作证明与备份）在NVidia Tesla V100S上，调用cuDNN，测试各个卷积层的性能并记录结果
                - V100S_Depthwise_PyTorch（该文件夹用于工作证明与备份）在NVidia Tesla V100S上，调用PyTorch的nn.Conv2d，测试各个卷积层的性能并记录结果

	- Pointwise
	    - Kernel (该文件夹也用于测试)
	        - .h文件: 每个文件分别包含了不同的Pointwise卷积核函数
                - DCU_Pointwise_Kernel.cpp: 测试一个核函数的正确性、与MIOpen对比性能
                - TestPointwiseKernel.py: 测试所有核函数，并记录测试结果
		- CMakeLists.txt: cmake编译文件
                - DCU_Pointwise_Kernel_Result.csv: 保存测试结果

	    - Extension
	        - DCU_Pointwise_Extension
		    - .h文件：每个文件分别包含了不同的Pointwise卷积核函数
		    - 有_hip后缀的文件：安装拓展时由hipify自动生成
                    - DCU_Pointwise_Kernel.hip：实际调用底层核函数
		    - DCU_Pointwise.cpp: 封装调用核函数的函数
		    - setup.py: 搭建、编译、安装名为optimizedPointwise的Pytorch拓展（已安装于容器中）

                - Test_DCU_Pointwise_Extension (该文件夹也用于测试)
		    - PointwiseLayer.py: 使用Python封装的Pointwise卷积层，底层调用了Pointwise拓展
                    - OriginalLayer.py：使用Python封装的Pointwise卷积层，底层调用了PyTorch的nn.Conv2d卷积
		    - DCU_TestOptimizedPointwiseExtension.py: 测试PointwiseLayer的前向传递性能，与OriginalLayer对比
		    - DCU_TestOptimizedPointwiseExtension_ForwardBackward.py:测试PointwiseLayer的前向传递+后向传递的性能，与OriginalLayer对比
		    - DCU_Pointwise_Extension_Result.csv：记录PointwiseLayer与OriginalLayer的前向传递性能
		    - DCU_Pointwise_Extension_ForwardBackward_Result.csv: 记录PointwiseLayer与OriginalLayer的前向传递+后向传递性能
            
                - V100S_CUDNN（该文件夹用于工作证明与备份): 在NVidia Tesla V100S上，调用cuDNN,测试各个卷积层的性能并记录结果
                - V100S_Pointwise_PyTorch（该文件夹用于工作证明与备份）: 在NVidia Tesla V100S上，调用PyTorch的nn.Conv2d测试各个卷积层的性能并记录结果
	        - AutomaticPointwiseHardwareSelector（该文件夹用于工作证明与备份）: 实现Pointwise卷积时，需要考虑硬件资源划分并从多个选项中找到最优划分方案。用代码自动生成大量选项，然后自动生成不同选项对应的核函数，并自动测试核函数，找到最佳选项。

	  - Models
	    - data: cifar10数据集。该数据集包含50000个训练数据和10000个测试数据以及对应的分类，共10类。每个数据均为32x32x3的图片。
	    - EfficientNetB0 (该文件夹也用于测试)
		- DepthwiseLayer.py: 使用Python封装的Depthwise卷积层，底层调用了Depthwise拓展
		- PointwiseLayer.py: 使用Python封装的Pointwise卷积层，底层调用了Pointwise拓展
		- StructureOriginalEfficientNetB0.py: 查看PyTorch的EfficientNet B0模型
		- ModifiedEfficientNetB0Model.py：修改PyTorch的EfficientNet B0模型，将其中的深度卷积层和点卷积层替换为DepthwiseLayer和PointwiseLayer
		- DCU_TestModifiedEfficientNetB0_Forward.py: 测试修改版EfficientNet B0模型的前向传递性能
		- DCU_TestModifiedEfficientNetB0.py: 测试修改版EfficientNet B0模型的前向传递+后向传递性能
		- 各个csv与txt文件用于记录测试结果和输出信息

            - MobileNetV2 (该文件夹也用于测试)
		- DepthwiseLayer.py: 使用Python封装的Depthwise卷积层，底层调用了Depthwise拓展
		- PointwiseLayer.py: 使用Python封装的Pointwise卷积层，底层调用了Pointwise拓展
		- StructureOriginalMobileNetV2.py: 查看PyTorch的MobileNet V2模型
		- ModifiedMobileNetV2Model.py：修改PyTorch的MobileNetV2模型，将其中的深度卷积层和点卷积层替换为DepthwiseLayer和PointwiseLayer
		- DCU_TestModifiedMobileNetV2_Forward.py: 测试修改版MobileNet V2模型的前向传递性能
		- DCU_TestModifiedMobileNetV2.py: 测试修改版MobileNet V2模型的给前向传递+后向传递性能
		- 各个csv与txt文件用于记录测试结果和输出信息

	    - V100S_EfficientNetB0（该文件夹用于工作证明与备份)：在NVidia Tesla V100S上，调用PyTorch，测试EfficientNet B0的性能并记录结果
            - V100S_MobileNetV2（该文件夹用于工作证明与备份)：在NVidia Tesla V100S上，调用PyTorch，测试MobileNet V2的性能并记录结果


# 软件编译安装流程（含安装成功标识。如有其它环境工具，应说明环境变量加载指令env.sh）
    1. 启动容器并通过SSH连接后，请使用source env.sh指令加载环境变量。env.sh只包含一行：export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
    2. 容器中已经安装 optimizedDepthwise 和 optimizedPointwise，请使用 pip list指令查看确认


# 算例介绍
##算例一：测试Depthwise Kernel的正确性与性能

  算例名称及简介：
      - 名称：Test_DCU_Depthwise_Kernel
      - 简介：该算例使用随机生成的数据（小算例）测试每个Depthwise卷积核函数的正确性与性能，并与MIOpen对比。

  运行指令：
      1. cd 进入examples/Test_DCU_Depthwise_Kernel文件夹
      2. 使用以下指令编译代码并生成可执行文件“kernel”：
          - mkdir build
          - cd build # 进入build文件夹
          - cmake ..
          - make
      3. cd 退回到Test_DCU_Depthwise_Kernel文件夹
      4. 使用 python TestDepthwiseKernel.py 指令运行测试脚本
      注意：
          - 在容器中运行测试时，CMakeList.txt 第22行应为 SET CMAKE_CXX_COMPILER "/opt/dtk-23.04.1/bin/hipcc")
	  - TestDepthwiseKernel.py第31行，可以设置每个kernel的运行次数，最后计算平均时间

  运行结果：
      - 测试脚本在运行过程中，会打印输出以下信息：
	Calculating Input Batch: xxx, Input Channel: xxx, Input Height: xxx, Filter Height:xxx, Stride: xxx for xxx time.
	Kernel Calculation Correct.
      - 测试结束后，结果自动保存在 DCU_Depthwise_Kernel_Result.csv 中

  正确性一致说明：
      各个Kernel的计算结果都会与MIOpen的计算结果对比，如果误差范围为0.001，则判定为计算正确，并且打印输出Kernel Calculation Correct

  性能说明：
      a.DCU单卡比CPU单核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU单核心加速比
      b.DCU单卡比CPU 32核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU 32核心加速比
      c.DCU多卡/单卡 加速比及并行效率：该项目主要目标为优化底层卷积运算，而非优化模型并行方法，因此没有额外测试多卡加速比。后续其他项目可以尝试优化模型并行、数据并行、流水线并行等方法。
      d.与成熟软件对比，正确性和性能说明：
          d.1. 所有kernel都与MIOpen对比计算结果，测试发现结果都在极小的误差范围内（由于float类型导致的误差）
	  d.2. 所有kernel都与MIOpen和cuDNN对比性能
                        - 与MIOpen的性能对比结果保存在 Test_DCU_Depthwise_Kernel/DCU_Depthwise_Kernel_Result.csv中
                        - cuDNN的性能测试结果保存在 source/Depthwise/V100S_CUDNN/V100S_Depthwise_CUDNN_Result.csv中
	  d.3. 测试主要对比了我们的Kernel与MIOpen的性能，数据批尺寸（Batch Size）为1、8、16、32、64、128：
	    - 当Batch Size = 1时, 我们的Kernel比MIOpen平均快了60%
            - 当Batch Size = 8时, 我们的Kernel比MIOpen平均快了68%
	    - 当Batch Size = 16时, 我们的Kernel比MIOpen平均快了70%
	    - 当Batch Size = 32时, 我们的Kernel比MIOpen平均快了73%
	    - 当Batch Size = 64时, 我们的Kernel比MIOpen平均快了71%
	    - 当Batch Size = 128时, 我们的Kernel比MIOpen平均快了76%
	
##算例二: 测试Depthwise Extension的正确性与性能

  算例名称及简介：
      - 名称：Test_DCU_Depthwise_Extension
      - 简介：该算例使用随机生成的数据（小算例）测试每个Depthwise拓展的正确性与性能，并与PyTorch的nn.Conv2d对比

  运行指令：
      1. 通过pip list查看环境中是否已有optimizedDepthwise库。如果没有，请前往source/Depthwise/Extension/DCU_Depthwise_Extension文件夹，使用 sudo python setup.py install命令安装。等待编译安装后，输出信息的结尾会显示installed...optimizedDepthwise==1.0...的信息
      2. cd 进入examples/Test_DCU_Depthwise_Extension文件夹
      3. 使用 python DCU_TestOptimizedDepthwiseExtension.py 指令运行测试脚本，该脚本测试了Depthwise拓展与PyTorch的nn.Conv2d的前向传递性能
      4. 使用 python DCU_TestOptimizedDepthwiseExtension_ForwardBackward.py 指令运行测试脚本，该脚本测试了Depthwise拓展与PyTorch的nn.Conv2d的前向传递+后向传递性能

  运行结果：
      - 测试脚本在运行过程中，会打印输出Depthwise拓展与nn.Conv2d的前向传递和后向传递的时间
      - 测试结束后的结果，分别保存在 DCU_Depthwise_Extension_Result.csv 和 DCU_Depthwise_Extension_ForwardBackward_Result.csv中

  正确性一致说明：
      Depthwise拓展封装了之前的Depthwise Kernel，因此，Kernel的计算正确性保证了拓展的正确性。另外，后向传递部分直接调用了PyTorch库，因此可以保证计算正确性。

  性能说明：
      a.DCU单卡比CPU单核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU单核心加速比
      b.DCU单卡比CPU 32核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU 32核心加速比
      c.DCU多卡/单卡 加速比及并行效率：由于该项目的目标为优化底层卷积运算，而非优化模型并行训练的效率，因此没有额外测试多卡加速比。后续其他项目可以尝试优化模型并行、数据并行、流水线并行等方法。
      d.与成熟软件对比，正确性和性能说明：
          d.1. 所有Depthwise拓展都与DCU PyTorch和V100S PyTorch对比性能。
	      - 与DCU PyTorch的性能对比结果保存在 DCU_Depthwise_Extension_Result.csv 和 DCU_Depthwise_Extension_ForwardBackward_Result.csv 中。
	      - V100S PyTorch的性能结果保存在source/Depthwise/V100S_Depthwise_PyTorch/V100S_Depthwise_PyTorch_Result.csv 和 source/Depthwise/V100S_Depthwise_PyTorch/V100S_Depthwise_PyTorch_ForwardBackward_Result.csv中
	  d.2. 对于 前向传递 测试，数据批尺寸（Batch Size）为1、8、16、32、64、128的情况：
	      - 当Batch Size = 1时, 我们的Depthwise拓展比DCU PyTorch平均快了39%
              - 当Batch Size = 8时, 我们的Depthwise拓展比DCU PyTorch平均快了40%
	      - 当Batch Size = 16时,我们的Depthwise拓展比DCU PyTorch平均快了43%
	      - 当Batch Size = 32时,我们的Depthwise拓展比DCU PyTorch平均快了47%
	      - 当Batch Size = 64时,我们的Depthwise拓展比DCU PyTorch平均快了41%
	      - 当Batch Size = 128时,我们的Depthwise拓展比DCU PyTorch平均快了39%
	  d.3. 因为后向传递直接调用了PyTorch库，与原有的nn.Conv2d基本一致，在此不额外讨论
	  d.4. 我们通过PTorch Profiler分析了PyTorch拓展的调用逻辑。我们发现：
              - PyTorch拓展引入了额外的开销（比如为数据分配空间、额外的aten.empty()、aten._zero()操作等）。因此，与直接测试kernel相比，PyTorch拓展的加速比有所下降。后续其他项目可以尝试将Depthwise Kernel直接集成到PyTorch库中，减少额外开销。
	      - 该项目不针对后向传递进行优化，但实际上，后向传递占据了非常多的时间（记录在DCU_Depthwise_Extension_ForwardBackward_Result.csv中）。后续其他项目可以进一步优化后向传递计算，可能会进一步提升性能。

##算例三：测试Pointwise Kernel的正确性与性能

  算例名称及简介：
      - 名称：Test_DCU_Pointwise_Kernel
      - 简介：该算例使用随机生成的数据（小算例）测试每个Pointwise Kernel的正确性与性能，并与MIOpen对比。

  运行指令：
      1. cd 进入examples/Test_DCU_Pointwise_Kernel文件夹
      2. 使用以下指令编译代码并生成可执行文件“kernel”：
             - mkdir build
             - cd build # 进入build文件夹
             - cmake ..
             - make
      3. cd 退回到Test_DCU_Pointwise_Kernel文件夹
      4. 使用 python TestPointwiseKernel.py 指令运行测试脚本
      注意：
          - 在容器中测试时，CMakeList.txt 第22行 为 SET CMAKE_CXX_COMPILER "/opt/dtk-23.04.1/bin/hipcc")
	  - TestPointwiseKernel.py 第45行，可以设置每个kernel的运行次数，最后计算平均时间

  运行结果：
      - 测试脚本在运行中，会打印输出以下信息：
	Calculating Input Batch: xxx, Input Channel: xxx, Input Height: xxx, Output Channel: xxx for xxx time.
	Kernel Calculation Correct.
      - 测试结束后，结果保存在 DCU_Pointwise_Kernel_Result.csv 中

  正确性一致说明：
      各个Kernel的计算结果都会与MIOpen的计算结果对比，如果误差范围为0.001，则判定为计算结果正确，并且打印输出Kernel Calculation Correct

  性能说明：
      a.DCU单卡比CPU单核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU单核心加速比
      b.DCU单卡比CPU 32核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU 32核心加速比
      c.DCU多卡/单卡 加速比及并行效率：由于该项目的目标为优化底层卷积运算，而非优化模型并行训练的效率，因此没有额外测试多卡加速比。后续其他项目可以尝试优化模型并行、数据并行、流水线并行等方法。
      d.与成熟软件对比，正确性和性能说明：
          d.1. 所有kernel都与MIOpen对比计算结果，测试发现结果都在极小的误差范围内（由于float类型导致的误差）
	  d.2. 所有kernel都与MIOpen和cuDNN对比了性能。与MIOpen的性能对比结果保存在 Test_DCU_Pointwise_Kernel/DCU_Pointwise_Kernel_Result.csv中，而cuDNN的性能结果保存在source/Pointwise/V100S_CUDNN/V100S_Pointwise_CUDNN_Result.csv中
	  d.3. 测试了数据批尺寸（Batch Size）为1、8、16、32、64、128的情况：
              - 当Batch Size = 1时, 我们的Kernel比MIOpen平均快了70%
	      - 当Batch Size = 8时, 我们的Kernel比MIOpen平均快了46%
	      - 当Batch Size = 16时, 我们的Kernel比MIOpen平均快了24%
	      - 当Batch Size = 32时, 我们的Kernel比MIOpen在最好情况下快了78%，平均快了7%
	      - 当Batch Size = 64时, 我们的Kernel比MIOpen在最好情况下快了59%，但平均快了-21%
	      - 当Batch Size = 128时, 我们的Kernel比MIOpen在最好情况下快了47%，但平均快了-47%
	  d.4. 对于pointwise卷积，在batch size较小时，优化提升效果比较明显。
	
##算例四: 测试Pointwise拓展的正确性与性能

  算例名称及简介：
      - 名称：Test_DCU_Pointwise_Extension
      - 简介：该算例使用随机生成的数据（小算例）测试每个Pointwise拓展的正确性与性能，并与PyTorch的nn.Conv2d对比。

  运行指令：
      1. 通过pip list查看环境中是否已有optimizedPointwise库。如果没有，请前往source/Pointwise/Extension/DCU_Pointwise_Extension文件夹，使用 sudo python setup.py install命令安装。等待编译安装后，输出信息的结尾会显示installed optimizedPointwise == 1.0信息
      2. cd 进入examples/Test_DCU_Pointwise_Extension文件夹
      3. 使用 python DCU_TestOptimizedPointwiseExtension.py 指令运行测试脚本，该脚本测试了Pointwise拓展与PyTorch的nn.Conv2d的前向传递性能
      4. 使用 python DCU_TestOptimizedPointwiseExtension_ForwardBackward.py 指令运行测试脚本，该脚本测试了Pointwise拓展与PyTorch的nn.Conv2d的前向传递+后向传递性能

  运行结果：
      - 测试脚本在运行过程中，会打印输出Pointwise拓展与nn.Conv2d的前向传递和后向传递的时间
      - 测试结束后的结果，自动保存在 DCU_Pointwise_Extension_Result.csv 和 DCU_Pointwise_Extension_ForwardBackward_Result.csv中

  正确性一致说明：
      Pointwise拓展封装了之前的Pointwise Kernel。因此，Kernel的计算正确性保证了拓展的正确性。另外，后向传递部分直接调用了PyTorch库，因此可以保证计算正确性。

  性能说明：
      a.DCU单卡比CPU单核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU单核心加速比
      b.DCU单卡比CPU 32核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU 32核心加速比
      c.DCU多卡/单卡 加速比及并行效率：由于该项目的目标为优化底层卷积运算，而非优化模型并行训练的效率，因此没有额外测试多卡加速比。后续其他项目可以尝试优化模型并行、数据并行、流水线并行等方法。
      d.与成熟软件对比，正确性和性能说明：
          d.1. 所有Pointwise拓展都与DCU PyTorch和V100S PyTorch对比性能。
	      - 与DCU PyTorch的性能对比结果保存在 Test_DCU_Pointwise_Kernel/DCU_Pointwise_Extension_Result.csv 和 Test_DCU_Pointwise_Kernel/DCU_Pointwise_Extension_ForwardBackward_Result.csv 中。
	      - V100S PyTorch的性能结果保存在source/Pointwise/V100S_Pointwise_PyTorch/V100S_Pointwise_PyTorch_Result.csv 和 source/Pointwise/V100S_Pointwise_PyTorch/V100S_Pointwise_PyTorch_ForwardBackward_Result.csv中
	  d.2. 对于 前向传递 测试，数据批尺寸（Batch Size）为1、8、16、32、64、128的情况：
	      - 当Batch Size = 1时, 我们的Pointwise拓展比DCU PyTorch平均快了46%
              - 当Batch Size = 8时, 我们的Pointwise拓展比DCU PyTorch在最好情况下快了48%，平均快了28%
	      - 当Batch Size = 16时,我们的Pointwise拓展比DCU PyTorch在最好情况下快了46%，平均快了12%
	      - 当Batch Size = 32时,我们的Pointwise拓展比DCU PyTorch在最好情况下快了31%，但平均快了-10%
	      - 当Batch Size = 64时,我们的Pointwise拓展比DCU PyTorch在最好情况下快了16%，但平均快了-30%
	      - 当Batch Size = 128时,我们的Pointwise拓展比DCU PyTorch在最好情况下快了4%，但平均快了-63%
	  d.3. 由于PyTorch拓展引入了额外的开销（比如为数据分配空间、额外的aten.empty()、aten._zero()操作等），因此与直接测试kernel相比，PyTorch拓展的加速比有所下降。后续其他项目可以尝试将Pointwise Kernel直接集成到PyTorch库中，减少额外开销。
	  d.4. 该项目不针对训练中的后向传递进行优化，但实际上，后向传递占据了非常多的时间（记录在DCU_Pointwise_Extension_ForwardBackward_Result.csv中）。后续其他项目可以进一步优化后向传递计算，可能会进一步提升性能。

##算例五: 测试修改版MobileNet V2的性能 

  算例名称及简介：
      - 名称：Test_Models-MobileNetV2
      - 简介：该算例使用cifar10数据集，测试修改版MobileNet V2训练推理的性能，并与PyTorch的MobileNet V2对比。还使用了随机生成的数据，测试修改版MobileNet V2前向传递性能，并与PyTorch的MobileNet V2对比。

  运行指令：
      1. 通过pip list查看环境中是否已有optimizedDepthwise和optimizedPointwise库。
          - 如果没有optimizedDepthwise，请前往source/Depthwise/Extension/DCU_Depthwise_Extension文件夹，使用 sudo python setup.py install命令安装。等待编译安装后，输出信息的结尾会显示installed optimizedDepthwise == 1.0信息
          - 如果没有optimizedPointwise，请前往source/Pointwise/Extension/DCU_Pointwise_Extension文件夹，使用 sudo python setup.py install命令安装。等待编译安装后，输出信息的结尾会显示installed optimizedPointwise == 1.0信息
      2. cd 进入examples/Test_Models/MobileNetV2文件夹
      3. 使用 python DCU_TestModifiedMobileNetV2_Forward.py 指令运行测试脚本，该脚本测试了修改版MobileNet V2与PyTorch的MobileNet V2的前向传递性能
      4. 使用 python DCU_TestModifiedMobileNetV2.py 指令运行测试脚本，该脚本测试了修改版MobileNet V2与PyTorch的MobileNet V2的训练+推理性能

  运行结果：
      - DCU_TestModifiedMobileNetV2_Forward.py 在运行过程中，会打印两种MobileNet V2前向传递的时间。结果保存在DCU_ModifiedMobileNetV2_Forward_Result.csv
      - DCU_TestModifiedMobileNetV2.py 在运行过程中，会打印两种MobileNet V2的训练、推理的开始与结束，以及训练、推理的时间。结果保存在 DCU_ModifiedMobileNetV2_Result_Batchxxx_Epochyyy.csv 的文件中

  正确性一致说明：
      修改版MobileNet V2是基于PyTorch的MobileNet V2的。替换进去的Depthwise Layer和Pointwise Layer都是用之前测试正确的Extension和Kernel搭建。另外，训练过程中，打印信息显示，模型推理准确率提升，也证明模型正确训练。

  性能说明：
      a.DCU单卡比CPU单核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU单核心加速比
      b.DCU单卡比CPU 32核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU 32核心加速比
      c.DCU多卡/单卡 加速比及并行效率：由于该项目的目标为优化底层卷积运算，而非优化模型并行训练的效率，因此没有额外测试多卡加速比。后续其他项目可以尝试优化模型并行、数据并行、流水线并行等方法。
      d.与成熟软件对比，正确性和性能说明：
          d.1. DCU上的修改版MobileNet V2与DCU PyTorch的MobileNet V2和 V100S PyTorch的MobileNet V2对比
              - 与DCU PyTorch的性能对比结果保存在DCU_ModifiedMobileNetV2_Forward_Result.csv 和DCU_ModifiedMobileNetV2_Result_Batchxxx_Epochyyy.csv 中
	      - V100S PyTorch的测试结果保存在 source/Models/V100S_MobileNetV2/V100S_MobileNetV2_Forward_Result.csv 和 source/Models/V100S_MobileNetV2/V100S_MobileNetV2_Result_Batchxxx_Epochyyy.csv 中
	  d.2. 对于 前向传递 测试，数据批尺寸（Batch Size）为1、8、16、32、64、128的情况：
	      - 当Batch Size = 1时, 修改版MobileNet V2比原有模型平均快了26%，速度达到了V100S上的98%
              - 当Batch Size = 8时, 修改版MobileNet V2比原有模型平均快了22%，速度达到了V100S上的65%
	      - 当Batch Size = 16时,修改版MobileNet V2比原有模型平均快了-8%，速度达到了V100S上的46%
	      - 当Batch Size = 32时,修改版MobileNet V2比原有模型平均快了-11%，速度达到了V100S上的45%
	      - 当Batch Size = 64时,修改版MobileNet V2比原有模型平均快了-13%，速度达到了V100S上的47%
	      - 当Batch Size = 128时,修改版MobileNet V2比原有模型平均快了-15%，速度达到了V100S上的48%
	  d.3. 由于完整训练模型需要花费20多个小时，因此，我们仅测量了训练3个epoch花费的时间。在 DCU_TestModifiedMobileNetV2.py 第24行，可以修改EPOCHS数量，如果修改为160左右，可完整训练模型。
	      - 当Batch Size = 8时，修改版MobileNet V2比原有模型，训练速度快了-4.5%，推理速度快了29%。训练速度达到了V100S的43%，推理速度达到了V100S的61%。DCU已有模型训练速度到达了V100S的44%，推理速度达到V100S的43%
	      - 当Batch Size = 16时，修改版MobileNet V2比原有模型，训练速度快了0.6，推理速度快了8%。训练速度达到了V100S的63%，推理速度达到了V100S的73%。DCU已有模型训练速度到达了V100S的63%，推理速度达到V100S的67%
	      - 当Batch Size = 32时，修改版MobileNet V2比原有模型，训练速度快了-3.8%，推理速度快了-12%。训练速度达到了V100S的66%，推理速度达到了V100S的67%。DCU已有模型训练速度到达了V100S的68%，推理速度达到V100S的75%
	      - 当Batch Size = 64时，修改版MobileNet V2比原有模型，训练速度快了-4.7%，推理速度快了-14%。训练速度达到了V100S的67%，推理速度达到了V100S的70%。DCU已有模型训练速度到达了V100S的70%，推理速度达到V100S的80%
	      - 当Batch Size = 128时，修改版MobileNet V2比原有模型，训练速度快了-5.3%，推理速度快了-12%。训练速度达到了V100S的72%，推理速度达到了V100S的71%。DCU已有模型训练速度到达了V100S的76%，推理速度达到V100S的80%
	  d.4. 之前的测试证明，我们的核函数在某些情况下是有优化效果的。但是有几个重要的原因使得模型整体优化效果不佳：
	      - 训练和推理过程中的数据移动时间（cpu -> dcu，dcu -> cpu）。这是模型训练和推理性能的重要瓶颈，也依赖于硬件的提升
	      - 项目不针对后向传递和模型中的其他层（池化层、Batch Normalization层、全连接层等）进行优化。它们也占据了很多时间
	      - PyTorch Extension引入了一些额外的开销

##算例六: 测试修改版EfficientNet B0的性能 

  算例名称及简介：
      - 名称：Test_Models-EfficientNetB0
      - 简介：该算例使用cifar10数据集，测试修改版EfficientNet B0的训练推理性能，并与PyTorch的EfficientNet B0对比。还使用了随机生成的数据，测试修改版EfficientNet B0的前向传递性能，并与PyTorch的EfficientNet B0对比。

  运行指令：
      1. 通过pip list查看环境中是否已有optimizedDepthwise和optimizedPointwise库。
          - 如果没有optimizedDepthwise，请前往source/Depthwise/Extension/DCU_Depthwise_Extension文件夹，使用 sudo python setup.py install命令安装。等待编译安装后，输出信息的结尾会显示installed optimizedDepthwise == 1.0信息
          - 如果没有optimizedPointwise，请前往source/Pointwise/Extension/DCU_Pointwise_Extension文件夹，使用 sudo python setup.py install命令安装。等待编译安装后，输出信息的结尾会显示installed optimizedPointwise == 1.0信息
      2. cd 进入examples/Test_Models/EfficientNetB0文件夹
      3. 使用 python DCU_TestModifiedEfficientNetB0_Forward.py 指令运行测试脚本，该脚本测试了修改版EfficientNet B0与PyTorch的EfficientNet B0的前向传递性能
      4. 使用 python DCU_TestModifiedEfficientNetB0.py 指令运行测试脚本，该脚本测试了修改版EfficientNet B0与PyTorch的EfficientNet B0的训练推理性能

  运行结果：
      - DCU_TestModifiedEfficientNetB0_Forward.py 在运行过程中，会打印两种EfficientNet B0前向传递的时间，结果保存在DCU_ModifiedEfficientNetB0_Forward_Result.csv
      - DCU_TestModifiedEfficientNetB0.py 在运行过程中，会打印两种EfficientNet B0的训练、推理的开始与结束，以及训练、推理的时间。结果保存在 DCU_ModifiedEfficientNetB0_Result_Batchxxx_Epochyyy.csv 的文件中

  正确性一致说明：
      修改版EfficientNet B0是基于PyTorch的EfficientNet B0的。替换进去的Depthwise Layer和Pointwise Layer都是用之前测试正确的Extension、Kernel搭建。另外，训练过程中，打印信息显示，模型推理准确率提升，也证明模型正确训练。

  性能说明：
      a.DCU单卡比CPU单核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU单核心加速比
      b.DCU单卡比CPU 32核心加速比：该项目主要是与ROCm平台的MIOpen库与CUDA平台的cuDNN库对比，它们都是基于加速卡（DCU或GPU）的并行计算库，因此我们没有额外测试CPU 32核心加速比
      c.DCU多卡/单卡 加速比及并行效率：由于该项目的目标为优化底层卷积运算，而非优化模型并行训练的效率，因此没有额外测试多卡加速比。后续其他项目可以尝试优化模型并行、数据并行、流水线并行等方法。
      d.与成熟软件对比，正确性和性能说明：
          d.1. DCU上的修改版EfficientNet B0 与 DCU PyTorch的EfficientNet B0 和 V100S PyTorch的EfficientNet B0对比
              - 与DCU PyTorch的性能对比结果保存在 Test_Models/EfficientNetB0/DCU_ModifiedEfficientNetB0_Forward_Result.csv 和 Test_Models/EfficientNetB0/DCU_ModifiedEfficientNetB0_Result_Batchxxx_Epochyyy.csv 中
	      - V100S PyTorch的性能结果保存在 source/Models/V00S_EfficientNetB0/V100S_EfficientNetB0_Forward_Result.csv 和 source/Models/V100S_EfficientNetB0/V100S_EfficientNetB0_Result_Batchxxx_Epochyyy.csv 中
	  d.2. 对于 前向传递 测试，数据批尺寸（Batch Size）为1、8、16、32、64、128的情况：
	      - 当Batch Size = 1时, 修改版EfficientNet B0比原有模型平均快了15%，速度达到了V100S上的91%
	      - 当Batch Size = 8时, 修改版EfficientNet B0比原有模型平均快了14%，速度达到了V100S上的61%
	      - 当Batch Size = 16时,修改版EfficientNet B0比原有模型平均快了1%，速度达到了V100S上的53%
	      - 当Batch Size = 32时,修改版EfficientNet B0比原有模型平均快了-10%，速度达到了V100S上的46%
	      - 当Batch Size = 64时,修改版EfficientNet B0比原有模型平均快了-11%，速度达到了V100S上的48%
	      - 当Batch Size = 128时,修改版EfficientNet B0比原有模型平均快了-13%，速度达到了V100S上的51%
         d.3. 由于完整训练一次模型需要花费20多个小时，因此，我们仅测量了训练3个epoch花费的时间。在 DCU_TestModifiedEfficientNet.py 第24行，可以修改EPOCHS数量，如果修改为185左右，即可完整地训练模型。
	      - 当Batch Size = 8时，修改版EfficientNet B0比原有模型，训练速度快了-2.7%，推理速度快了11.8%。训练速度达到了V100S的43%，推理速度达到了V100S的51%。DCU已有模型训练速度到达了V100S的44%，推理速度达到V100S的45%
	      - 当Batch Size = 16时，修改版EfficientNet B0比原有模型，训练速度快了-2.8%，推理速度快了11%。训练速度达到了V100S的57%，推理速度达到了V100S的60%。DCU已有模型训练速度到达了V100S的59%，推理速度达到V100S的53%
	      - 当Batch Size = 32时，修改版EfficientNet B0比原有模型，训练速度快了-4.1%，推理速度快了-8%。训练速度达到了V100S的70%，推理速度达到了V100S的67%。DCU已有模型训练速度到达了V100S的72%，推理速度达到V100S的72%
	      - 当Batch Size = 64时，修改版EfficientNet B0比原有模型，训练速度快了-4.3%，推理速度快了-9.3%。训练速度达到了V100S的73%，推理速度达到了V100S的72%。DCU已有模型训练速度到达了V100S的76%，推理速度达到V100S的79%
	      - 当Batch Size = 128时，修改版EfficientNet B0比原有模型，训练速度快了-5.3%，推理速度快了-12%。训练速度达到了V100S的77%，推理速度达到了V100S的71%。DCU已有模型训练速度到达了V100S的81%，推理速度达到V100S的80%
	 d.4. 之前的测试证明，我们的核函数在某些情况下是有优化效果的。但是有几个重要的原因使得模型整体优化效果不佳：
	      - 训练和推理过程中的数据移动时间（cpu -> dcu，dcu -> cpu）。这是模型训练和推理性能的重要瓶颈，也依赖于硬件的提升
	      - 项目不针对后向传递和模型中的其他层（池化层、Batch Normalization层、全连接层等）进行优化。它们也占据了很多时间
	      - PyTorch Extension引入了一些额外的开销



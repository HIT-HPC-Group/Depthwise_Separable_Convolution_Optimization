from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='optimizedDepthwise',
    version="1.0",
    author="Zheng Liu",
    description="Optimized Depthwise Convolution Implementations for MobileNet V2 and EfficientNet B0",
    author_email="zhengmichaelliu@gmail.com",
    keywords="Deep Learning Depthwise Separable Convolution",
    
    ext_modules=[
        CUDAExtension(
            name='optimizedDepthwise_cuda', 
            sources=['V100S_Depthwise.cpp','V100S_Depthwise_Kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)

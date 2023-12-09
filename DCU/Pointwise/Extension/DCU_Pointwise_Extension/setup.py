from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='optimizedPointwise',
    version="1.0",
    author="Zheng Liu",
    description="Optimized Pointwise Convolution Implementations for MobileNet V2 and EfficientNet B0",
    author_email="zhengmichaelliu@gmail.com",
    keywords="Deep Learning Depthwise Separable Convolution",
    
    ext_modules=[
        CUDAExtension(
            name='optimizedPointwise_cuda', 
            sources=['DCU_Pointwise.cpp','DCU_Pointwise_Kernel.hip'])
    ],
    cmdclass={'build_ext': BuildExtension}
)

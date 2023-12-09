#include <torch/extension.h>
#include <vector>
//#include <ATen/NativeFunctions.h>
//#include <ATen/Functions.h>
//#include <ATen/Config.h>
#include <array>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declaration
torch::Tensor optimizedDepthwise_cuda_forward(
  torch::Tensor input, 
  torch::Tensor filter,
  int filterHeight,
  int stride);

// CUDA forward definition
torch::Tensor optimizedDepthwise_forward(
    torch::Tensor input,
    torch::Tensor filter,
    int filterHeight,
    int stride) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(filter);

    return optimizedDepthwise_cuda_forward(
      input,
      filter,
      filterHeight,
      stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimizedDepthwise_forward, "Optimized Depthwise forward (CUDA)");
}

#include <torch/extension.h>
#include <vector>
//#include <ATen/NativeFunctions.h>
//#include <ATen/Functions.h>
//#include <ATen/Config.h>
#include <array>

// CUDA forward declaration
torch::Tensor optimizedPointwise_cuda_forward(
  torch::Tensor input, 
  torch::Tensor filter);

// CUDA forward definition
torch::Tensor optimizedPointwise_forward(
    torch::Tensor input,
    torch::Tensor filter) {

    return optimizedPointwise_cuda_forward(
      input,
      filter);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &optimizedPointwise_forward, "Optimized Pointwise forward (CUDA)");
}

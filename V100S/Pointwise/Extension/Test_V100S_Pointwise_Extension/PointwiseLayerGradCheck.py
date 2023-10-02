import torch
from torch import nn

import math

# extension name
from PointwiseLayer import OptimizedPointwiseFunction

cuda_device = torch.device("cuda")

inputData = torch.randn(128, 7, 576, 576, requires_grad = False).to(cuda_device)
filter = torch.randn(160, 576, 1, 1, requires_grad = True).to(cuda_device)
print("Testing")
print(torch.autograd.gradcheck(OptimizedPointwiseFunction.apply, (inputData, filter), eps = 1e-2, atol = 1e-2))

import torch
from torch import nn

import math

# extension name
from DepthwiseLayer import OptimizedDepthwiseFunction

cuda_device = torch.device("cuda")

# (32, 112, 112, 3, 1)
inputData = torch.randn(1, 32, 112, 112, requires_grad = False).to(cuda_device)
filter = torch.randn(32, 1, 3, 3, requires_grad = True).to(cuda_device)
print("Testing")
print(torch.autograd.gradcheck(OptimizedDepthwiseFunction.apply, (inputData, filter, 3, 1, 1, 1, 32), eps=1e-2, atol=1e-2))

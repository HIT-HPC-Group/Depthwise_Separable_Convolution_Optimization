import torch
from torch import nn

import math

# extension name
import optimizedPointwise_cuda

class OptimizedPointwiseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter):
        ctx.save_for_backward(input, filter)

        output = optimizedPointwise_cuda.forward(input, filter)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, filter = ctx.saved_tensors

        conf = ctx.conf
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            #input_ = grad_output.new_empty(1).expand(input.shape)
            # parameters: grad_output, input, filter, None, (conf["stride"], conf["stride"]), (conf["padding"], conf["padding"]), (conf["dilation"], conf["dilation"]), False, [0], conf["groups"], (True, False, False)
            grad_input = torch.ops.aten.convolution_backward(grad_output, input, filter, None, (1, 1), (0, 0), (1, 1), False, [0], 1, (True, False, False))[0]
        
        if ctx.needs_input_grad[1]:
            #filter_ = grad_output.new_empty(1).expand(filter.shape)
            grad_weight = torch.ops.aten.convolution_backward(grad_output, input, filter, None, (1, 1), (0, 0), (1, 1), False, [0], 1, (False, True, False))[1]
        
        return grad_input, grad_weight
        
class OptimizedPointwiseLayer(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(OptimizedPointwiseLayer, self).__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel

        self.filter = nn.Parameter(torch.empty((self.outputChannel, self.inputChannel, 1, 1), dtype = torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.outputChannel * self.inputChannel)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return OptimizedPointwiseFunction.apply(
            input, 
            self.filter)

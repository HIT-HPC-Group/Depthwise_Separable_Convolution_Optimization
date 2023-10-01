import torch
from torch import nn

import math

# extension name
import optimizedDepthwise_cuda

class OptimizedDepthwiseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, filterHeight, stride, padding, dilation, groups):
        ctx.save_for_backward(input, filter)    
        ctx.conf = {
            "filterHeight": filterHeight,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups
        }

        output = optimizedDepthwise_cuda.forward(input, filter, filterHeight, stride)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, filter = ctx.saved_tensors

        conf = ctx.conf
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            #input_ = grad_output.new_empty(1).expand(input.shape)
            grad_input = torch.ops.aten.convolution_backward(grad_output, input, filter, None,
                                               (conf["stride"], conf["stride"]), (conf["padding"], conf["padding"]), (conf["dilation"], conf["dilation"]),
                                               False, [0], conf["groups"], (True, False, False))[0]
        
        if ctx.needs_input_grad[1]:
            #filter_ = grad_output.new_empty(1).expand(filter.shape)
            grad_weight = torch.ops.aten.convolution_backward(grad_output, input, filter, None,
                                               (conf["stride"], conf["stride"]), (conf["padding"], conf["padding"]), (conf["dilation"], conf["dilation"]),
                                               False, [0], conf["groups"], (False, True, False))[1]
        
        return grad_input, grad_weight, None, None, None, None, None
        
class OptimizedDepthwiseLayer(nn.Module):
    def __init__(self, inputChannel, outputChannel, filterHeight, stride):
        super(OptimizedDepthwiseLayer, self).__init__()
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.filterHeight = filterHeight
        self.stride = stride
        if(self.filterHeight == 3):
            self.padding = 1
        elif(self.filterHeight == 5):
            self.padding = 2
        self.dilation = 1
        self.groups = inputChannel
    
        self.filter = nn.Parameter(torch.empty((self.inputChannel, 1, self.filterHeight, self.filterHeight), dtype = torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.inputChannel * self.filterHeight * self.filterHeight)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return OptimizedDepthwiseFunction.apply(
            input, 
            self.filter, 
            self.filterHeight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups)

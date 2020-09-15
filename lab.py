import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import math

'''
Given a batch of images and a HUGE tensor produced by the network,
compute the output batch of images
'''

def calcOutput(data, kernel, ker_size=3):
    N = (list(data.size()))[0]
    in_ch = (list(data.size()))[1]
    x = (list(data.size()))[2]
    y = (list(data.size()))[3]
    pad = (ker_size-1)//2
    ZeroPad = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    data_pad = ZeroPad(data)
    # Add padding to the original data
    print("data_pad:") #checked
    print(data_pad)
    reshape_data = data_pad.unfold(2,ker_size,1).unfold(3,ker_size,1)
    print("reshape_data (splitting the data into patches):")#checked
    print(reshape_data.size())
    print(reshape_data)
    
    soft_max = nn.Softmax(dim=4)
    
    reshape_kernel = kernel.reshape(N, in_ch, x, y, ker_size**2)
    exp_kernel = reshape_kernel.clone()
    reshape_kernel = soft_max(reshape_kernel)
    
    print("Softmaxed:") #checked
    print(reshape_kernel)
    
    # Stupid way of looping to get the softmax result. It should be the same with the previous tensor
    exp_kernel = torch.exp(exp_kernel)
    for i in range(N):
        for j in range(in_ch):
            for k in range(x):
                for l in range(y):
                    tmp = torch.sum(exp_kernel[i][j][k][l])
                    exp_kernel[i][j][k][l]/=tmp
                    
    print("Hand Calculated:")
    print(exp_kernel)
    
    reshape_kernel = reshape_kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
    print("Reshaped kernel:") #checked
    print(reshape_kernel)
    scalar_product = torch.mul(reshape_data, reshape_kernel)
    print("Scalar product:")
    print(scalar_product)
    result = torch.sum(scalar_product, dim=(4,5), keepdim=False)
    print("Sum:")
    print(result)
    return result

if __name__=="__main__":
    data = torch.randn(3,1,5,5)
    print("Input data: ")
    print(data)
    kernel = torch.randn(3,9,5,5)
    print("Input kernel: ")
    print(kernel)
    t = calcOutput(data, kernel, 3)

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
    
    print("Softmaxed:")
    print(reshape_kernel)
    
    exp_kernel = torch.exp(exp_kernel)
    for i in range(N):
        for j in range(in_ch):
            for k in range(x):
                for l in range(y):
                    tmp = torch.sum(exp_kernel[i][j][k][l])
                    exp_kernel[i][j][k][l]/=tmp
                    
    print("Hand Calculated:")
    print(exp_kernel)
    
    
    exp_kernel = exp_kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
    print("Reshaped kernel:")
    print(exp_kernel)
    scalar_product = torch.mul(reshape_data, exp_kernel)
    print("Scalar product:")
    print(scalar_product)
    result = torch.sum(scalar_product, dim=(4,5), keepdim=False)
    print("Sum:")
    print(result)
    return result

'''
if __name__=="__main__":

    arr = np.array([[[[1, 2, 3, 4, 5],[2, 3, 5, 4, 1],[1, 2, 4, 3, 5],[1, 5, 3, 4, 2],[1, 3, 2, 5, 4]]]])
    data = torch.from_numpy(arr)
    print(data.size())
    kernel = torch.zeros(1,9,5,5)
    for i in range(9):
        kernel[0][i] = data[0][0]
    print(kernel)
    
    t = calcOutput(data, kernel, 3)
'''
if __name__=="__main__":
    data = torch.randn(3,1,5,5)
    print("Input data: ")
    print(data)
    kernel = torch.randn(3,9,5,5)
    print("Input kernel: ")
    print(kernel)
    t = calcOutput(data, kernel, 3)

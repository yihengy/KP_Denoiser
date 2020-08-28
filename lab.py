import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np

'''
Given a batch of images and a HUGE tensor produced by the network,
compute the output batch of images
'''

def calcOutput(data, kernel, ker_size=5):
    pad = (ker_size-1)//2
    ZeroPad = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    data_pad = ZeroPad(data)
    N = (list(data.size()))[0]
    in_ch = (list(data.size()))[1]
    x = (list(data.size()))[2]
    y = (list(data.size()))[3]
    pad = (ker_size-1)//2
    ZeroPad = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    data_pad = ZeroPad(data)
    reshape_data = data_pad.unfold(2,ker_size,1).unfold(3,ker_size,1)
    soft_max = nn.Softmax(dim=3)
    reshape_kernel = kernel.reshape(N, in_ch, x, y, ker_size**2)
    reshape_kernel = soft_max(reshape_kernel)
    reshape_kernel = reshape_kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
    scalar_product = torch.mul(reshape_data, reshape_kernel)
    result = torch.sum(scalar_product, dim=(4,5), keepdim=False)
    return result
    
if __name__=="__main__":
    arr = np.array([[[[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]]])
    data = torch.from_numpy(arr)
    print(data.size())
    kernel = torch.zeros(1,9,5,5)
    for i in range(9):
        kernel[0][i] = data[0][0]
    print(kernel)
    

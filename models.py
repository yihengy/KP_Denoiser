import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

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
    soft_max = nn.Softmax2d()
    reshape_kernel = kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
    print(reshape_kernel)
    for a in range(N):
        for b in range(in_ch):
            reshape_kernel[a][b] = soft_max(reshape_kernel[a][b])
    print(reshape_kernel)
    
    scalar_product = torch.mul(reshape_data, reshape_kernel)
    result = torch.sum(scalar_product, dim=(4,5), keepdim=False)
    return result

def calcOutput_2(data, kernel, ker_size=5):
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
    soft_max = nn.Softmax(dim=5)
    reshape_kernel = kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
    #print(reshape_kernel)
    reshape_kernel = soft_max(reshape_kernel)
    print(reshape_kernel)
    
    scalar_product = torch.mul(reshape_data, reshape_kernel)
    result = torch.sum(scalar_product, dim=(4,5), keepdim=False)
    return result
    
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9, ker_size=3, o_k_size=5):
        super(DnCNN, self).__init__()
        padding = (ker_size-1)//2
        o_channels = o_k_size**2
        features = 150
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=ker_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ker_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(in_channels=features, out_channels=o_channels, kernel_size=ker_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
    def forward(self, data, o_k_size=5):
        o_kernel = self.dncnn(data)
        result = calcOutput(data, o_kernel, o_k_size)
        return result


class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            max_patch_loss = 0
            for j in range(list(output_patches.size())[0]):
                for k in range(list(output_patches.size())[1]):
                    max_patch_loss = max(max_patch_loss, f.l1_loss(output_patches[j][k], target_patches[j][k]))
            avg_loss+=max_patch_loss
        avg_loss/=len(output)
        return avg_loss


if __name__=="__main__":
    criterion = PatchLoss()
    dtype = torch.FloatTensor
    x = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    y = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    loss = criterion(x, y, 10)
    test = torch.rand(10,1,5,5).unfold(2,3,1).unfold(3,3,1)
    print(test.size())
    
    data = 2*torch.ones(10, 1, 5, 5)
    kernel = 2*torch.ones(10, 9, 5, 5)
    result = calcOutput(data, kernel, 3)
    result2 = calcOutput_2(data, kernel, 3)
    #print(result.size())



'''
NOTE
Example: input: torch.Size([10, 1, 16, 16]), batchsize=10, 1 channel, image size 16*16
output: torch.Size([10, 9, 16, 16])
'''
    

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
    N = (list(data.size()))[0]
    in_ch = (list(data.size()))[1]
    x = (list(data.size()))[2]
    y = (list(data.size()))[3]
    pad = (ker_size-1)//2
    ZeroPad = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    data_pad = ZeroPad(data)
    reshape_data = data_pad.unfold(2,ker_size,1).unfold(3,ker_size,1)
    soft_max = nn.Softmax(dim=4)
    reshape_kernel = kernel.reshape(N, in_ch, x, y, ker_size**2)
    reshape_kernel = soft_max(reshape_kernel)
    reshape_kernel = reshape_kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
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
        
class NewPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(NewPatchLoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, output, target, patch_size):
        N = (list(output.size()))[0]
        num_ch = (list(output.size()))[1]
        loss_val=0
        for i in range(num_ch):
            avg_loss = 0
            for j in range(N):
                output_patches = output[j][i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
                target_patches = target[j][i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
                max_patch_loss = 0
                for j in range(list(output_patches.size())[0]):
                    for k in range(list(output_patches.size())[1]):
                        max_patch_loss = max(max_patch_loss, f.l1_loss(output_patches[j][k], target_patches[j][k]))
                avg_loss+=max_patch_loss
            avg_loss /= N
            loss_val+=avg_loss
        return loss_val/num_ch
        
if __name__=="__main__":
    criterion_1 = PatchLoss()
    criterion_2 = NewPatchLoss()
    x = torch.randn(5, 100, 100)
    y = torch.randn(5, 100, 100)
    
    x_unsqueeze = x.unsqueeze(1)
    y_unsqueeze = y.unsqueeze(1)
    
    loss_1 = criterion_1(x, y, 10)
    print(loss_1)
    loss_2 = criterion_2(x_unsqueeze, y_unsqueeze, 10)
    print(loss_2)



'''
NOTE
Example: input: torch.Size([10, 1, 16, 16]), batchsize=10, 1 channel, image size 16*16
output: torch.Size([10, 9, 16, 16])
'''
    

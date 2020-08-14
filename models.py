import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9, kernel_size=3, o_k_size=5):
        super(DnCNN, self).__init__()
        padding = (kernel_size-1)/2
        o_channels = o_k_size**2
        o_padding = (o_k_size-1)/2
        features = 150
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(in_channels=features, out_channels=o_channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, data):
        o_kernel = self.dncnn(data).reshape(o_k_size,o_k_size)
        batch_size = list(data.size())[0]
        img_size = list(data.size())[1] # size of the input image
        output = torch.zeros(batch_size, img_size, img_size)
        neighborhood = #slicing
        for i in range(batch_size):
            for j in rang(img_size):
                for k in range(img_size):
                    data[i][j][k] = o_kernel*data[i][j][k]
                    sum_elem = data[i][j][k].flatten()
        
        
        
        return out






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
'''
    x = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    y = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    
    loss = criterion(x, y, 10)
    
    
    net = DnCNN()
    input = torch.randn(1, 1, 32, 32)
    print("input:")
    print(input)
    
    out = net(input)
    print("output:")
    print(out)
'''

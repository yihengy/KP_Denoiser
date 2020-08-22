import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

'''
Given a batch of images and a HUGE tensor produced by the network,
compute the output batch of images
'''

# reshape an N*input_channel*x*y tensor into an N*input_channel*x*y*(k*k) tensor
def reshape(data,ker_size=5):
    pad = (ker_size-1)//2
    ZeroPad = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    data_pad = ZeroPad(data)
    
    N = (list(data.size()))[0]
    in_ch = (list(data.size()))[1]
    x = (list(data.size()))[2]
    y = (list(data.size()))[3]
    
    reshape_data = torch.zeros(N, in_ch, x, y, in_ch*ker_size**2)
    
    for i in range(N):
        for j in range(in_ch):
            for k in range(x):
                for l in range(y):
                    m = 0
                    for n in range(ker_size):
                        for o in range(ker_size):
                            reshape_data[i][j][k][l][m] = data_pad[i][j][k+n][l+o]
                            m += 1
    return reshape_data

# Reshape an N*1*x*y tensor into an N*1*x*y*(k*k) tensor
def reshape1Ch(data, ker_size=5):
    pad = (ker_size-1)//2
    ZeroPad = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
    data_pad = ZeroPad(data)
    output = data_pad.unfold(2,ker_size,1).unfold(3,ker_size,1)
    return output
    
def reshape_output(kernel, ker_size=5):
    N = (list(kernel.size()))[0]
    in_ch = (list(kernel.size()))[1] // (ker_size**2)
    x = (list(kernel.size()))[2]
    y = (list(kernel.size()))[3]
    o_ch = (list(kernel.size()))[1]
    
    reshape_kernel = torch.zeros(N, in_ch, x, y, o_ch)
    for i in range(N):
        for j in range(in_ch):
            for k in range(x):
                for l in range(y):
                    for m in range(o_ch):
                        reshape_kernel[i][j][k][l][m] = kernel[i][in_ch*j+m][k][l]
    # tested so far
    
    # Normalization of reshape_kernel = torch.zeros(N, in_ch, x, y, o_ch)
    for a in range(N):
        for b in range(in_ch):
            for c in range(x):
                for d in range(y):
                    sum = 0
                    for e in range(o_ch):
                        sum += reshape_kernel[a][b][c][d][e]
                        reshape_kernel[a][b][c][d] /= sum
    return reshape_kernel

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
    
    flatten_kernel = torch.flatten(kernel)
    
    reshape_kernel = flatten_kernel.reshape(N, in_ch, x, y, ker_size, ker_size)
    for a in range(N):
        for b in range(in_ch):
            for c in range(x):
                for d in range(y):
                    reshape_kernel[a][b][c][d]/=torch.sum(reshape_kernel[a][b][c][d])
    scalar_product = torch.mul(reshape_data, reshape_kernel)
    
    result = torch.zeros(N, in_ch, x, y)
    
    for o in range(N):
        for p in range(in_ch):
            for q in range(x):
                for r in range(y):
                    result[o][p][q][r] = torch.sum(scalar_product[o][p][q][r])
                    
    return result
    
    

def compute_output(data, kernel, ker_size=5):
    # data = N*in_channel*x*y
    # kernel = N*out_channel*x*y, out_channel=in_ch*k**2
    # transform kernel into N*input_channel*x*y*(k*k), and then do a simple matrix scalar product
    reshape_data = reshape(data, ker_size)
    reshape_kernel = reshape_output(kernel, ker_size)
    
    N = (list(data.size()))[0]
    in_ch = (list(data.size()))[1]
    x = (list(data.size()))[2]
    y = (list(data.size()))[3]
    o_ch = (list(kernel.size()))[1] // in_ch
    scalar_product = torch.mul(reshape_kernel, reshape_data)
    
    result = torch.zeros(N, in_ch, x, y)
    
    for o in range(N):
        for p in range(in_ch):
            for q in range(x):
                for r in range(y):
                    sum = 0
                    for s in range(o_ch):
                        sum += scalar_product[o][p][q][r][s]
                    result[o][p][q][r] = sum
                        
    #print(result.size())
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
    
    data = torch.rand(10, 1, 16, 16)
    kernel = torch.rand(10, 25, 16, 16)
    result = calcOutput(data, kernel, 5)
    print(result.size())
    
'''
    input = torch.randn(10, 25, 16, 16)
    reshape_output(input, 5)
    
    data = torch.randn(10, 1, 16, 16)
    reshape(data,5)
    
    t = compute_output(data, input, 5)
    print(t.size())
    
    net = DnCNN()
    shit = torch.randn(10, 1, 8, 8)
    shithole = net(shit, o_k_size=5)
    print(shithole.size())
'''




'''
NOTE
Example: input: torch.Size([10, 1, 16, 16]), batchsize=10, 1 channel, image size 16*16
output: torch.Size([10, 9, 16, 16])
'''
    

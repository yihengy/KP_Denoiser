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
    

def compute_output(data, kernel, ker_size):
    # data = N*in_channel*x*y
    # kernel = N*out_channel*x*y, out_channel=in_ch*k**2
    # transform kernel into N*input_channel*x*y*(k*k), and then do a simple matrix scalar product
    N = (list(data.size()))[0]
    in_ch = (list(data.size()))[1]
    x = (list(data.size()))[2]
    y = (list(data.size()))[3]
    o_ch = (list(kernel.size()))[1] // in_ch
    
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
    
    reshape_data = reshape(data, ker_size)
    
    scalar_product = torch.mul(reshape_kernel, reshape_data)
    
    result = torch.zeros(N, in_ch, x, y)
    
    for o in range(N):
        for p in range(in_ch):
            for q in range(x):
                for r
    
    
    
                    
    
    
    
    
                    

    
    

'''
def predict_output(data, kernel)
    # data = N*in_channel*x*y
    # kernel = N*out_channel*x*y
    
    # produce an output tensor with the same size as the input data
    data_size = list(data.size())
    output = torch.zeros(data_size)
    
    # ker_size = k*k
    ker_size = (list(kernel.size())[1]
    # batch_size N
    N = (list(data.size()))[0]
    in_channel = (list(data.size()))[1]
    # spacial dim
    length = (list(data.size()))[2]
    width = (list(data.size()))[3]
    
    # a HUGE tensor to do the computation
    compute_size = data_size.append(ker_size)
    compute_tensor = torch.zeros(compute_size)
    
    # put kernel into tensor with size N*x*y*out_channel
    for i in range(N):
        for j in range(length):
            for k in range(width):
                for l in range(ker_size):
                    compute_tensor[i][j][k][l] = kernel[l][i][j][k].clone()
                    
    # Normalize
    for i in range(N):
        for j in range(length):
            for k in range(width):
                for l in range(ker_size):
                    np_compute_tensor = compute_tensor.numpy()
                    devider = np.sum(np_compute_tensor[i][j][k])
                    compute_tensor[i][j][k][l] /= devider
    '''
    
    
    # reshaping the data


    
    
    
    

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9, kernel_size=3, o_k_size=3):
        super(DnCNN, self).__init__()
        #padding = (kernel_size-1)/2
        padding = 1
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
        o_kernel = self.dncnn(data)
        return o_kernel


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
    
    net = DnCNN()
    input = torch.randn(10, 1, 16, 16)
    out = net(input)
    #print(out.size())
    
    input = torch.randn(2, 1, 4, 4)
    kernel = torch.randn(2, 4, 4, 4)
    print(kernel)
    output = compute_output(input, kernel)
    print(output.size())
    print(output)
    
    
'''
NOTE
Example: input: torch.Size([10, 1, 16, 16]), batchsize=10, 1 channel, image size 16*16
output: torch.Size([10, 9, 16, 16])
'''
    

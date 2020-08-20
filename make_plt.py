import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
import os
import argparse
from models import DnCNN, PatchLoss, WeightedPatchLoss
from dataset import *
import glob
import torch.optim as optim
import uproot
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
#from magiconfig import ArgumentParser, MagiConfigOptions
from torch.utils.data import DataLoader
import math

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--datafile", nargs="?", type=str, default="./test.root", help='data path')
parser.add_argument("--model", type=str, default="./logs/81norm_100p/net.pth", help="Existing model")
parser.add_argument("--batchSize", type=int, default=100, help="batch size")
parser.add_argument("--noiselevel", type=int, default=15, help="noise")
parser.add_argument("--num_of_layers", type=int, default=9, help="layers")
parser.add_argument("--outf", type=str, default="", help="output director path")
parser.add_argument("--patchSize", type=int, default=50, help="patch size")
args = parser.parse_args()

model = DnCNN()
model.load_state_dict(torch.load(args.model))
model.eval()
model.to('cuda')

branch = get_all_histograms("./test.root")
model.to('cpu')
data_means = np.zeros(50)
noisy_means = np.zeros(50)
output_means = np.zeros(50)
n_d_ratio = np.zeros(50)
o_d_ratio = np.zeros(50)

for image in range(50):

    data = get_bin_weights(branch, image).copy()
    
    data_mean = np.mean(data)
    stdevs = np.std(data)
    
    noisy = add_noise(data, args.noiselevel).copy()
    
    data_norm = (data-data_mean)/stdevs
    noisy_norm = (noisy-data_mean)/stdevs
    
    data_norm = torch.from_numpy(data_norm)
    noisy_norm = torch.from_numpy(noisy_norm)
    
    noisy_norm = noisy_norm.unsqueeze(0)
    noisy_norm = noisy_norm.unsqueeze(1)
    
    output_norm = model(noisy_norm.float()).squeeze(0).squeeze(0).detach().numpy()
    output = (output_norm * stdevs) + data_mean
    
    noisy_mean = np.mean(noisy)
    output_mean = np.mean(output)
    
    data_means[image] = data_mean
    noisy_means[image] = noisy_mean
    output_means[image] = output_mean
    n_d_ratio[image] = noisy_mean/data_mean
    o_d_ratio[image] = output_mean/data_mean

plt.hist(n_d_ratio, histtype='bar', bins=100, color='b', label='noisy/truth')
plt.hist(o_d_ratio, histtype='bar', bins=100, color='r', label='recon/truth')
plt.legend()
plt.savefig(args.outf+"image_ratio.png")
plt.close()


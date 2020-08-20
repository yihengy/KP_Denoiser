import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import torch.utils.data as udata
import torch
import torchvision
import torchvision.transforms as tf
import math

def get_all_histograms(file_path):
    file = uproot.rootio.open(file_path)
    tree = file["g4SimHits/tree"]
    branch = tree.array("bin_weights")
    return branch;

def get_bin_weights(branch, n):
    data = np.zeros((100,100))
    count = 0
    for y in range(100):
        for x in range(100):
            data[99-x][y]=branch[n][count]
            count+=1
    # do random rotation/flips
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    if (flipx):
        data = np.fliplr(data)
    if flipy:
        data = np.flipud(data)
    for i in range(rot):
        data = np.rot90(data)
    return data;
    

def add_noise(data, sigma):
    return np.clip(data + np.random.normal(loc=0.0,scale=sigma, size=[100,100]), a_min=0, a_max=None);

class RootDataset(udata.Dataset):
    def __init__(self, root_file, sigma):
        self.root_file = root_file
        self.sigma = sigma
        self.histograms = get_all_histograms(root_file)

    def __len__(self):
        return len(self.histograms)

    def __getitem__(self, idx):
        truth_np = get_bin_weights(self.histograms, idx).copy()
        '''
        means, stdevs = [], []
        means.append(np.mean(truth_np))
        stdevs.append(np.std(truth_np))
        noisy_np = add_noise(truth_np, self.sigma).copy()
        truth = torch.from_numpy(truth_np)
        truth_norm = tf.Normalize(means,stdevs,inplace=False)(truth)
        noisy = torch.from_numpy(noisy_np)
        noisy_norm = tf.Normalize(means,stdevs,inplace=False)(noisy)
        '''
        
        means = np.mean(truth_np)
        stdevs = np.std(truth_np)
        noisy_np = add_noise(truth_np, self.sigma).copy()
        # manual normalization
        truth_np -= means
        truth_np /= stdevs
        
        noisy_np -= means
        noisy_np /= stdevs
        
        truth_norm = torch.from_numpy(truth_np)
        noisy_norm = torch.from_numpy(noisy_np)
        
        return truth_norm, noisy_norm
    
    def __stat__(self):
        return means, stdevs

if __name__=="__main__":
    dataset = RootDataset("./data/50pixels/part2.root", 20)
    truth, noise = dataset.__getitem__(0)
    plt.imshow(truth.numpy())
    plt.colorbar()
    plt.savefig("truth_norm.png")
    plt.close()
    plt.imshow(noise.numpy())
    plt.colorbar()
    plt.savefig("noise_norm.png")

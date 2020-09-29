# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, PatchLoss
from dataset import *
import glob
import torch.optim as optim
import uproot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=20, help='noise level')
parser.add_argument("--outf", type=str, default="logs/kp_930", help='path of log files')
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfile", type=str, default="./data/training/part1.root", help='path of .root file for training')
parser.add_argument("--valfile", type=str, default="./data/training/part1.root", help='path of .root file for validation')
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--model", type=str, default=None, help="Existing model, if applicable")
parser.add_argument("--patchSize", type=int, default=20, help="Size of patches to apply in loss function")
parser.add_argument("--kernelSize", type=int, default=3, help="Training kernel size")
parser.add_argument("--outKerSize", type=int, default=5, help="Output kernel size")
args = parser.parse_args()

'''
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
'''
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():

    # creat_readme()
    # choose cpu or gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print('Loading Dataset--')
    dataset_train = RootDataset(root_file=args.trainfile, sigma=args.sigma)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize)
    dataset_val = RootDataset(root_file=args.valfile, sigma=args.sigma)
    val_train = DataLoader(dataset=dataset_val, batch_size=args.batchSize)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers, ker_size=args.kernelSize, o_k_size=args.outKerSize).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model")
    else:
        print("Loading model from file" + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = PatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    MyOptim = optim.Adam(model.parameters(), lr = args.lr)
    MyScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=MyOptim, factor=0.1, patience=10, verbose=True)

    # training and validation
    step = 0
    training_losses = np.zeros(args.epochs)
    validation_losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Epoch #" + str(epoch))
        # training
        train_loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            MyOptim.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model(noise.float().to(args.device), args.outKerSize)
            batch_loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device),args.patchSize).to(args.device)
            batch_loss.backward()
            MyOptim.step()
            model.eval()
            train_loss += batch_loss.item()
        training_losses[epoch] = train_loss
        print("Training Loss: "+ str(train_loss))
        
        val_loss = 0
        for i, data in enumerate(val_train, 0):
            val_truth, val_noise =  data
            val_output = model(val_noise.unsqueeze(1).float().to(args.device), args.outKerSize)
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_truth.to(args.device),args.patchSize).to(args.device)
            val_loss+=output_loss.item()
        MyScheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss
        print("Validation Loss: "+ str(val_loss))
        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    training = plt.plot(training_losses, label='Training')
    validation = plt.plot(validation_losses, label='Validation')
    plt.legend()
    plt.savefig(args.outf + "/lossplt.png")
    
    branch = get_all_histograms("./data/training/part1.root")
    model.to('cpu')
    for image in range(10):
    
        data = get_bin_weights(branch, image).copy()
        np.savetxt(args.outf+'/truth#' + str(image) + '.txt', data)
        
        means = np.mean(data)
        stdevs = np.std(data)
        
        noisy = add_noise(data, args.sigma).copy()
        np.savetxt(args.outf+'/noisy#' + str(image) + '.txt', noisy)
        
        data_norm = (data-means)/stdevs
        #np.savetxt(args.outf+'/truth_norm#' + str(image) + '.txt', data_norm)
        noisy_norm = (noisy-means)/stdevs
        #np.savetxt(args.outf+'/noisy_norm#' + str(image) + '.txt', noisy_norm)
        
        data_norm = torch.from_numpy(data_norm)
        noisy_norm = torch.from_numpy(noisy_norm)
        noisy_norm = noisy_norm.unsqueeze(0)
        noisy_norm = noisy_norm.unsqueeze(1)
        output_norm = model(noisy_norm.float(), args.outKerSize).squeeze(0).squeeze(0).detach().numpy()
        #np.savetxt(args.outf+'/output_norm#' + str(image) + '.txt', output_norm)
        output = (output_norm * stdevs) + means
        np.savetxt(args.outf+'/output#' + str(image) + '.txt', output)
        #truth = data.numpy()
        #noisy = noisy.numpy()
        #diff = output-truth
        #noisy_diff = noisy-truth
        #np.savetxt(args.outf+'/diff#' + str(image) + '.txt', diff)
    model.to('cuda')
    
if __name__ == "__main__":
    main()

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpn import GPN
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
from env import Env


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description="GPN with RL")
    parser.add_argument('--size', default=400, help="size of TSP")
    parser.add_argument('--epoch', default=100, help="number of epochs")
    parser.add_argument('--batch_size', default=128, help='')
    parser.add_argument('--train_size', default=2500, help='')
    parser.add_argument('--val_size', default=1000, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    args = vars(parser.parse_args())

    size = int(args['size'])
    learn_rate = args['lr']    # learning rate
    B = int(args['batch_size'])    # batch_size
    B_val = int(args['val_size'])    # validation size
    steps = int(args['train_size'])    # training steps
    n_epoch = int(args['epoch'])    # epochs
    save_root ='./model/gpn_tsp_a_'+str(size)+'.pt'
    
    print('=========================')
    print('prepare to train')
    print('=========================')
    print('Hyperparameters:')
    print('size', size)
    print('learning rate', learn_rate)
    print('batch size', B)
    print('validation size', B_val)
    print('steps', steps)
    print('epoch', n_epoch)
    print('save root:', save_root)
    print('=========================')
    
    device = torch.device('cuda:2')
    
    model = GPN(n_feature=2, n_hidden=4, device=device).to(device)
    model = torch.load(save_root).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    lr_decay_step = 2500
    lr_decay_rate = 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                         lr_decay_step), gamma=lr_decay_rate)
    
    img = np.load('../data/data/a.npy')
    X = np.expand_dims(np.asarray(np.where(img==0)).T, axis=0)

    # B = X.shape[0]
    size_val = X.shape[1]
    B_val = X.shape[0]

    # validation data
    X_val = X.copy()

    C = 0     # baseline
    R = 0     # reward

    # R_mean = []
    # R_std = []
    for epoch in range(n_epoch):
        for i in range(steps):
            optimizer.zero_grad()

            X = np.random.rand(B, size, 2)

            # X = np.expand_dims(np.asarray(np.where(img==0)).T, axis=0)/20
        
            X = torch.Tensor(X).to(device)
            
            mask = torch.zeros(B,size).to(device)
        
            R = 0
            logprobs = 0
            reward = 0
            
            Y = X.view(B,size,2)
            x = Y[:,0,:]
            h = None
            c = None
        
            for k in range(size):
              
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
        
                Y1 = Y[[i for i in range(B)], idx.data].clone()
                if k == 0:
                    Y_ini = Y1.clone()
                if k > 0:
                    reward = torch.norm(Y1-Y0, dim=1)
                    
                Y0 = Y1.clone()
                x = Y[[i for i in range(B)], idx.data].clone()
                
                R += reward
                    
                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY) 
                
                mask[[i for i in range(B)], idx.data] += -np.inf 
                
            # R += torch.norm(Y1-Y_ini, dim=1)
            
            # self-critic base line
            mask = torch.zeros(B,size).to(device)
            
            C = 0
            baseline = 0
            
            Y = X.view(B,size,2)
            x = Y[:,0,:]
            h = None
            c = None
            
            for k in range(size):
            
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
            
                # sampler = torch.distributions.Categorical(output)
                # idx = sampler.sample()         # now the idx has B elements
                idx = torch.argmax(output, dim=1)    # greedy baseline
            
                Y1 = Y[[i for i in range(B)], idx.data].clone()
                if k == 0:
                    Y_ini = Y1.clone()
                if k > 0:
                    baseline = torch.norm(Y1-Y0, dim=1)
            
                Y0 = Y1.clone()
                x = Y[[i for i in range(B)], idx.data].clone()
            
                C += baseline
                mask[[i for i in range(B)], idx.data] += -np.inf
        
            # C += torch.norm(Y1-Y_ini, dim=1)
        
            gap = (R-C).mean()
            loss = ((R-C-gap)*logprobs).mean()

            # loss = R*logprobs
        
            loss.backward()
            
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()

            if i % 50 == 0:
                print("epoch:{}, batch:{}/{}, reward:{}"
                    .format(epoch, i, steps, R.mean().item()))
                # R_mean.append(R.mean().item())
                # R_std.append(R.std().item())
                
                # greedy validation
                
                tour_len = 0

                X = np.expand_dims(np.asarray(np.where(img==0)).T, axis=0)/int(size**(1/2))
                X = torch.Tensor(X).to(device)
                
                mask = torch.zeros(B_val,size_val).to(device)
                
                R = 0
                logprobs = 0
                Idx = []
                reward = 0
                
                Y = X.view(B_val, size_val, 2)    # to the same batch size
                x = Y[:,0,:]
                h = None
                c = None
                
                for k in range(size):
                    
                    output, h, c, hidden_u = model(x=x, X_all=X, h=h, c=c, mask=mask)
                    
                    sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)
                    Idx.append(idx.detach().cpu().numpy()[0])
                
                    Y1 = Y[[i for i in range(B_val)], idx.data]
                    
                    if k == 0:
                        Y_ini = Y1.clone()
                    if k > 0:
                        reward = torch.norm(Y1-Y0, dim=1)
            
                    Y0 = Y1.clone()
                    x = Y[[i for i in range(B_val)], idx.data]
                    
                    R += reward
                    
                    mask[[i for i in range(B_val)], idx.data] += -np.inf
            
                # R += torch.norm(Y1-Y_ini, dim=1)
                tour_len += R.mean().item()
                print('validation tour length:', tour_len)
                pred = X_val[0][Idx].T 
                plt.imshow(img)
                plt.plot(pred[1], pred[0])
                plt.savefig(f'./pred_a_{str(size)}.png')
                plt.close()

            print(f'step: {i+1}/{steps} {i/steps*100:3.1f}%, loss:{loss.detach().cpu().numpy():.5f}', end='\r')

        print('save model to: ', save_root)
        torch.save(model, save_root)

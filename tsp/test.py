import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from gpn import GPN
from tqdm import tqdm
import numpy as np
from pathlib import Path
from k_opt import two_opt, cal_distance, three_opt

# args
parser = argparse.ArgumentParser(description="GPN test")
parser.add_argument('--size', default=400, help="size of model")
args = vars(parser.parse_args())
size = int(args['size'])

device = torch.device('cuda:2')

# test image
# data_path = '../data/data/a.npy'
data_path = '../data/square/npy/medium_20x20.npy'
img = np.load(data_path)
orig_X = np.expand_dims(np.asarray(np.where(img==0)).T, axis=0)

load_root ='./model/gpn_tsp_a_'+str(size)+'.pt'

print('=========================')
print('test for TSP'+str(size))
print('=========================')

model = torch.load(load_root).to(device)
model.eval()

# greedy
total_tour_len = 0

tour_len = 0

X = np.expand_dims(np.asarray(np.where(img==0)).T, axis=0)/int(size**(1/2))
X = torch.Tensor(X).to(device)

B = X.shape[0]
size = X.shape[1]

mask = torch.zeros(B, size).to(device)

R = 0
Idx = []
reward = 0

Y = X.view(B, size, 2)           # to the same batch size
x = Y[:,0,:]
h = None
c = None

for k in range(size):
    
    output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
    
    idx = torch.argmax(output, dim=1)
    Idx.append(idx.detach().cpu().numpy()[0])

    Y1 = Y[[i for i in range(B)], idx.data].clone()
    if k == 0:
        Y_ini = Y1.clone()
    if k > 0:
        reward = torch.norm(Y1-Y0, dim=1)

    Y0 = Y1.clone()
    x = Y[[i for i in range(B)], idx.data].clone()
    
    R += reward

    mask[[i for i in range(B)], idx.data] += -np.inf
    
# tour_len += R.mean().item()
print('Original tour length:', cal_distance(Idx, orig_X[0]))
# total_tour_len += tour_len

search_size = 100

best_route, best_distance = two_opt(Idx, orig_X[0], improvement_threshold=0.001)
# best_route, best_distance = three_opt(Idx, orig_X[0], improvement_threshold=0.001)

pred = orig_X[0][best_route].T
plt.imshow(img)
plt.plot(pred[1], pred[0])
plt.savefig(f'./pred_{Path(data_path).stem}_{str(size)}_2opt.png')
plt.close()

print('Final tour length:', best_distance)

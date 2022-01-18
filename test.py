from env import Env

import numpy as np
from copy import deepcopy
from tqdm import tqdm

class Graph:
    
    def __init__(self, env):
        self.env = env
        self.nodes_key = []
        self.nodes_value = []
        self.edge_info = {}
    
    def build_graph(self):
        init_obs = self.env.reset().flatten()
        if str(init_obs) not in self.nodes_key:
            self.nodes_key.append(str(init_obs))
            self.nodes_value.append(init_obs.tolist())
        
        for _ in range(1):
            copy_env = deepcopy(self.env)
            obs = init_obs.copy()

            done = False
            while not done:
                copy_env.render()
                random_output = np.random.rand(copy_env.action_space)
                action = np.argmax(random_output)
                next_obs, reward, done, _ = copy_env.step(action)

                next_obs = next_obs.flatten()
                if str(next_obs) not in self.nodes_key:
                    self.nodes_key.append(str(next_obs))
                    self.nodes_value.append(next_obs.tolist())

                obs_idx = self.nodes_key.index(str(obs))
                next_obs_idx = self.nodes_key.index(str(next_obs))
                self.edge_info[(obs_idx, next_obs_idx)] = reward

                obs = next_obs.copy()
    
    def build_adj_matrix(self):
        adj_matrix = np.zeros((len(self.nodes_key), len(self.nodes_key)))
        adj_info_matrix = np.zeros((len(self.nodes_key), len(self.nodes_key)))
#         adj_matrix *= -5
        for key, value in self.edge_info.items():
            adj_matrix[key] = 1
            adj_info_matrix[key] = value
        self.adj_matrix = adj_matrix
        self.adj_info_matrix = adj_info_matrix
        return adj_matrix, adj_info_matrix


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref


class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
    
    
    def forward(self, x, h, c):       # query and reference
        
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c


class GPN(torch.nn.Module):
    
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        
        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)
        
        
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
        
#         self.output = nn.Conv1d(1, n_output, 3)
    
    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)
        
        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)
        
        
        # =============================
        # vector context
        # =============================
        
        # x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1)   # (B, size)
        # X_all = X_all - x_expand
        
        # the weights share across all the cities
        x = self.embedding_x(x)
        context = self.embedding_all(X_all)
        
        # =============================
        # process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
        
        
        # =============================
        # graph neural network encoder
        # =============================
        
        # (B, size, dim)
        context = context.view(-1, self.dim)
        
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context))
        
        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context))
        
        
        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # query vector
        q = h
        
        # pointer
        u, _ = self.pointer(q, context)
        
        latent_u = u.clone()
        
        u = 10 * torch.tanh(u) * mask
        
        if latent is not None:
            u += self.alpha * latent
        
#         u = self.output(u.unsqueeze(1))
#         u = torch.mean(u, -1)
    
        return F.softmax(u, dim=1), h, c, latent_u

if __name__=="__main__":

    n_epoch = 100
    n_steps = 250

    img = np.load('./data/square/npy/extreme_5x5.npy')
    env = Env([img])

    model = GPN(n_feature=2, n_hidden=128).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    graph = Graph(env)
    [graph.build_graph() for _ in range(2)]
    adj_matrix_val, adj_info_matrix_val = graph.build_adj_matrix()
    
    print(adj_matrix_val)

    X_val = torch.tensor(graph.nodes_value, dtype=torch.float).view(1, len(graph.nodes_value), 25).cuda()
    mask_val = torch.from_numpy(graph.adj_matrix[0]).unsqueeze(0).cuda()
    
    val_crane_distances = []
    
    for epoch in range(n_epoch):
        for step in tqdm(range(n_steps)):
            
            graph = Graph(env)
            graph.build_graph()
            graph.build_adj_matrix()
            X = torch.tensor(graph.nodes_value, dtype=torch.float).view(1, len(graph.nodes_value), 25).cuda()
            mask = torch.from_numpy(graph.adj_matrix[0]).unsqueeze(0).cuda()
            x = X[:,0,:]
            h = None
            c = None
            
            old_idx = 0
            
            R = 0
            logprobs = 0
            reward = 0
            
            for k in range(len(graph.nodes_key)):
                if torch.sum(x)==0:
                    continue
                output, h, c, _ = model(x=x, X_all=X, mask=mask, h=None, c=None)
                
                try:
                    sampler = torch.distributions.Categorical(output)
                    idx = sampler.sample()
                except Exception as e:
                    print(x)
                    print(output)
                    print(mask)
                    raise e
                
                Y1 = X[0, idx.data].clone()
                if k == 0:
                    Y_init = Y1.clone()
                if k > 0:
                    reward = graph.adj_info_matrix[[old_idx.detach().cpu().numpy().data, idx.data]]
                    
                Y0 = Y1.clone()
                x = X[0, idx.data].clone()
                
                R += reward
                
                TINY = 1e-15
                logprobs += torch.log(output[0, idx.data]+TINY) 

                mask = torch.from_numpy(graph.adj_matrix[idx.data]).unsqueeze(0).cuda()
                old_idx = idx.data
                
                if torch.sum(mask) ==0:
                    break
                
                
            C = 0
            baseline = 0
            
            mask = torch.from_numpy(graph.adj_matrix[0]).unsqueeze(0).cuda()
            x = X[:,0,:]
            h = None
            c = None
            
            old_idx = 0
            
            for k in range(len(graph.nodes_key)):
                if torch.sum(x)==0:
                    continue
                output, h, c, _ = model(x=x, X_all=X, mask=mask, h=None, c=None)
                
                idx = torch.argmax(output, dim=1)
                
                Y1 = X[0, idx.data].clone()
                if k == 0:
                    Y_init = Y1.clone()
                if k > 0:
                    baseline = graph.adj_info_matrix[[old_idx.detach().cpu().numpy().data, idx.data]]
                    
                Y0 = Y1.clone()
                x = X[0, idx.data].clone()
                
                C += baseline

                mask = torch.from_numpy(graph.adj_matrix[idx.data]).unsqueeze(0).cuda()
                old_idx = idx.data
                
                if torch.sum(mask) ==0:
                    break
            
            loss = torch.tensor(R-C).cuda()*logprobs
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0 and step > 0:
                print(f"epoch : {epoch}, step : {step}/{n_steps}, reward : {R}")
                crane_distance = 0
                
                X = X_val
                
                x = X[:,0,:]
                h = None
                c = None

                old_idx = 0
                print(x.view((5,5)))
                print()
                for k in range(len(graph.nodes_key)):
                    if torch.sum(x)==0:
                        continue
                    output, h, c, _ = model(x=x, X_all=X, mask=mask_val, h=None, c=None)

                    # sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)

                    Y1 = X[0, idx.data].clone()
                    if k == 0:
                        Y_init = Y1.clone()
                    if k > 0:
                        reward = adj_info_matrix_val[[old_idx.detach().cpu().numpy().data, idx.data]]

                    Y0 = Y1.clone()
                    x = X[0, idx.data].clone()
                    print(x.view((5,5)))
                    print()

                    crane_distance += reward

                    mask_val = torch.from_numpy(adj_matrix_val[idx.data]).unsqueeze(0).cuda()
                    old_idx = idx.data
                    
                print('validation crane_distance:', crane_distance)
                val_crane_distances.append(crane_distance)
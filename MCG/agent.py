from copy import deepcopy
import gc
import numpy as np
import random

import torch
import torch.optim as opt
from torch.nn import functional as F

from memory import MCTSMemory
from mcg import Node, Edge, Graph
from model import Model


class Agent:

    def __init__(self):

        self.buffer = MCTSMemory(10000)
        self.model = Model(8)
        self.lr = 0.0001
        self.device = 'cpu'
        self.gamma = 0.9
        self.optimizer = opt.Adam(self.model.parameters(), lr = self.lr)
        self.cpuct = 1
        self.mcg = None

    def choose_action(self, state, action_mask, tau, simulation_times, env):
        if self.mcg == None or str(state.flatten()) not in self.mcg.tree:
            self.build_mcg(state, str(state.flatten()))
        else:
            self.change_current_node_mcg(state)
        for sim in range(simulation_times):
            self.simulate(env)
        pi, values = self.get_action_value(env, 1)
        action, value = self.select_action(pi, values, action_mask, tau)
        return action, pi, value, values

    def simulate(self, env):


        copy_env = deepcopy(env)

        # print("current_node :", self.current_node)

        leaf, value, done, breadcrumbs = self.mcg.move_to_leaf(copy_env)

        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs, copy_env)

        del copy_env
        gc.collect()

        self.mcg.back_fill(value, breadcrumbs)

    def evaluate_leaf(self, leaf, value, done, breadcrumbs, env):

        if done == False:
            
            mask = env.get_action_mask()
            value, probs = self.get_preds(leaf.state, mask)
            state_action_pair = self.get_allowed_actions(env)

            probs = probs[list(state_action_pair.keys())]

            for idx, (action, next_state) in enumerate(state_action_pair.items()):

                if str(next_state.flatten()) not in self.mcg.tree:
                    node = Node(next_state, str(next_state.flatten()))
                    self.mcg.add_node(node)
                    new_edge = Edge(leaf, node, probs[idx], action)
                    leaf.edges.append((action, new_edge))

        return ((value, breadcrumbs))

    def get_action_value(self, env, tau):
        edges = self.mcg.current_node.edges
        pi = np.zeros(env.action_space, dtype=np.int32)
        values = np.zeros(env.action_space, dtype=np.float32)
        
        for action, edge in edges:
            pi[action] = edge.stats['N']**(1/tau)
            values[action] = edge.stats['Q']

        epsilon = 0.00001 if np.sum(pi) == 0 else 0

        pi = pi/(np.sum(pi)+epsilon)
        pi = np.exp(pi)/(np.sum(np.exp(pi)))
        
        return pi, values

    def get_allowed_actions(self, env):

        state_action_pair = {}
        mask = env.get_action_mask()
        if np.sum(mask) <= 0:
            return {}
        allowed_actions = np.where(mask==1)[0]

        for action in allowed_actions:
            copy_env = deepcopy(env)
            next_state, _, done, _ = copy_env.step(action)
            if done:
                continue
            state_action_pair[action] = next_state
        
        del copy_env
        gc.collect()
        return state_action_pair

    def get_preds(self, state, mask):
        preds = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

        value_array = preds[1]
        logits_array = preds[0]
        value = value_array[0].detach().cpu().numpy()

        logits = logits_array[0]

        mask_inf = torch.zeros_like(torch.tensor(mask), dtype=torch.float32, device=self.device)
        mask_inf.masked_fill_(mask_inf == 0, float('-inf'))

        for candi, m in enumerate(mask):
            if m==0: continue
            mask_inf[candi] = float(0.0)

        logits += mask_inf

        odds = np.exp(logits.squeeze().detach().cpu().numpy())
        probs = odds / np.sum(odds)

        return (value, probs)

    def select_action(self, pi, values, action_mask, tau):
        
        try:
            action_mask[action_mask==0] = -np.inf
            if sum(pi) == 0:
                pi = action_mask
            else:
                pi = action_mask*pi

            pi = np.exp(pi)/np.sum(np.exp(pi))

            if tau == 0:
                actions = np.argwhere(pi == max(pi))
                action = np.random.choice(actions)[0]
            else:
                action_idx = np.random.multinomial(1, pi)
                action = np.where(action_idx==1)[0][0]
        except Exception as e:
            print(pi)
            raise e

        value = values[action]

        return action, value

    def train(self):
        for i in range(10):
            minibatch = random.sample(self.buffer.ltmemory, min(128, len(self.buffer.ltmemory)))
            training_states = np.array([row['state'] for row in minibatch])            
            training_targets = {'value_head' : np.array([row['value'] for row in minibatch]), 'policy_head' : np.array([row['action_values'] for row in minibatch])}
            history = self.fit(training_states, training_targets)
        return history

    def fit(self, states, targets):
        
        v_loss = 0
        p_loss = 0
        for s, v_t, p_t in zip(states, targets['value_head'], targets['policy_head']):
            p, v = self.model(torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device))
            v_loss += F.mse_loss(torch.Tensor([v_t]).to(self.device), v.squeeze(0))
            p_loss += self.softmax_cross_entropy_with_logits(p_t, p)
        
        v_loss /= len(states)
        p_loss /= len(states)
        
        loss = 0.5*v_loss + 0.5*p_loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return {'loss' : loss, 'value_head_loss' : v_loss, 'policy_head_loss' : p_loss}
    
    def softmax_cross_entropy_with_logits(self, y_true, y_pred):

        p = y_pred
        pi = torch.Tensor(y_true).float().to(self.device)

        zero = torch.zeros(pi.shape).float().to(self.device)
        negatives = torch.ones(pi.shape).float().to(self.device) * (-100.0)
        p = torch.where(pi == zero, negatives, p).to(self.device)

        loss = -torch.mean(torch.sum(F.log_softmax(p, dim=1) * pi, dim=1))

        return loss

    def build_mcg(self, state, name):
        self.current_node = Node(state, name)
        self.mcg = Graph(self.current_node, self.cpuct)
        
    def change_current_node_mcg(self, state):
        self.mcg.current_node = self.mcg.tree[str(state.flatten())]
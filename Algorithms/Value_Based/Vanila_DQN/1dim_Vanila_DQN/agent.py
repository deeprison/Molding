import numpy as np
import time
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.optim as optim 
import torch.nn.functional as F 

from qnetwork import QNetwork 
from replay_buffer import ReplayBuffer

import wandb

class Agent:
    def __init__(self, 
                 env: 'Environment',
                 input_dim: ('int: The width and height of pre-processed input image'),
                 training_frames: ('int: The total number of training frames'),
                 eps_decay: ('float: Epsilon Decay_rate'),
                 gamma: ('float: Discount Factor'),
                 target_update_freq: ('int: Target Update Frequency (by frames)'),
                 update_type: ('str: Update type for target network. Hard or Soft')='hard',
                 soft_update_tau: ('float: Soft update ratio')=None,
                 batch_size: ('int: Update batch size')=32,
                 buffer_size: ('int: Replay buffer size')=1000000,
                 update_start_buffer_size: ('int: Update starting buffer size')=50000,
                 learning_rate: ('float: Learning rate')=0.0004,
                 eps_min: ('float: Epsilon Min')=0.1,
                 eps_max: ('float: Epsilon Max')=1.0,
                 device_num: ('int: GPU device number')=0,
                 rand_seed: ('int: Random seed')=None,
                 plot_option: ('str: Plotting option')=False,
                 model_path: ('str: Model saving path')='./',
                 trained_model_path: ('str: Trained model path')=''):

        self.action_dim = env.action_space
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        self.env = env
        self.input_dim = input_dim
        self.training_frames = training_frames
        self.epsilon = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_cnt = 0
        self.update_type = update_type
        self.tau = soft_update_tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_start = update_start_buffer_size
        self.seed = rand_seed
        self.plot_option = plot_option
        
        self.q_behave = QNetwork(self.input_dim, self.action_dim).to(self.device)
        self.q_target = QNetwork(self.input_dim, self.action_dim).to(self.device)
        if trained_model_path: # load a trained model if existing
            self.q_behave.load_state_dict(torch.load(trained_model_path))
            print("Trained model is loaded successfully.")
        
        # Initialize target network parameters with behavior network parameters
        self.q_target.load_state_dict(self.q_behave.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_behave.parameters(), lr=learning_rate) 

        self.memory = ReplayBuffer(self.buffer_size, self.input_dim, self.batch_size)

    def select_action(self, state: 'Must be pre-processed in the same way as updating current Q network. See def _compute_loss'):
        
        if np.random.random() < self.epsilon:
            return np.zeros(self.action_dim), np.random.randint(self.action_dim)
        else:
            # with no_grad to compute faster
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device) / 255
                Qs = self.q_behave(state)
                # take an action of a maximum Q-value
                action = Qs.argmax()
            
            # return action and Q-values (Q-values are not required for implementing algorithms. This is just for checking Q-values for each state. Not must-needed)  
            return Qs.detach().cpu().numpy(), action.detach().item()  

    def get_init_state(self):

        init_state = self.env.reset()
        for _ in range(0): # loop for a random initial starting point. range(0) means the same initial point.
            action = np.random.randint(self.action_dim)
            init_state, _, _, _ = self.env.step(action) 
        return init_state.flatten()

    def get_state(self, action):

        next_state, reward, done, _ = self.env.step(action)
        return reward, next_state.flatten(), done

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def update_behavior_q_net(self):
        # update behavior q network with a batch
        batch = self.memory.batch_load()
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def target_soft_update(self):
        ''' target network is updated with Soft Update. tau is a hyperparameter for the updating ratio betweeen target and behavior network  '''
        for target_param, current_param in zip(self.q_target.parameters(), self.q_behave.parameters()):
            target_param.data.copy_(self.tau*current_param.data + (1.0-self.tau)*target_param.data)

    def target_hard_update(self):
        ''' target network is updated with Hard Update '''
        self.update_cnt = (self.update_cnt+1) % self.target_update_freq
        if self.update_cnt==0:
            self.q_target.load_state_dict(self.q_behave.state_dict())

    def train(self):
        tic = time.time()
        losses = []
        scores = []
        epsilons = []
        avg_scores = [[-10000]] # As an initial score, set an arbitrary score of an episode.

        score = 0

        print("Storing initial buffer..") 
        state = self.get_init_state()
        for frame_idx in range(1, self.update_start+1):
            # Store transitions into the buffer until the number of 'self.update_start' transitions is stored 
            _, action = self.select_action(state)
            reward, next_state, done = self.get_state(action)
            self.store(state, action, reward, next_state, done)
            state = next_state
            if done: state = self.get_init_state()

        print("Done. Start learning..")
        history_store = []
        for frame_idx in range(1, self.training_frames+1):
            Qs, action = self.select_action(state)
            reward, next_state, done = self.get_state(action)
            self.store(state, action, reward, next_state, done)
            history_store.append([state, Qs, action, reward, next_state, done]) # history_store is for checking an episode later. Not must-needed.
            loss = self.update_behavior_q_net()

            if self.update_type=='hard':   self.target_hard_update()
            elif self.update_type=='soft': self.target_soft_update()
            
            score += reward
            losses.append(loss)

            if done:
                # For saving and plotting when an episode is done.
                scores.append(score)
                if np.mean(scores[-10:]) > max(avg_scores):
                    torch.save(self.q_behave.state_dict(), self.model_path+'{}_Score:{}.pt'.format(frame_idx, np.mean(scores[-10:])))
                    training_time = round((time.time()-tic)/3600, 1)
                    np.save(self.model_path+'{}_history_Score_{}_{}hrs.npy'.format(frame_idx, score, training_time), np.array(history_store, dtype=object))
                    print("          | Model saved. Recent scores: {}, Training time: {}hrs".format(scores[-10:], training_time), ' /'.join(os.getcwd().split('/')[-3:]))
                avg_scores.append(np.mean(scores[-10:]))

                if self.plot_option=='inline': 
                    scores.append(score)
                    epsilons.append(self.epsilon)
                    self._plot(frame_idx, scores, losses, epsilons)
                elif self.plot_option=='wandb': 
                    wandb.log({'Score': score, 'loss(10 frames avg)': np.mean(losses[-10:]), 'Epsilon': self.epsilon})
                    print(score, end='\r')
                else: 
                    print(score, end='\r')

                score=0
                state = self.get_init_state()
                history_store = []
            else: state = next_state

            self._epsilon_step()

        print("Total training time: {}(hrs)".format((time.time()-tic)/3600))

    def _epsilon_step(self):
        self.epsilon = max(self.epsilon-self.eps_decay, 0.1)

    def _compute_loss(self, batch: "Dictionary (S, A, R', S', Dones)"):
        states = torch.FloatTensor(batch['states']).to(self.device) / 255
        next_states = torch.FloatTensor(batch['next_states']).to(self.device) / 255
        actions = torch.LongTensor(batch['actions'].reshape(-1, 1)).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(self.device)
        dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(self.device)

        current_q = self.q_behave(states).gather(1, actions)

        # target value
        next_q = self.q_target(next_states).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - dones
        target = (rewards + (mask * self.gamma * next_q)).to(self.device)

        # Use smooth l1 loss for clipping loss between -1 to 1 as in DQN paper.
        loss = F.smooth_l1_loss(current_q, target)
        return loss

    def eval(self):
        tic = time.time()

        trained_model_path = "/home/iqsl/IQSL_Projects/JungKH/Molding/model_save/Test/91513_Score:158.1400000000056.pt"
        self.q_behave.load_state_dict(torch.load(trained_model_path))
        self.q_behave.eval()
        print("Trained model is loaded successfully.")

        score = 0
        frame_idx = 0

        history_store = []
        state = self.get_init_state()
        with torch.no_grad():
            while 1:
                frame_idx += 1
                Qs, action = self.select_action(state)
                reward, next_state, done = self.get_state(action)
                history_store.append([state.reshape(20, 20), Qs, action, reward, next_state.reshape(20, 20), done]) # history_store is for checking an episode later. Not must-needed.
                score += reward

                if done:
                    history_store = np.array(history_store, dtype=object)
                    np.save("eval_history_saved.npy", history_store)
                    break
                else: state = next_state

        print("Total time: {}(min)".format((time.time()-tic)/60))


    def _plot(self, frame_idx, scores, losses, epsilons):
        clear_output(True) 
        plt.figure(figsize=(20, 5), facecolor='w') 
        plt.subplot(131)  
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores) 
        plt.subplot(132) 
        plt.title('loss') 
        plt.plot(losses) 
        plt.subplot(133) 
        plt.title('epsilons')
        plt.plot(epsilons) 
        plt.show() 
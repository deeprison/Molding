import os
import gym    
import numpy as np

from datetime import datetime
from agent import Agent
import wandb   

env_list = {
    0: "CartPole-v0",
    1: "CartPole-v2",
    2: "LunarLander-v2",
    3: "Breakout-v4",
    4: "BreakoutDeterministic-v4",
    5: "BreakoutNoFrameskip-v4",
    6: "BoxingDeterministic-v4",
    7: "PongDeterministic-v4",
}
env_name = env_list[6]
env = gym.make(env_name)
input_dim = 84
input_frame = 4
print("env_name", env_name) 
print(env.unwrapped.get_action_meanings(), env.action_space.n)

update_start_buffer_size = 20000
tot_train_frames = 10000000

gamma = 0.99
buffer_size = int(8e5) 
batch_size = 32           
update_type = 'hard'
soft_update_tau = 0.002
learning_rate = 0.00025
target_update_freq = 8000
current_update_freq = 4 # Update frequency of current Q-Network.  
skipped_frame = 0

device_num = 1
rand_seed = None
rand_name = ('').join(map(str, np.random.randint(10, size=(3,))))
folder_name = os.getcwd().split('/')[-1] 

# NoisyNet Variable
initial_std = 0.5 

# Variables for Multi-step TD
n_step = 3

# Variables for Categorical RL
n_atoms = 51
Vmax = 10
Vmin = -10

model_number = 0
main_path = './model_save/'
now = datetime.now()
rand_name = f'{now.date()} {now.hour}:{now.minute}'
folder_name = os.getcwd().split('/')[-1] 

model_name = f'{env_name}_{rand_name}'
model_save_path = f'./model_save/{model_name}/'
if not os.path.exists('./model_save/'):
    os.mkdir('./model_save/')
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
print("model_save_path:", model_save_path)

plot_options = {0: 'wandb', 1: 'inline', 2: False} 
plot_option = plot_options[2]

''' wandb is one of visualizing tools ''' 
if plot_option=='wandb':
    project_name = 'rainbow-without-per'    
    os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
    wandb.init(
            entity="molding_rl",
            project="Noisy_Multi-2dim",
            name=f"{rand_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_frame": input_frame,
                    "input_dim": input_dim,
                    "initial_std (NoisyNet param)": initial_std,
                    "total_training_frames": tot_train_frames,
                    "skipped_frame": skipped_frame,
                    "gamma": gamma,
                    "n_step (Multi-step param)": n_step,
                    "buffer_size": buffer_size,
                    "update_start_buffer_size": update_start_buffer_size,
                    "batch_size": batch_size,
                    "update_type": update_type,
                    "soft_update_tau": soft_update_tau,
                    "learning_rate": learning_rate,
                    "target_update_freq (unit:frames)": target_update_freq,
                    "n_atoms (C51 param)": n_atoms,
                    "Vmax (C51 param)": Vmax,
                    "Vmin (C51 param)": Vmin
                    }
            )
            
trained_model_path = ''
agent = Agent( 
    env,
    input_frame,
    input_dim,
    initial_std,
    tot_train_frames,
    skipped_frame,
    gamma,
    n_step,
    target_update_freq,
    current_update_freq,  
    update_type,
    soft_update_tau,
    batch_size,
    buffer_size,
    update_start_buffer_size,
    learning_rate,
    device_num,
    rand_seed,
    plot_option,
    model_save_path,
    trained_model_path,
    n_atoms,
    Vmax,
    Vmin
) 

agent.train()
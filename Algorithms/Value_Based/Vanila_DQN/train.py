from agent import Agent
from datetime import datetime

import wandb   
import os, sys
sys.path.insert(0, "../../..")
sys.path.insert(0, ".")
from env import Env
env = Env()

env_name = "Mold_ENV"
input_dim = 84
input_frame = 4

update_start_buffer_size = 6000
num_frames = 600000
eps_max = 1.0
eps_min = 0.1
eps_decay = 1/100000
gamma = 0.99

buffer_size = int(3e4) 
batch_size = 32           
update_type = 'hard'
soft_update_tau = 0.002
learning_rate = 0.001
target_update_freq = 150
skipped_frame = 0

device_num = 0
rand_seed = None
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

trained_model_path = ''

plot_options = {0: 'wandb', 1: 'inline', 2: False} 
plot_option = plot_options[0]

''' wandb is one of visualizing tools ''' 
if plot_option=='wandb':
    os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
    wandb.init(
            entity="molding_rl",
            project="Vanila_DQN-2dim",
            name=f"{rand_name}_{folder_name}_{env_name}",
            config={"env_name": env_name, 
                    "input_frame": input_frame,
                    "input_dim": input_dim,
                    "eps_max": eps_max,
                    "eps_min": eps_min,
                    "eps_decay": eps_decay,
                    "num_frames": num_frames,
                    "skipped_frame": skipped_frame,
                    "gamma": gamma,
                    "buffer_size": buffer_size,
                    "update_start_buffer_size": update_start_buffer_size,
                    "batch_size": batch_size,
                    "update_type": update_type,
                    "soft_update_tau": soft_update_tau,
                    "learning_rate": learning_rate,
                    "target_update_freq (unit:frames)": target_update_freq,
                    }
            )

agent = Agent( 
    env,
    input_frame,
    input_dim,
    num_frames,
    skipped_frame,
    eps_decay,
    gamma,
    target_update_freq,
    update_type,
    soft_update_tau,
    batch_size,
    buffer_size,
    update_start_buffer_size,
    learning_rate,
    eps_min,
    eps_max,
    device_num,
    rand_seed,
    plot_option,
    model_save_path,
    trained_model_path
) 

agent.train()
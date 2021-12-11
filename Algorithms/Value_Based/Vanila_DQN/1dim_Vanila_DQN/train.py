import numpy as np
import gym    
import os

from agent import Agent
from datetime import datetime
import wandb   
import sys
sys.path.insert(0, "../../..")
sys.path.insert(0, ".")
from env import Env
env = Env()

# env_list = {
#     0: "CartPole-v0",
#     1: "CartPole-v2",
#     2: "LunarLander-v2",
# }

# env_name = env_list[0]
# env = gym.make(env_name)

input_dim = 400
env_name = "Mold_ENV"
update_start_buffer_size = 1000
training_frames = 100000
eps_max = 1.0
eps_min = 0.1
eps_decay = 1/5000
gamma = 0.99

buffer_size = int(5e3) 
batch_size = 32           
update_type = 'hard'
soft_update_tau = 0.002
learning_rate = 0.001
target_update_freq = 100

device_num = 0
rand_seed = None
eval_mode = False

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

agent = Agent( 
    env,
    input_dim,
    training_frames,
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

if eval_mode:
    agent.eval()
else:
    ''' wandb is one of visualizing tools ''' 
    if plot_option=='wandb':
        os.environ['WANDB_NOTEBOOK_NAME'] = 'RL_experiment'
        wandb.init(
                entity="molding_rl",
                project="Vanila_DQN-1dim",
                name=f"{rand_name}_{env_name}",
                config={"env_name": env_name, 
                        "input_dim": input_dim,
                        "eps_max": eps_max,
                        "eps_min": eps_min,
                        "eps_decay": eps_decay,
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

    agent.train()
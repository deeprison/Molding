from env import Env
import numpy as np
from copy import copy, deepcopy

from agent import Agent

img = np.load('../data/square/npy/extreme_5x5.npy')
env = Env([img], on_direction=False)

epochs = 99999
agent = Agent()
tau = 1
simulation_times = 5

for epoch in range(1, epochs+1):

    state = env.reset()
    action_mask = env.get_action_mask()
    done = False

    score = 0

    while not done:

        action, pi, value, values = agent.choose_action(state, action_mask, tau, simulation_times, env)

        # if agent.buffer != None:
        #     agent.buffer.commit_stmemory(state, pi)
        
        next_state, reward, done, _ = env.step(action)
        action_mask = env.get_action_mask()
        state = deepcopy(next_state)

        score += reward
    
    # if agent.buffer != None:
    #     for move in agent.buffer.stmemory:
    #         move['value'] = reward

    # agent.buffer.commit_ltmemory()
    # agent.buffer.clear_stmemory()

    # if (len(agent.buffer.ltmemory) >= 128):
    #     hist = agent.train()
    #     print(hist)

    print('epoch:', epoch, 'score:', score, 'tree size: ', len(agent.mcg.tree))
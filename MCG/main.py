from env import Env
import numpy as np

from agent import Agent

img = np.load('../data/square/npy/extreme_5x5.npy')
env = Env([img], on_direction=False)

epochs = 100
agent = Agent()
tau = 1
simulation_times = 10

for epoch in range(1, epochs+1):

    obs = env.reset()
    state = obs
    action_mask = env.get_action_mask()
    done = False

    score = 0

    while not done:

        env.render()
        action, pi, value, values = agent.choose_action(state, action_mask, tau, simulation_times, env)

        if agent.buffer != None:
            agent.buffer.commit_stmemory(state, pi)
        
        next_obs, reward, done, _ = env.step(action)
        action_mask = env.get_action_mask()
        next_state = next_obs

        score += reward
    
    if agent.buffer != None:
        for move in agent.buffer.stmemory:
            move['value'] = reward

    print('epoch:', epoch, 'score:', score)
import gym
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
os.chdir('/home/lamsadeuser/resnets_atari')

from utils import select_action
from ImageAtari import wrap_dqn

def Q_val(model):
    """Plot the Q_value for each action while playing one game
    You have to load a model first and pass it as an argument to the function
    
    Example 
    model = torch.load('model_atari_Seaquest_dqn.pkl')
    Q_val(model)"""
    game = input("What game ? ")
    env = gym.make(game + "NoFrameskip-v4")
    env.reset()
    env = wrap_dqn(env,True)
    x = env.action_space
    for i in range(18,0,-1):
        if x.contains(i):
            nb_actions = (i+1)
            break
    Ql = [[] for _ in range(nb_actions)]
    env.render()
    state = env.reset()
    state = torch.from_numpy(np.array(state))
    state = state.transpose(0,2).transpose(1,2).unsqueeze(0)
    nb_frames = 18000
    done = False
    while not done and nb_frames > 0:
        env.render()
        action=select_action(state, 0, model, 6)
        state, reward, done, _ = env.step(action[0,0])
        state = torch.from_numpy(np.array(state))
        state = state.transpose(0,2).transpose(1,2).unsqueeze(0)
        Q = model(Variable(state))
        for i,L in enumerate(Ql):
            inpu = Q.data.squeeze()[i]
            L.append(inpu)
            plt.plot(L)
        plt.show()
        nb_frames-=1

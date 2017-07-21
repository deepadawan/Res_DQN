import gym
import torch.optim as optim
import torch

from ImageAtari import SimpleMonitor, wrap_dqn
from utils import ReplayMemory

use_cuda = torch.cuda.is_available()

def init(game_name,momentum,lr,model,mem_size,mod):
    """Set some parameters for the training:

        Environment of the game and more:
        Environment
        Number of actions
        Optimizer (RMSprop for resnet and Adam for classic DeepMind model
        Replay Buffer
        Model Architecture object
    """
    env = gym.make(game_name + "NoFrameskip-v4")
    x = env.action_space
    for i in range(18,0,-1):
        if x.contains(i):
            nb_actions = (i+1)
            break
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env,True)
    env_no_clipping = wrap_dqn(monitored_env,False)

    model_obj = model(nb_actions)
    if use_cuda:
        model_obj = model_obj.cuda()

    if mod=='resnet':
        optimizer = optim.RMSprop(model_obj.parameters(), lr = lr,
                          momentum = momentum)
    elif mod=='dqn':
        optimizer = optim.Adam(model_obj.parameters(), lr=lr)
    

    memory = ReplayMemory(mem_size)



    return env, env_no_clipping, nb_actions, optimizer, memory, model_obj

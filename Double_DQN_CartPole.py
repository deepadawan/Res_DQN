#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:55:53 2017

@author: lamsadeuser
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:06:17 2017

@author: lamsadeuser
"""

import gym
import math
import random
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped #Set environment, why unwrap ?

############################# REPLAY MEMORy ? ###################################
#We now define Replay memory, same thing for Atari programs ?????????????? 

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity   # % = modulo
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
################################## Neural Network #############################
#Neural Network = universal function approximators


class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=5,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        #16*18*38
        self.conv2 = nn.Conv2d(16,32, kernel_size=5,stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        #32*7*17
        self.conv3 = nn.Conv2d(32,32, kernel_size=5,stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        #32*2*7 = 448
        self.head = nn.Linear(448,2)
        
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0),-1))
    
############################# Image Processing ? ################################
resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
screen_width = 600
#SIZE = 3*40*80

def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(torch.Tensor)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()


############################# HYPERPARAMETERS #################################
BATCH_SIZE = 8
GAMMA = 0.999
EPS_START = 0.9   #epsilon greedy policy, EXPLORATION/EXPLOITATION
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(10000)

#EPSILON-GREEDY : Choose a random action with probability epsilon


steps_done = 0


def select_action(state):
    global steps_done               #Outside function variable
    sample = random.random()        #Uniform [0,1] sample
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #Definition of the decay of epsilon greedy policy
    
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1]
            #~~~~~~ Take maximum action of next state
    else:
        return torch.LongTensor([[random.randrange(2)]])
        #Choose a random action between left and right


########################## Plot ? ############################################
############################# TRAINING ########################################
last_sync = 0
model_0 = model
def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:   
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))


    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.Tensor))
    arg_max = model(non_final_next_states).max(1)[1]
    next_state_values[non_final_mask] = model_0(non_final_next_states)[arg_max.data]

    next_state_values.volatile = False

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)


    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
        
        
num_episodes = 100
counter=0
for i_episode in range(num_episodes):
    print(i_episode)
    env.reset()
    last_screen = get_screen()                  #??????????????
    current_screen = get_screen()               #??????????????
    state = current_screen - last_screen        #??????????????
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action[0, 0])
        reward = torch.Tensor([reward])


        last_screen = current_screen            #??????????????
        current_screen = get_screen()           #??????????????
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        counter+=1
        if counter%20==0:
            model_0=model

print('Complete')
env.render(close=True)
env.close()
plt.ioff()
plt.show()
        
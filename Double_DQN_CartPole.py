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
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped 
############################# REPLAY MEMORY: OK ###############################
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

#??????????????????????????????????????????????????????????????????????????????
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity   
 #?????????????????????????????????????????????????????????????????????????????   
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
    
################################## Neural Network: OK #########################


class DDQN(nn.Module):
    def __init__(self):
        super(DDQN,self).__init__()
        #3*40*80
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
    
    
    
############################# Image Processing ? ##############################
#??????????????????????????????????????????????????????????????????????????????
resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
screen_width = 600

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
#??????????????????????????????????????????????????????????????????????????????
############################# HYPERPARAMETERS: OK #############################


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1   
EPS_END = 0.1
i_END = 50
iter_param = 200
num_episodes = 500
mem_size = 10000
lr = 1e-3
momentum = 0.9
model_beta = DDQN()
optimizer = optim.RMSprop(model_beta.parameters(), lr = lr, momentum = momentum)


############################# TRAINING ########################################



def select_action(state, eps_threshold):           
    sample = random.random()        
    if sample > eps_threshold:
        return model_beta(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1]
    else:
        return torch.LongTensor([[random.randrange(2)]])
    
    

def optimize_model():
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
    state_action_values = model_beta(state_batch).gather(1, action_batch)
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.Tensor))
    arg_max = model_beta(non_final_next_states).max(1)[1].squeeze()
    next_state_values[non_final_mask] = torch.index_select(model_beta_0(non_final_next_states), dim = 1, index = arg_max).diag()
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    #print(torch.mean(model_beta.parameters()))
    for param in model_beta.parameters():
        vector.append(param)
        param.grad.data.clamp_(-1, 1)  
    optimizer.step()
    
    
    
def eval_model():
    env.reset()
    last_screen = get_screen()                         
    current_screen = get_screen()           
    state = current_screen - last_screen  
    done=False
    score=0
    while not done:
        action=select_action(state, 0)
        _, reward, done, _ = env.step(action[0,0])
        score += reward
        last_screen = current_screen                            
        current_screen = get_screen()  
        state = current_screen - last_screen
    return score



counter_iter_param=0
memory = ReplayMemory(mem_size)
episode_durations = []
episode_duration_full_greedy = []
y_abs_0 = []
y_abs_1=[]
plot_Q_value = 1
plot_durations = 1
model_beta_0 = deepcopy(model_beta)
for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen()              #??               
    current_screen = get_screen()           #?? 
    state = current_screen - last_screen    #?? 
    tot_reward = 0
    done=False
    if i_episode <i_END:
        eps_threshold = i_episode*(EPS_END-EPS_START)/i_END + EPS_START
    else:
        eps_threshold = EPS_END
    while not done:
        action = select_action(state,eps_threshold)
        _, reward, done, _ = env.step(action[0, 0])         
        tot_reward += reward
        reward = torch.Tensor([reward])
        last_screen = current_screen                        #?    
        current_screen = get_screen()                       #?
        if not done:           
            next_state = current_screen - last_screen       #?
        else:
            next_state = None
        memory.push(state, action, next_state, reward)
        if i_episode%plot_Q_value==0:
            temp = model_beta(Variable(state))
            y_abs_0.append(temp.data.squeeze()[0])
            y_abs_1.append(temp.data.squeeze()[1])
        state = next_state
        optimize_model()
        counter_iter_param+=1
        if counter_iter_param%iter_param==0:
            model_beta_0=deepcopy(model_beta)
    episode_durations.append(tot_reward)
    print('Episode duration at epoch ', i_episode, ' ', episode_durations[-50:])
    episode_duration_full_greedy.append(eval_model())
    if i_episode%plot_Q_value==0:
        plt.plot(y_abs_0)
        plt.plot(y_abs_1)
        plt.title(str(i_episode))
        plt.xlabel('Number of Updates')
        plt.ylabel('Q_value')
        plt.legend(['Left','Right'])
        plt.show()
        y_abs_0 = []
        y_abs_1=[] 
    if i_episode%plot_durations==0:
        plt.plot(episode_durations)
        plt.plot(episode_duration_full_greedy)
        plt.xlabel('Number of episodes')
        plt.ylabel('Total Reward')
        plt.legend(['e_greedy','full_greedy'])
        plt.show()
print('Training Complete')


#Clamp Gradient
#Memory -> Capacity
#get-screen


        

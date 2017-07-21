import torch
import torch.nn as nn
import gym
import torch.optim as optim
import numpy as np 
import sys
from torch.autograd import Variable
from copy import deepcopy

from NetworkAtari import Atari_2600
from ResNet_18_Atari import resnet
from ImageAtari import SimpleMonitor, wrap_dqn
from utils import ReplayMemory, select_action
from optim import optimize_model

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor



def pred(game_name,network):
    """Training of an agent that learn if the next reward is +1, -1 or 0 only
    by updating the parameters of the fully connected layers.
    We take a model previously trained to play the game, then freeze the 
    parameters of the convolutions and re-initialize the parameters of the 
    fully connected layers"""
    
    model0 = torch.load("model_atari_repr2_{}_{}.pkl".format(game_name,network))
    
    
    env = gym.make(game_name + "NoFrameskip-v4")
    x = env.action_space
    for i in range(18,0,-1):
        if x.contains(i):
            nb_actions = (i+1)
            break
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env,True)
    
    if network=="dqn":
        model = deepcopy(model0)
        
        for i,param in enumerate(model.parameters()):
                param.requires_grad = False
        model.linear = nn.Linear(3136,512)
        model.head = nn.Linear(512,3)
        D_parameters = [
            {'params': model.linear.parameters()},
            {'params': model.head.parameters()}
        ]
        optimizer = optim.Adam(D_parameters, lr=0.00025)
    elif network=="resnet":
        model = deepcopy(model0)
        model = model.cuda()
        for i,param in enumerate(model.parameters()):
                param.requires_grad = False
        model.fc = nn.Linear(512, 3).cuda()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.00025)
    else:
        print("UNKNOWN MODEL")
    
    
    
    memory0_train = ReplayMemory(10000)
    memory1_train = ReplayMemory(10000)
    memory0_test = ReplayMemory(2000)
    memory1_test = ReplayMemory(2000)
    
    done = False
    plot = []
    counter_iter_param=1
    for i_episode in range(1,int(1e6)):
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()
        state = state.transpose(0,2).transpose(1,2).unsqueeze(0)
        if use_cuda:
            state = state.cuda()
            
        tot_reward = 0
        step=0
        ended = done=bb =False
        while (not done) and (not ended):
            #env.render()
            action = select_action(state,0,model0,nb_actions)
            next_state, reward, done, _ = env.step(action[0, 0])
            next_state = torch.from_numpy(np.array(next_state)).float()
            next_state = next_state.transpose(0,2).transpose(1,2).unsqueeze(0)
            if use_cuda:
                next_state = next_state.cuda()
            tot_reward += reward
            reward = Tensor([reward])
            
            if counter_iter_param%5==1:
                if reward[0]==0:
                    memory0_test.push(state, action, next_state, reward)
                else:
                    memory1_test.push(state,action,next_state,reward)
            else:
                if reward[0]==0:
                    memory0_train.push(state, action, next_state, reward)
                else:
                    memory1_train.push(state,action,next_state,reward)
                if len(memory0_train)>32 and len(memory1_train) >32:
                    bb=True 
                    loss=optimize_model(32,_,model,memory0_train,_, optimizer
                                        ,memory1_train,False)
            
                 
            ended = (counter_iter_param >= 60000000000)
            step += 1
            done = done or (step >= 18000)
                    
            state = next_state        
            counter_iter_param +=1
            
        if bb:
            print(loss,"loss")
        if i_episode%50==0:
            total =0
            correct = 0
            for data in memory0_test.memory+memory1_test.memory:
                out = model(Variable(data[0], volatile=True).type(FloatTensor))
                prediction = torch.max(out.data,1)[1]
                correct += (prediction.squeeze(0)==data[3].type(LongTensor))[0]
                total +=1
            result=100*correct/total 
            plot.append(result)
            print(plot,"result")
         
            
            
if __name__=='__main__':
    if use_cuda:
       print("Using CUDA")
    else:
        print("WARNING: Cuda not available")
    pred(sys.argv[1],sys.argv[2])

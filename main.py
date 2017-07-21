import torch
import sys
import os
from copy import deepcopy
import numpy as np
import time

from NetworkAtari import Atari_2600
from utils import select_action, save,  eval_model
from init_atari import init
from optim import optimize_model

from ResNet_18_Atari import resnet

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor

def training_Atari(game_name, model, GAMMA=0.99, EPS_START = 1, EPS_END = 0.01,
                   iter_param = 1e4, max_frame = 50000000,
                   mem_size = int(1e4), momentum = 0.95):
    """
        ** Arguments
    Model                           Deep Learning architecture Resnet18 (resnet)
                                    or classic DeepMind model (Atari_2600)
    game_name                       Name of the game
    BATCH_SIZE                      Size of the batch
    GAMMA                           Discount factor
    EPS_START                       Initial value in epsilon greedy exploration
    EPS_END                         Final value in epsilon greedy exploration
    iter_END                        nb of frames over wich epsilon is linearly
                                    annealed to its final value
    iter_param                      frequency with which we update target network
    max_frame                       Maximum number of training frames
    mem_size                        Size of the Replay memory
    lr                              Learning rate
    momentum                        Momentum
    
    This algorithm is used for two different architectures : Resnet and more 
    classical CNN like DeepMind models for Atari
    """

    start = time.time()
    
    if model == resnet:
        mod = 'resnet'
        lr = 2.5e-4
        BATCH_SIZE=64
        save_test = 50
        iter_END = 100000
        
    elif model==Atari_2600:
        mod='dqn'
        lr = 1e-4
        BATCH_SIZE=256
        save_test=200
        iter_END = 1000000
        
    recover = input("Do you want to resume a previous training ? (y/n) ").lower()
    controller = False
    while not controller:
        if recover=="n":
            #If you don't load a previous training it loads a model with
            #random paramaters
            env, env_no_clipping, nb_actions, optimizer, memory, model_atari = \
                    init(game_name,momentum,lr,model,mem_size,mod)
            controller=True
            counter_iter_param=1
            session=0
            
        elif recover=="y":
            env, env_no_clipping, nb_actions, optimizer, memory, _ = \
                init(game_name,momentum,lr,model,mem_size,mod)
            name = input("Please enter the name of the file you want to load: ")
            if os.path.exists(name):
                if use_cuda:
                    model_atari = torch.load(name)
                else:
                    model_atari=torch.load(name, map_location=lambda storage, 
                                           loc: storage)
                controller=True
                counter_iter_param = int(input("""Please enter the number of 
                                frames viewed in the last session: """))
                session = int(input("Please enter number session: "))
            else:
                print("This file doesn't exist")
    
    reward_list = []
    eval_reward_list = []
    model_atari_0 = deepcopy(model_atari)
    eps_threshold = 1
    for i_episode in range(1,int(1e6)):

        print((i_episode,game_name,model_atari.name,eps_threshold))
        
        #We reset environment and make appropriate transformation on raw pixel
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()
        state = state.transpose(0,2).transpose(1,2).unsqueeze(0)
        if use_cuda:
            state = state.cuda()

        tot_reward = 0
        step = 0
        ended = done =False
        while (not done) and (not ended):
            #Update of the epsilon
            if counter_iter_param <iter_END:
                eps_threshold = counter_iter_param*(EPS_END-EPS_START)/iter_END + EPS_START
            else:
                if counter_iter_param == iter_END:
                    #Save the game number where its ends the annihilation of epsilon
                    save(i_episode,"end_epsilon_{}_{}.txt".format(game_name,model_atari.name))
                eps_threshold = EPS_END

            #We Select an action according to epsilon-greedy policy
            action = select_action(state,eps_threshold,model_atari_0,nb_actions)
            next_state, reward, done, _ = env.step(action[0, 0])
            next_state = torch.from_numpy(np.array(next_state)).float()
            next_state = next_state.transpose(0,2).transpose(1,2).unsqueeze(0)
            if torch.cuda.is_available():
                next_state = next_state.cuda()
            tot_reward += reward
            reward = Tensor([reward])
            memory.push(state, action, next_state, reward)

            state = next_state
            optimize_model(BATCH_SIZE,model_atari_0, model_atari,memory,
                           GAMMA, optimizer,_,True)

            #Update of the model
            if counter_iter_param%iter_param==0:
                model_atari_0=deepcopy(model_atari)
            counter_iter_param+=1
        
            ended = (counter_iter_param >= max_frame)
            step += 1
            done = done or (step >= 18000)
        reward_list.append((counter_iter_param,i_episode,tot_reward))
       
        
        if i_episode%save_test==0 or ended:
            evaluation = eval_model(env_no_clipping,model_atari,nb_actions)
            eval_reward_list.append((counter_iter_param,i_episode,evaluation))
            print("Saved")
            #We save data in different files that can be load after training
            save(eval_reward_list,"evaluation_model_{}_{}.txt".format(game_name,model_atari.name))
            save(reward_list,"current_results_{}_{}.txt".format(game_name,model_atari.name))
            torch.save(model_atari,"model_atari_repr{}_{}_{}.pkl".format(session,game_name,model_atari.name))
        if ended:
            break

    
    end = time.time()
    t = end-start
    save(t,"run_time_{}_{}.txt".format(game_name,model_atari.name))
    return 'Fin'




if __name__ == '__main__':
    if use_cuda:
       print("Using CUDA")
    else:
        print("WARNING: Cuda not available")

    assert len(sys.argv) > 2, "Please give the name of the game as argument"
    model = sys.argv[2]
    if model == "resnet":
        training_Atari(sys.argv[1], model = resnet)
    elif model == "dqn":
        training_Atari(sys.argv[1], model = Atari_2600)
    else:
        assert False, "Unknown model"
        

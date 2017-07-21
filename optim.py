import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import clip_by_norm, Transition

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

def optimize_model(BATCH_SIZE,model_atari_0, model_atari,memory, GAMMA, optimizer
                   ,memory1,mod):
    """Do one updtate of the model you are training
    If you are training a classic training to learn how to play one of the
    Atari's game set mod to True
    
    If you are training a model that predict if the next reward will be 
    positive, negative of zero set mod to False
    """
    if len(memory) < BATCH_SIZE:
        #We do not update until me can fill a BATCH
        return

    #Take a sample in the memory buffer
    if mod:
        transitions = memory.sample(BATCH_SIZE)
    else:
        transitions = memory.sample(BATCH_SIZE//2) + memory1.sample(BATCH_SIZE//2) 

    batch = Transition(*zip(*transitions))

    #Create a mask where 0 stands for final state, 1 otherwise
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    #Take only the non final states:
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    
    if mod:
        state_action_values = model_atari(state_batch).gather(1, action_batch)

        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
        arg_max = model_atari(non_final_next_states).max(1)[1].squeeze()
    
        next_state_values[non_final_mask] =  \
            torch.index_select(model_atari_0(non_final_next_states),
                           dim = 1, index = arg_max).diag()
        next_state_values.volatile = False
    
        #Define Target of the DDQN Network
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        #Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    else:
        expected_reward = model_atari(state_batch)
        for i in reward_batch:
            if i.data[0]==-1:
                i.data[0] = 0
            elif i.data[0] == 0:
                i.data[0] = 1
            else:
                i.data[0]=2
        reward_batch = reward_batch.type(LongTensor)
        loss = F.cross_entropy(expected_reward,reward_batch)
        
    #Gradients Computation
    optimizer.zero_grad()
    loss.backward()

    #We Normalize too big gradients
    if False:
        for param in model_atari.parameters():
            param.grad.data = clip_by_norm(param.grad.data,10)

    #Update of the Parameters
    optimizer.step()
    
    return loss

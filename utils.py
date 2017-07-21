from collections import namedtuple
import random
from torch.autograd import Variable
import torch
import pickle
import numpy as np


Transition = namedtuple('Transition',('state','action','next_state','reward'))
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


class ReplayMemory(object):
    """Genereic Class for creating Replay Memory"""
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def select_action(state, eps_threshold,model,nb_action):
    """Choose an action according to eps_threshold-greedy policy"""
    sample = random.random()
    if sample > eps_threshold:
        out = model(Variable(state, volatile=True).type(FloatTensor))
        return out.data.max(1)[1]
    else:
        return LongTensor([[random.randrange(nb_action)]])


def clip_by_norm(t,clip_norm):
    if torch.norm(t)<clip_norm:
         return t
    else:
        return clip_norm*t/torch.norm(t)


def save(contenu,name):
    """Save data (named contenu) in a file (named name)"""
    with open(name, "wb") as fichier:
        mon_pickler = pickle.Pickler(fichier)
        mon_pickler.dump(contenu)
        fichier.close()


def eval_model(env,model,nb_actions):
    """Evaluation of the model for one game with 0.01-greedy exploration"""
    state = env.reset()
    state = torch.from_numpy(np.array(state))
    state = state.transpose(0,2).transpose(1,2).unsqueeze(0)
    if use_cuda:
        state = state.cuda()
    done=False
    score=0
    nb_frames = 18000
    while not done and nb_frames > 0:
        action=select_action(state, 0.01, model, nb_actions)
        state, reward, done, _ = env.step(action[0,0])
        state = torch.from_numpy(np.array(state))
        state = state.transpose(0,2).transpose(1,2).unsqueeze(0)
        if torch.cuda.is_available():
            state = state.cuda()
        score += reward
        nb_frames-=1
    return score

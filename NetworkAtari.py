import torch.nn as nn
import torch.nn.functional as F

class Atari_2600(nn.Module):
    """Reseau convolutionnel:
        taille de l'input 4*84*84
        conv 1 32*20*20
        conv 2 64*9*9
        conv 3 64*7*7
        lin1 3136 -> 512
        lin2 512 -> nb actions
        Piste d'amélioration : DropOut, BatchNorm, Dueling Network, ResNet """
    def __init__(self,nb_actions):
        super(Atari_2600,self).__init__()
        self.conv1 = nn.Conv2d(4,32, kernel_size=8,stride=4,padding=1)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4,stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3,stride=1)
        #self.bn3 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(3136,512)
        self.head = nn.Linear(512,nb_actions)
        self.name = "dqn"

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.linear(x))
        return self.head(x)

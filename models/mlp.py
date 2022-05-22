import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def weight_init(module):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(module.weight, -w, w)
        nn.init.uniform_(module.bias, -w, w)


WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4
    

N1 = 512; N2 = 512
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, N1)
        self.ln1 = nn.LayerNorm(N1)
        self.l2 = nn.Linear(N1, N2)
        self.ln2 = nn.LayerNorm(N2)
        self.l3 = nn.Linear(N2, action_dim)
        self.max_action = max_action
        
        self.apply(weight_init)
        
        '''
        fan_in_uniform_init(self.l1.weight)
        fan_in_uniform_init(self.l1.bias)
        fan_in_uniform_init(self.l2.weight)
        fan_in_uniform_init(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.l3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT) 
        '''

    def forward(self, x):
        x = self.l1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, N1)
        self.ln1 = nn.LayerNorm(N1)
        self.l2 = nn.Linear(N1, N2)
        self.ln2 = nn.LayerNorm(N2)
        self.l3 = nn.Linear(N2, 1)
        self.apply(weight_init)
    
        '''
        fan_in_uniform_init(self.l1.weight)
        fan_in_uniform_init(self.l1.bias)
        fan_in_uniform_init(self.l2.weight)
        fan_in_uniform_init(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.l3.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)
        '''
        
    def forward(self, x, u):
        x = self.l1(torch.cat([x, u], 1))
        x = self.ln1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x


    

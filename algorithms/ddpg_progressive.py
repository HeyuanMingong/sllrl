#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:12:02 2021

@author: baobao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from models.progressive_nets import PNN


    

N1 = N2 = 512
class DDPG_Progressive(object):
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99,
            tau=0.005, n_layers=3, device='cpu'):
        self.critic = PNN(n_layers, ac='critic').to(device)
        self.critic_target = PNN(n_layers, ac='critic').to(device)
        
        self.actor = PNN(n_layers, max_action=max_action, ac='actor').to(device)
        self.actor_target = PNN(n_layers, max_action=max_action, ac='actor').to(device)
    
        self.device = device
        self.tau = tau; self.gamma = gamma
        
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.lr = lr
        self.task_id = -1

    def expand_pnn(self):
        self.actor.new_task([self.state_dim, N1, N2, self.action_dim])
        print('num of actor params: ', sum(param.numel() for param in self.actor.parameters()))
        self.actor.freeze_columns(skip=[len(self.actor.columns)-1])
        self.actor_target.new_task([self.state_dim, N1, N2, self.action_dim])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.1*self.lr)
        self.actor.columns.to(self.device)
        self.actor_target.columns.to(self.device)

        self.critic.new_task([self.state_dim+self.action_dim, N1, N2, 1])
        print('num of critic params: ', sum(param.numel() for param in self.critic.parameters()))
        self.critic.freeze_columns(skip=[len(self.critic.columns)-1])
        self.critic_target.new_task([self.state_dim+self.action_dim, N1, N2, 1])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic.columns.to(self.device)
        self.critic_target.columns.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state, task_id=self.task_id)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=32):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # Compute the target Q value
        next_sa = torch.cat([next_state, self.actor_target(next_state, task_id=self.task_id)], 1)
        target_Q = self.critic_target(next_sa, task_id=self.task_id)
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        ### get current Q estimate
        sa = torch.cat([state, action], 1)
        current_Q = self.critic(sa, task_id=self.task_id)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        sa_ = torch.cat([state, self.actor(state, task_id=self.task_id)], 1)
        actor_loss = -self.critic(sa_, task_id=self.task_id).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return actor_loss, critic_loss        





























        
        
        
        

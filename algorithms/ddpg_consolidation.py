import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os, math

from models.mlp import Actor, Critic
from buffers import ReplayBuffer



class DDPG_Consolidation(object):
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99,
            tau=0.005, depth=3, decay=0.5, device='cpu'):
        self.critics, self.actors = [], []
        self.actor_params, self.critic_params = [], []
        for idx in range(depth):
            actor = Actor(state_dim, action_dim, max_action).to(device)
            critic = Critic(state_dim, action_dim).to(device)
            if idx > 0:
                actor.load_state_dict(self.actors[0].state_dict())
                critic.load_state_dict(self.critics[0].state_dict())

            self.actors.append(actor)
            self.critics.append(critic)

            self.actor_params += list(actor.parameters())
            self.critic_params += list(critic.parameters())

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target.load_state_dict(self.critics[0].state_dict())
        self.actor_target.load_state_dict(self.actors[0].state_dict())

        self.critic_optim = optim.Adam(self.critic_params, lr=lr)
        self.actor_optim = optim.Adam(self.actor_params, lr=lr)

        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.depth = depth
        self.decay = decay

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actors[0](state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=32):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(x).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        ### compute the loss of the visible critic 
        # Compute the target Q value
        target_Q = self.critic_target(next_state,self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        ### get current Q estimate
        current_Q = self.critics[0](state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        ### compute the loss of the hidden critics
        critic_kl = 0.0
        Qs = [current_Q]
        for idx in range(1, self.depth):
            Qs.append(self.critics[idx](state, action))
            critic_kl += 0.1 * (self.decay**idx) * F.mse_loss(Qs[idx-1], Qs[idx])

        critic_loss_total = critic_loss + critic_kl
        #print('critic loss: %.3f, critic kl: %.3f' % (critic_loss.data.numpy(), critic_kl.data.numpy())) 
        
        self.critic_optim.zero_grad()
        critic_loss_total.backward()
        self.critic_optim.step()

        ### Compute visible actor loss
        actor_loss = -self.critics[0](state, self.actors[0](state)).mean()
        
        ### compute the loss of hidden actors
        actor_kl = 0.0
        actions = [self.actors[0](state)]
        for idx in range(1, self.depth):
            actions.append(self.actors[idx](state))
            actor_kl += (self.decay**idx) * F.mse_loss(actions[idx-1], 
                    actions[idx])
        
        actor_loss_total = actor_loss + actor_kl
        #print('actor loss: %.3f, actor kl: %.3f' % (actor_loss.data.numpy(), actor_kl.data.numpy())) 
        
        self.actor_optim.zero_grad()
        actor_loss_total.backward()
        self.actor_optim.step()

        # Update the frozen target models
        for param, target_param in zip(self.critics[0].parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actors[0].parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss_total, critic_loss_total



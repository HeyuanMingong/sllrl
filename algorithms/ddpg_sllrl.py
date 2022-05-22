import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os, math

from models.mlp import Actor, Critic
from buffers import ReplayBuffer



class DDPG_SLLRL(object):
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99,
            tau=0.005, device='cpu'):
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.1*lr)

        self.device = device
        self.tau = tau; self.gamma = gamma

    def load_from(self, agent):
        self.actor.load_state_dict(agent.actor.state_dict())
        self.actor_target.load_state_dict(agent.actor_target.state_dict())
        self.critic.load_state_dict(agent.critic.state_dict())
        self.critic_target.load_state_dict(agent.critic_target.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=32):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(x).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        ### get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
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


    def compute_likelihood(self, transitions, sigma=0.25):
        # Sample replay buffer
        #x, y, u, r, d = replay_buffer.sample(len(replay_buffer.storage))
        S, Y, U, R, D = [], [], [], [], [] 
        for (s, y, u, r, d) in transitions:
            S.append(np.array(s, copy=False))
            Y.append(np.array(y, copy=False))
            U.append(np.array(u, copy=False))
            R.append(np.array(r, copy=False))
            D.append(np.array(d, copy=False))
        S = np.array(S)
        Y = np.array(Y)
        U = np.array(U)
        R = np.array(R).reshape(-1, 1)
        D = np.array(D).reshape(-1, 1)

        state = torch.FloatTensor(S).to(self.device)
        action = torch.FloatTensor(U).to(self.device)
        next_state = torch.FloatTensor(Y).to(self.device)
        done = torch.FloatTensor(D).to(self.device)
        reward = torch.FloatTensor(R).to(self.device)

        # Compute the target Q value
        target_Q = self.critic(next_state, self.actor(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()
        # Get current Q estimate
        current_Q = self.critic(state, action).detach()

        # Compute the mean likelihood
        a0 = torch.mul(target_Q-current_Q, target_Q-current_Q) / (sigma*sigma)
        a = -torch.clamp(a0, 0, 9)/2

        p = a.exp() / (math.sqrt(2*math.pi)*sigma)
        p_mean = p.log().mean().exp()
        return p_mean.cpu().numpy()


    def Bellman_residual(self, replay_buffer):
        # Sample replay buffer
        x, y, u, r, d = replay_buffer.sample(len(replay_buffer.storage))
        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # Compute the target Q value
        target_Q = self.critic(next_state, self.actor(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        return critic_loss.cpu().data.numpy()

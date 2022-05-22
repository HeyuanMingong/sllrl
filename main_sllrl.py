import argparse
from itertools import count
import os, sys, random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
#from tensorboardX import SummaryWriter
import time
start_time = time.time()
from tqdm import tqdm


### personal libraries
import envs
from buffers import ReplayBuffer
from algorithms.ddpg_sllrl import DDPG_SLLRL 
from crp import CRP

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Navigation2D-v1")
parser.add_argument('--output', type=str, default='output/navi_v1')
parser.add_argument('--tau',  default=0.05, type=float, help='target smoothing coefficient')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float) 
parser.add_argument('--capacity', default=10000, type=int) 
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--random_seed', default=47, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_epochs', default=200, type=int)
parser.add_argument('--max_steps', default=100, type=int) 
parser.add_argument('--update_iterations', default=10, type=int)
parser.add_argument('--num_tasks', type=int, default=50)
parser.add_argument('--pnn1', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
print(args)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
np.set_printoptions(precision=3)
print('seed = %d'%args.random_seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
if not os.path.exists(args.output): 
    os.makedirs(args.output)

env = gym.make(args.env_name).unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print('State dim: %d, action dim: %d, max action: %.3f'%(state_dim, action_dim, max_action))

### generate a sequence of 50 tasks to simulate the lifelong learning environment
tasks = np.random.uniform(-0.5, 0.5, size=(args.num_tasks, 2))


###################### Small Functions ##########################
def softmax_normalize(array, temperature=1.0):
    array = np.array(array).reshape(-1)
    array -= array.mean()
    array_exp = np.exp(array * temperature)
    array_exp /= array_exp.sum()
    return array_exp

def collect_samples(agent, env, epochs=1, random=False):
    transitions = []
    for i in range(epochs):
        state = env.reset()
        for t in range(args.max_steps):
            if random:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action_noise = np.random.normal(0, args.exploration_noise*max_action,
                        size=env.action_space.shape[0])
                action = (action + action_noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, done, info = env.step(action)
            transitions.append((state,next_state,action,reward,np.float(done)))
            state = np.copy(next_state)
            if done: break
    print('Num of collected samples: %d'%len(transitions))
    return transitions

def train_ddpg(agent, replay_buffer, env):
    rewards_on = np.zeros(args.max_epochs)
    rewards_off = np.zeros(args.max_epochs)
    coordinates = np.zeros((args.max_steps, 2))

    action_dim = env.action_space.shape[0]
    for idx in tqdm(range(args.max_epochs)):
        state = env.reset()
        ep_r = 0.0
        for t in range(args.max_steps):
            action = agent.select_action(state)
            action_noise = np.random.normal(0, args.exploration_noise*max_action, size=action_dim)
            action = (action + action_noise).clip(env.action_space.low, env.action_space.high)

            next_state, reward, done, info = env.step(action)
            ep_r += reward
            replay_buffer.push((state, next_state, action, reward, np.float(done)))

            state = next_state
            if done: break
        rewards_on[idx] = ep_r

        for it in range(args.update_iterations):
            _ = agent.update(replay_buffer, batch_size=args.batch_size)

        ### evaluate
        state = env.reset()
        ep_r = 0.0
        for t in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            ep_r += reward
            state = next_state
            if done: break
        rewards_off[idx] = ep_r

    return agent, replay_buffer, rewards_on, rewards_off




print('\n=== A Dirichlet Mixture of Robust Tabsk Models for Scalable LifeLong RL ===')
### record the task-to-cluster assignments
task_ids = np.zeros((len(tasks), 1))

### some hyper-parameters
EPOCHS = 10; N_INIT = 10; ITERS_INIT = 200; SIGMA = 0.25


replay_buffer = ReplayBuffer()
agent_init = DDPG_SLLRL(state_dim, action_dim, max_action, lr=args.lr, 
        tau=args.tau, gamma=args.gamma, device=device)

### training the agent for initialization using domain randomization
replay_buffer.reset()
for idx in range(N_INIT):
    env.reset_task(tasks[idx])
    transitions = collect_samples(agent_init, env, epochs=EPOCHS, random=True)
    replay_buffer.batch_push(transitions)

print('\n===== Training the universal model =====')
print('Buffer size: %d' % len(replay_buffer.storage))
for it in tqdm(range(ITERS_INIT)):
    _ = agent_init.update(replay_buffer, batch_size=args.batch_size)


### the first time period
rewards_on = np.zeros((len(tasks), args.max_epochs))
rewards_off = np.zeros((len(tasks), args.max_epochs))
env.reset_task(tasks[0])
replay_buffer.reset(max_size=args.capacity)
agent_nominal = DDPG_SLLRL(state_dim, action_dim, max_action, lr=args.lr, 
        tau=args.tau, gamma=args.gamma, device=device)
agent_nominal.load_from(agent_init)

print('\n========== Period 1 ==========')
print('Task 1: ', tasks[0])
agent_nominal, replay_buffer, r_on, r_off = train_ddpg(agent_nominal, replay_buffer, env)
print('Average reward: %.2f'%(r_on.mean()))
rewards_on[0] = r_on; rewards_off[0] = r_off
task_ids[0] = 1
agents = [agent_nominal]

### create a Chinese restaurant process
crp = CRP(zeta=2.0)


### following periods
prior_tau = list(np.linspace(0.2, 0.1, 10)) + [0.1] * args.num_tasks
### posterior selection of task-to-cluster assignment
l_post = 1
for period in range(1, args.num_tasks):
    print("\n========== Period %d =========="%(period+1))
    print('Task %d' % (period+1), tasks[period])
    L = crp._L
    prior = crp._prior
    
    ### change the task
    env.reset_task(tasks[period])

    ### collect a few transitions for task inference
    transitions = collect_samples(agents[l_post-1], env, epochs=EPOCHS, random=True)

    ### create a potentially new model
    agent_new = DDPG_SLLRL(state_dim, action_dim, max_action, lr=args.lr, 
            tau=args.tau, gamma=args.gamma, device=device)
    agent_new.load_from(agent_init)

    ### predictive likelihood
    lls = np.zeros(L+1)
    for idx in range(L):
        lls[idx] = agents[idx].compute_likelihood(transitions, sigma=SIGMA)
    lls[-1] = agent_new.compute_likelihood(transitions, sigma=SIGMA)
    lls = softmax_normalize(lls, temperature=1.0)
    print('Predictive likelihood: ', lls)

    ### CRP prior distribution
    prior = softmax_normalize(prior, temperature=prior_tau[period])
    print('Prior distribution: ', prior)

    ### posterior probability
    posterior = lls * prior[:lls.shape[0]]
    posterior = softmax_normalize(posterior, temperature=1.0)
    print('Posterior: ', posterior)
    
    l_post = np.argmax(posterior) + 1
    print('Posterior selection: %d'%l_post)
    if l_post == L+1:
        print('Add a new cluster...')
        agents.append(agent_new)

    ### update the CRP prior distribution
    crp.update(l_post)
    task_ids[period] = l_post

    ### training the chosen agent
    agent = agents[l_post-1]
    replay_buffer.reset(max_size=args.capacity)
    replay_buffer.batch_push(transitions)
    agent, replay_buffer, r_on, r_off = train_ddpg(agent, replay_buffer, env)
    rewards_on[period] = r_on
    rewards_off[period] = r_off
   
    np.save(os.path.join(args.output, 'rewards_sllrl.npy'), rewards_off)
    print('Average reward:', np.mean(rewards_off[ : period+1], axis=1))

    task_info = np.concatenate((tasks, task_ids), axis=1)
    np.save(os.path.join(args.output, 'task_info_sllrl.npy'), task_info)



































    






























print('Running time: %.2f minutes'%((time.time()-start_time)/60.0))





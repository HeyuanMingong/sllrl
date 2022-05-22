#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The base class for a 2D maze environment.
The maze is a m x n matrix, '0' indicates a navigable cell and '1' indicates a block cell.
The state is the 2D cooridnate (x, y), and in each state, four actions are available: 
    '0' stands for 'up'
    '1' stands for 'down'
    '2' stands for 'left'
    '3' stands for 'right'
"""

import numpy as np 
import copy
import gym 
from gym import spaces 
from gym.utils import seeding 

class Maze2DEnv(gym.Env):
    def __init__(self, height=11, width=11, start=[5,5], goal=[8,8]):
        super(Maze2DEnv, self).__init__()
        ### env: m x n matrix, 0 or 1
        self._env = np.zeros((height, width), dtype=np.bool)
        self._start = start; self._goal = goal
        self.width = width; self.height = height
        self.r_goal = 1.0; self.r_obs = -1.0; self.r_step = -0.1

        self.observation_space = spaces.Discrete(width*height)
        self.action_space = spaces.Discrete(4) 

        self.set_obstacles()

    def set_obstacles(self):
        #self._obstacles = [[1,5], [2,5], [3,5], [7,5], [8,5], [9,5],
        #        [5,1], [5,2], [5,3], [5,7], [5,8], [5,9]]
        self._obstacles = []
        for obs in self._obstacles: self._env[obs[0], obs[1]] = True

    def check_goal(self, goal):
        goal = list(goal)
        if goal in self._obstacles or goal == self._start:
            return False
        else:
            return True

    def reset(self):
        self._state = self.to_1d_state(self._start)
        return self._state 

    def to_1d_state(self, state_2d):
        return int(state_2d[0]*self.width + state_2d[1])

    def to_2d_state(self, state_1d):
        state_2d_0 = state_1d // self.width 
        state_2d_1 = state_1d - self.width*state_2d_0 
        return (int(state_2d_0), int(state_2d_1))
            
    def step(self, action):
        assert self.action_space.contains(action)
        state_2d = self.to_2d_state(self._state)
        done = False; r = 0

        if action == 0: # up
            new_2d_0 = state_2d[0] - 1; new_2d_1 = state_2d[1]
        elif action == 1: # down
            new_2d_0 = state_2d[0] + 1; new_2d_1 = state_2d[1]
        elif action == 2: # left
            new_2d_0 = state_2d[0]; new_2d_1 = state_2d[1] - 1
        elif action == 3: #right
            new_2d_0 = state_2d[0]; new_2d_1 = state_2d[1] + 1
        else:
            print('Error! Unkown action...')

        if new_2d_0==self._goal[0] and new_2d_1==self._goal[1]:
            done = True
            r += self.r_goal
        else: 
            if new_2d_0 < 0:
                new_2d_0 = 0; r += self.r_obs
            if new_2d_0 > self.height - 1:
                new_2d_0 = self.height - 1; r += self.r_obs
            if new_2d_1 < 0:
                new_2d_1 = 0; r += self.r_obs
            if new_2d_1 > self.width - 1:
                new_2d_1 = self.width - 1; r += self.r_obs
            if self._env[new_2d_0, new_2d_1]:
                new_2d_0 = state_2d[0]; new_2d_1 = state_2d[1]; r += self.r_obs
            
        # r = -np.sqrt((new_2d_0-self._goal[0])**2+(new_2d_1-self._goal[1])**2)
        r += self.r_step
        self._state = self.to_1d_state((new_2d_0, new_2d_1))

        return self._state, r, done, None  

    def reset_task(self, goal):
        self._goal = list(goal)

    

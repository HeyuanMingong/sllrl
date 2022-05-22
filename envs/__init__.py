#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:14 2018

@author: qiutian
"""
from gym.envs.registration import register

register(
    'Navigation2D-v1',
    entry_point='envs.navigation:Navigation2DEnvV1',
    max_episode_steps=1000
    )

register(
    'Maze2D-v1',
    entry_point='envs.maze:Maze2DEnv',
    max_episode_steps=1000
    )

register(
    'WalkerVel-v1',
    entry_point='envs.mujoco.walker2d:WalkerVelEnv',
    max_episode_steps=1000
    )

register(
    'HopperVel-v1',
    entry_point='envs.mujoco.hopper:HopperVelEnv',
    max_episode_steps=1000
    )

register(
    'HalfCheetahVel-v1',
    entry_point='envs.mujoco.half_cheetah:HalfCheetahVelEnv',
    max_episode_steps=1000
    )

register(
    'AntVel-v1',
    entry_point='envs.mujoco.ant:AntVelEnv',
    max_episode_steps=1000
    )

register(
    'SwimmerVel-v1',
    entry_point='envs.mujoco.swimmer:SwimmerVelEnv',
    max_episode_steps=1000
    )

register(
    'AntNavi-v1',
    entry_point='envs.mujoco.ant:AntNaviEnv',
    max_episode_steps=1000
    )

register(
    'ReacherPos-v1',
    entry_point='envs.mujoco.reacher:ReacherPosEnv',
    max_episode_steps=1000
    )



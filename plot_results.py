#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:27:58 2018

@author: qiutian
"""

import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.manifold import TSNE

###############################################################################
### re-scale the bottom part of the y-axis
def bottom_scale(arr, scale=[-200, -60, -50], dim=None):
    if scale is None:
        return arr
    if dim is None:
        arr = np.array(arr).reshape(-1); arr_n = np.zeros(arr.shape)
        ratio = (scale[2] - scale[1])/(scale[2] - scale[0])
        def shrink(num):
            if num < scale[2]:
                return scale[2] - ratio * (scale[2] - num)
            else:
                return num
        for idx, ii in enumerate(arr):
             arr_n[idx] = shrink(ii)
        return arr_n
    else:
        arr_n = []
        for row in arr:
            arr_n.append(bottom_scale(row, scale=scale, dim=None))
        return np.array(arr_n)
    




p_output = 'output/navi_v1'
###############################################################################
### plot the performance of baselines and our method
def plot_performance():
    ic = range(0, 50); num = 200; mark = 10

    rews_finetune_0 = np.load(os.path.join(p_output, 'rewards_finetune.npy'))[ic, :num]
    rews_reservoir_0 = np.load(os.path.join(p_output, 'rewards_reservoir.npy'))[ic, :num]
    rews_consolidation_0 = np.load(os.path.join(p_output, 'rewards_consolidation.npy'))[ic, :num]
    rews_progressive_0 = np.load(os.path.join(p_output, 'rewards_progressive.npy'))[ic, :num]
    rews_sllrl_0 = np.load(os.path.join(p_output, 'rewards_sllrl.npy'))[ic, :num]
    # rews_progressive_0 = rews_consolidation_0
    
    def ave_rews_stats(arr):
        ave_rews = np.mean(arr, axis=0)
        ave_ave_rews = np.mean(ave_rews)
        err_ave_rews = np.std(ave_rews, ddof=1) / np.sqrt(len(ave_rews))
        return ave_ave_rews, err_ave_rews
    
    stats = np.zeros((5,2))
    stats[0] = ave_rews_stats(rews_finetune_0)
    stats[1] = ave_rews_stats(rews_reservoir_0)
    stats[2] = ave_rews_stats(rews_consolidation_0)
    stats[3] = ave_rews_stats(rews_progressive_0)
    stats[4] = ave_rews_stats(rews_sllrl_0)
    
    print('Fine-tune, mean: %.3f, standard error: %.3f' % (stats[0, 0], stats[0, 1]))
    print('Reservoir, mean: %.3f, standard error: %.3f' % (stats[1, 0], stats[1, 1]))
    print('Consolidation, mean: %.3f, standard error: %.3f' % (stats[2, 0], stats[2, 1]))
    print('Progressive, mean: %.3f, standard error: %.3f' % (stats[3, 0], stats[3, 1]))
    print('Our Method, mean: %.3f, standard error: %.3f' % (stats[4, 0], stats[4, 1]))
    
    def conf_int(arr):
        arr_stats = np.zeros((3, arr.shape[1])) 
        for idx in range(arr.shape[1]):
            col = arr[:, idx]
            arr_stats[0, idx] = np.mean(col)
            down, up = st.t.interval(0.95, len(col)-1, loc=np.mean(col),
                                     scale=st.sem(col))
            arr_stats[1, idx], arr_stats[2, idx] = down, up
        return arr_stats
    
    rews_finetune = conf_int(rews_finetune_0)
    rews_reservoir = conf_int(rews_reservoir_0)
    rews_consolidation = conf_int(rews_consolidation_0)
    rews_progressive = conf_int(rews_progressive_0)
    rews_sllrl = conf_int(rews_sllrl_0)

    
    scale = [-200, -60, -50]
    rews_finetune = bottom_scale(rews_finetune, scale=scale, dim=1)
    rews_reservoir = bottom_scale(rews_reservoir, scale=scale, dim=1)
    rews_consolidation = bottom_scale(rews_consolidation, scale=scale, dim=1)
    rews_progressive = bottom_scale(rews_progressive, scale=scale, dim=1)
    rews_sllrl = bottom_scale(rews_sllrl, scale=scale, dim=1)
    
    
    plt.figure(figsize=(4,3), dpi=200)
    alpha = 0.1; ms = 4; lw = 1; mew = 1
    tick_size = 8; label_size = 10
    x = np.arange(1, num+1)
    
    plt.fill_between(x, rews_finetune[1], rews_finetune[2], color='darkorange', alpha=alpha)
    plt.plot(x, rews_finetune[0], color='darkorange', lw=lw,
             marker='^', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.fill_between(x, rews_reservoir[1], rews_reservoir[2], color='blue', alpha=alpha)
    plt.plot(x, rews_reservoir[0], color='blue', lw=lw,
             marker='+', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.fill_between(x, rews_consolidation[1], rews_consolidation[2], color='green', alpha=alpha)
    plt.plot(x, rews_consolidation[0], color='green', lw=lw,
             marker='o', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.fill_between(x, rews_progressive[1], rews_progressive[2], color='c', alpha=alpha)
    plt.plot(x, rews_progressive[0], color='c', lw=lw,
             marker='s', markevery=mark, ms=ms, mew=mew, mfc='white')
    
    plt.fill_between(x, rews_sllrl[1], rews_sllrl[2], 
                     color='red', alpha=alpha)
    plt.plot(x, rews_sllrl[0], color='red', lw=lw,
             marker='x', markevery=mark, ms=ms, mew=mew, mfc='white')

    
    plt.legend(['Fine-tune', 'Reservoir', 'Consolidation', 'Progessive', 'Our Method'],
               labelspacing=0.1,
               fancybox=True, shadow=True, fontsize=label_size)
    
    plt.xlabel('Learning Episodes', fontsize=label_size)
    plt.ylabel('Return', fontsize=label_size)
    plt.xticks(np.arange(0,num+1,num//5), fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.grid(axis='y', ls='-', lw=0.2)
    plt.grid(axis='x', ls='-', lw=0.2)
    
    plt.axis([0, num, -60, 2])
    yticks = [-200, -50, -40, -30, -20, -10, 0]
    plt.yticks(np.arange(-60, 10, 10), yticks)
        
plot_performance()



###############################################################################
### visualizing the task clustering of our method
def visualze_task_clustering():
    plt.figure(figsize=(3,3),dpi=200)
    
    info = np.load(os.path.join(p_output, 'task_info_sllrl.npy'))[0:50]
    clusters = max(info[:, -1])
    points = np.zeros(int(clusters))
    print('num of clusters', int(clusters))
    
    for idx in range(info.shape[0]):
        item = info[idx][-3:]  
        if int(item[-1]): points[int(item[-1])-1] += 1
        
        S = 50
        if item[-1] == 1:
            plt.scatter(item[0], item[1], marker='x', color='k', s=S)
        elif item[-1] == 2:
            plt.scatter(item[0], item[1], marker='+', color='r', s=S)
        elif item[-1] == 3:
            plt.scatter(item[0], item[1], marker='*', color='m', s=S)
        elif item[-1] == 4:
            plt.scatter(item[0], item[1], marker='^', color='g', s=S)
        elif item[-1] == 5:
            plt.scatter(item[0], item[1], marker='s', color='b', s=S)

    plt.grid(axis='x', ls='--')
    plt.grid(axis='y', ls='--')
    tick_real = [-0.5, -0.25, 0, 0.25, 0.5]
    tick_show = [-0.5, '', '', '', 0.5]
    tick_show_y = ['', '', '', '', 0.5]
    plt.yticks(tick_real, tick_show_y, fontsize=12)
    plt.xticks(tick_real, tick_show, fontsize=12)
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    print('number of points in each cluster: ', points)
    
visualze_task_clustering() 


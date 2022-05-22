import torch 
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import random
from torch.optim import SGD, Adam
import numpy as np
from tqdm import tqdm
import copy
import math


class CRP(object):
    def __init__(self, zeta=1.0):
        super(CRP, self).__init__()
        ### concentration parameter
        self._zeta = zeta 
        ### number of non-empty clusters
        self._L = 1
        ### time period
        self._t = 2
        ### prior distribution
        self._prior = np.array([1/(1+zeta), zeta/(1+zeta)])

    def select(self):
        index = np.random.choice(1+np.arange(self._L+1), p=self._prior)
        return index

    def update(self, index):
        # print('Update the CRP prior after choosing the %d cluster'%index)
        self._t += 1
        if index == self._L + 1:
            print('A new cluster is expanded...')
            self._prior = np.concatenate((self._prior, np.zeros(1)), axis=0)
            self._prior[-1] = self._zeta / (self._t - 1 + self._zeta)
            self._prior[-2] = 1 / (self._t - 1 + self._zeta)
            for idx in range(self._L):
                self._prior[idx] *= (self._t-2+self._zeta)/(self._t-1+self._zeta)
            self._L += 1
        else:
            print('No new cluster...')
            for idx in range (self._L + 1):
                if idx == index - 1:
                    self._prior[idx] = ((self._t-2+self._zeta)*self._prior[idx]+1) / (self._t-1+self._zeta)
                else:
                    self._prior[idx] *= (self._t-2+self._zeta) / (self._t-1+self._zeta)
        #print(self._prior, self._prior.sum())
        #print(self._t); print(self._L)
















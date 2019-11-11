# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019

@author: Brynhildr
"""

from scipy.io import loadmat

#%%
eeg = loadmat(r'E:\dataset\S01.mat')
info = loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
freqs = info['freqs']
phases = info['phases']

del eeg, info

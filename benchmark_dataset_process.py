# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019
Use benchmark dataset to complete my research
@author: Brynhildr
"""
#%% import third part module
import numpy as np
import scipy.io as io
import pandas as pd
import os
import mne
import matplotlib
import seaborn as sns
import signal_processing_function as SPF 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

#%% load benchmark dataset & relative information
# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S01.mat')
info = io.loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
# (n_chans, n_times, n_conditions, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250Hz, HOLLY SHIT!

# condition infomation
sfreq = 250
freqs = info['freqs'].T
phases = info['phases'].T

# load channels information from .txt file
channels = {}
file = open(r'E:\dataset\64-channels.txt')
for line in file.readlines():
    line = line.strip()
    v = line.split(' ')[0]
    k = line.split(' ')[1]
    channels[k] = v
file.close()

#del eeg, info

#%% load multiple data file
filepath = r'E:\dataset\data'

filelist = []
for file in os.listdir(filepath):
    full_path = os.path.join(filepath, file)
    filelist.append(full_path)

i = 0
eeg = np.zeros((35, 64, 1500, 40, 6))
for file in filelist:
    temp = io.loadmat(file)
    eeg[i,:,:,:,:] = temp['data']
    i += 1


#%%

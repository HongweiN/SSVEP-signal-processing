# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019
Use benchmark dataset to complete my research
@author: Brynhildr
"""
#%% import third part module
import numpy as np
from numpy import transpose
import scipy.io as io
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import os

import mne
from mne.filter import filter_data
from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 

#%% load benchmark dataset & relative information

# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S01.mat')
info = io.loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
# (64, 1500, 40, 6) = (n_chans, n_times, n_conditions, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250 Hz, HOLLY SHIT!

# reshape data array
data = data.transpose((3, 2, 0, 1))  # (n_blocks, n_conditions, n_chans, n_times)

# combine data array
# param X: (n_trials, n_chans, n_times)
# X = data[0,:,:,:]
# for i in range(temp.shape[0]-1):
    # X = np.concatenate((X, temp[i+1,:,:,:]), axis=0)

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

del v, k, file, line       # release RAM
del eeg, info     

#%% load multiple data file & also can be used to process multiple data

# CAUTION: may lead to RAM crash (5-D array takes more than 6125MB)
# Now I know why people need 32G's RAM...PLEASE SKIP THIS PART!!!
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
    
# add more codes here to achieve multiple data processing (PLEASE DON'T)
    
del temp, i, file, filelist, filepath, full_path

#%% data preprocessing
# filtering
f_data = np.zeros((6,40,64,1500))
for i in range(data.shape[0]):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5, h_freq=40,
        n_jobs=6)

del data

# get data for linear regression
w1 = f_data[:,:,:,0:125]
w2 = f_data[:,:,:,1375:]
w3 = f_data[:,:,:,1375:1437]
w4 = f_data[:,:,:,1437:]

# get data for comparision
signal_data = f_data[:,:,:,125:1375]

del f_data

# save model data to release RAM, reload before use
w1_path = r'E:\dataset\model_data\S01\w1'
w2_path = r'E:\dataset\model_data\S01\w2'
w3_path = r'E:\dataset\model_data\S01\w1'
w4_path = r'E:\dataset\model_data\S01\w1'
s_path = r'E:\dataset\model_data\S01\w1'

io.savemat(w1_path, {'w1':w1})
io.savemat(w2_path, {'w2':w1})
io.savemat(w3_path, {'w3':w1})
io.savemat(w4_path, {'w4':w1})
io.savemat(s_path, {'signal_data':signal_data})

del w1, w2, w3, w4, signal_data
del w1_path, w2_path, w3_path, w4_path, s_path

# %%

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
# (64, 1500, 40, 6) = (n_chans, n_times, n_stims, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250 Hz, HOLLY SHIT!

# reshape data array: (n_stims, n_epochs, n_chans, n_times)
data = data.transpose((2, 3, 0, 1))  

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
f_data = np.zeros((40,6,64,1500))
for i in range(data.shape[0]):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5, h_freq=40,
        n_jobs=6)

del data

# get data for linear regression
w1 = f_data[:,:,:,0:125]
w2 = f_data[:,:,:,0:63]
w3 = f_data[:,:,:,63:125]

# get data for comparision
signal_data = f_data[:,:,:,125:1375]

del f_data

# save model data to release RAM, reload before use
w1_path = r'E:\dataset\model_data\S01\w1'
w2_path = r'E:\dataset\model_data\S01\w2'
w3_path = r'E:\dataset\model_data\S01\w3'
s_path = r'E:\dataset\signal_data\S01'

io.savemat(w1_path, {'w1':w1})
io.savemat(w2_path, {'w2':w2})
io.savemat(w3_path, {'w3':w3})
io.savemat(s_path, {'signal_data':signal_data})

del w1, w2, w3, signal_data
del w1_path, w2_path, w3_path, s_path

#%% reload data, then divide input&output data for model
# data size: (n_stims, n_epochs, n_chans, n_blocks) 
w1 = io.loadmat(r'E:\dataset\model_data\S01\w1.mat')
w2 = io.loadmat(r'E:\dataset\model_data\S01\w2.mat')
w3 = io.loadmat(r'E:\dataset\model_data\S01\w3.mat')
signal_data = io.loadmat(r'E:\dataset\signal_data\S01.mat')

# w1 model data: 0-500ms
w1_i = w1[:,:,?,:]  # pick input channels:
w1_o = w1[:,:,?,:]  # choose output channels:

# w2 model data: 0-250ms
w2_i = w2[:,:,?,:]  # pick input channels:
w2_o = w2[:,:,?,:]  # choose output channels:

# w3 model data: 250-500ms
w3_i = w3[:,:,?,:]  # pick input channels:
w3_o = w3[:,:,?,:]  # choose output channels:

del w1, w2, w3

#%% multi-linear regression analysis
# regression coefficient, intercept, R^2
# w1 estimate & extract data:(n_stims, n_epochs, n_times)
rc_w1, ri_w1, r2_w1, w1_es_w1, w1_ex_w1 = SPF.mlr_analysis(w1_i, w1_o, w1_i,
                            regression=True, constant=True)

# the same but w2 part data:
rc_w2, ri_w2, r2_w2, w2_es_w2, w2_ex_w2 = SPF.mlr_analysis(w2_i, w2_o, w2_i,
                            regression=True, constant=True)

# the same but w3 part data (use w2)
w2_es_w3, w2_ex_w3 = SPF.mlr_analysis(w2_i, w2_o, w3_i,
                            regression=False, constant=True)

# the same but w3 part data (use w3)
rc_w3, ri_w3, r2_w3, w3_es_w3, w3_ex_w3 = SPF.mlr_analysis(w3_i, w3_o, w3_i,
                            regression=True, constant=True)

# signal part data (use w1):
s_es_w1, s_ex_w1 = SPF.mlr_analysis(w1_i, w1_o, signal_data[:,:,?,:],
                            regression=False, constant=True)

# signal part data (use w2):
s_es_w2, s_ex_w2 = SPF.mlr_analysis(w2_i, w2_o, signal_data[:,:,?,:],
                            regression=False, constant=True)

# signal part data (use w3): 
s_es_w3, s_ex_w3 = SPF.mlr_analysis(w3_i, w3_o, signal_data[:,:,?,:],
                            regression=False, constant=True)

#%% use inverse array to make spatial filter


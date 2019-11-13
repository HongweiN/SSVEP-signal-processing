# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019
Use benchmark dataset to complete my research
@author: Brynhildr
"""
#%% Import third part module
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

#%% Load benchmark dataset & relative information
# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S01.mat')
info = io.loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
# (64, 1500, 40, 6) = (n_chans, n_times, n_events, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250 Hz, HOLLY SHIT!

# reshape data array: (n_events, n_epochs, n_chans, n_times)
data = data.transpose((2, 3, 0, 1))  

# combine data array: np.concatenate(X, Y, axis=)

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

#%% Load multiple data file & also can be used to process multiple data
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

#%% Data preprocessing
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

#%% Reload data 
# data size: (n_events, n_epochs, n_chans, n_times) 
w1 = io.loadmat(r'E:\dataset\model_data\S01\w1.mat')
w2 = io.loadmat(r'E:\dataset\model_data\S01\w2.mat')
w3 = io.loadmat(r'E:\dataset\model_data\S01\w3.mat')
signal_data = io.loadmat(r'E:\dataset\signal_data\S01.mat')

#%% Divide input&output data for model

# pick input channels:
# choose output channels:

# w1 model data: 0-500ms
w1_i = w1[:,:,?,:]
w1_o = w1[:,:,?,:]

# w2 model data: 0-250ms
w2_i = w2[:,:,?,:]
w2_o = w2[:,:,?,:]

# w3 model data: 250-500ms
w3_i = w3[:,:,?,:]
w3_o = w3[:,:,?,:]

# signal part data: 500ms-1250ms
sig_i = signal_data[:,:,?,:]
sig_o = signal_data[:,:,?,:]

#%% Inter-channel correlation analysis: canonical correlation analysis (CCA)

#%% Inter-channel correlation analysis: Spearman correlation
w1_corr_sp = SPF.corr_coef(w1, 'spearman')
w2_corr_sp = SPF.corr_coef(w2, 'spearman')
w3_corr_sp = SPF.corr_coef(w3, 'spearman')

# may need to compute in different parts
sig_corr_sp = SPF.corr_coef(signal_data[:,:,?,:], mode='spearman')

#%% Inter-channel correlation analysis: Pearson correlation
w1_corr_sp = SPF.corr_coef(w1, mode='pearson')
w2_corr_sp = SPF.corr_coef(w2, mode='pearson')
w3_corr_sp = SPF.corr_coef(w3, mode='pearson')

# may need to compute in different parts
sig_corr_sp = SPF.corr_coef(signal_data[:,:,?,:], mode='pearson')

#%% Spatial filter: multi-linear regression method
# regression coefficient, intercept, R^2
rc_w1, ri_w1, r2_w1 = SPF.mlr_analysis(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_mes_w1, w1_mex_w1 = SPF.sig_extract(rc_w1, w1_i, w1_o, intercept=ri_w1)

# the same but w2 part data:
rc_w2, ri_w2, r2_w2 = SPF.mlr_analysis(w2_i, w2_o, w2_i)
w2_mes_w2, w2_mex_w2 = SPF.sig_extract(rc_w2, w2_i, w2_o, intercept=ri_w2)

# the same but w3 part data (use w2)
w2_mes_w3, w2_mex_w3 = SPF.sig_extract(rc_w2, w3_i, w3_o, intercept=ri_w2)

# the same but w3 part data (use w3)
rc_w3, ri_w3, r2_w3 = SPF.mlr_analysis(w3_i, w3_o, w3_i)
w3_mes_w3, w3_mex_w3 = SPF.sig_extract(rc_w3, w3_i, w3_o, intercept=ri_w3)

# signal part data (use w1):
s_mes_w1, s_mex_w1 = SPF.sig_extract(rc_w1, sig_i, sig_o, intercept=ri_w1)

# signal part data (use w2):
s_mes_w2, s_mex_w2 = SPF.sig_extract(rc_w2, sig_i, sig_o, intercept=ri_w2)

# signal part data (use w3): 
s_mes_w3, s_mex_w3 = SPF.sig_extract(rc_w3, sig_i, sig_o, intercept=ri_w3)

#%% Spatial filter: inverse array method
# filter coefficient
sp_w1 = SPF.inv_spa(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_ies_w1, w1_iex_w1 = SPF.sig_extract(sp_w1, w1_i, w1_o)

# the same but w2 part data:
sp_w2 = SPF.inv_spa(w2_i, w2_o)
w2_ies_w2, w2_iex_w2 = SPF.sig_extract(sp_w2, w2_i, w2_o)

# the same but w3 part data (use w2):
w2_ies_w3, w2_iex_w3 = SPF.sig_extract(sp_w2, w3_i, w3_o)

# the same but w3 part data (use w3):
sp_w3 = SPF.inv_spa(w3_i, w3_o)
w3_ies_w3, w3_iex_w3 = SPF.sig_extract(sp_w3, w3_i, w3_o)

# signal part data (use w1):
s_ies_w1, s_iex_w1 = SPF.sig_extract(sp_w1, sig_i, sig_o)

# signal part data (use w2):
s_ies_w2, s_iex_w2 = SPF.sig_extract(sp_w2, sig_i, sig_o)

# signal part data (use w3):
s_ies_w3, s_iex_w3 = SPF.sig_extract(sp_w3, sig_i, sig_o)

#%% Variance

#%% Cosine similarity (background part): normal sim
w1_w1_m_nsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='normal')
w2_w2_m_nsim = 
w2_w3_m_nsim = 
w3_w3_m_nsim = 

w1_w1_i_nsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='normal')
w2_w2_i_nsim = 
w2_w3_i_nsim = 
w3_w3_i_nsim = 

#%% 
# tanimoto coefficient (generalized Jaccard)
w1_w1_m_nsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='normal')
w2_w2_m_nsim = 
w2_w3_m_nsim = 
w3_w3_m_nsim = 

w1_w1_i_nsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='normal')
w2_w2_i_nsim = 
w2_w3_i_nsim = 
w3_w3_i_nsim = 


#%% Power spectrum density

#%% Precise FFT transform

#%% SNR in time domain

#%% SNR in frequency domain

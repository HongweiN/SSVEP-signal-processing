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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import os

import mne
from mne.filter import filter_data
from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 

#%% prevent ticking 'F5'
???

#%%*************************Part I: processing data*************************
#%% Load benchmark dataset & relative information
# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S15.mat')
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
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=6)

#del data, i

# get data for linear regression
w1 = f_data[:,:,:,0:125]
w2 = f_data[:,:,:,0:63]
w3 = f_data[:,:,:,63:125]

# get data for comparision
signal_data = f_data[:,:,:,125:1375]

del f_data

# save model data to release RAM, reload before use
w1_path = r'E:\dataset\model_data\S15\w1'
w2_path = r'E:\dataset\model_data\S15\w2'
w3_path = r'E:\dataset\model_data\S15\w3'
s_path = r'E:\dataset\signal_data\S15'

io.savemat(w1_path, {'w1':w1})
io.savemat(w2_path, {'w2':w2})
io.savemat(w3_path, {'w3':w3})
io.savemat(s_path, {'signal_data':signal_data})

del w1, w2, w3, signal_data
del w1_path, w2_path, w3_path, s_path


#%% Reload data 
# data size: (n_events, n_epochs, n_chans, n_times) 
w1 = io.loadmat(r'E:\dataset\model_data\S15\w1.mat')
w1 = w1['w1']
w2 = io.loadmat(r'E:\dataset\model_data\S15\w2.mat')
w2 = w2['w2']
w3 = io.loadmat(r'E:\dataset\model_data\S15\w3.mat')
w3 = w3['w3']
signal_data = io.loadmat(r'E:\dataset\signal_data\S15.mat')
signal_data = signal_data['signal_data']


#%% Inter-channel correlation analysis: Spearman correlation
w1_corr_sp = SPF.corr_coef(w1, 'spearman')
w2_corr_sp = SPF.corr_coef(w2, 'spearman')
w3_corr_sp = SPF.corr_coef(w3, 'spearman')

sig_corr_sp = SPF.corr_coef(signal_data, mode='spearman')

compare = w1_corr_sp - sig_corr_sp

for i in range(64):
    for j in range(64):
        if compare[i,j] < 0.03:
            compare[i,j] = 0


#%% Inter-channel correlation analysis: Pearson correlation
w1_corr_sp = SPF.corr_coef(w1, 'pearson')
w2_corr_sp = SPF.corr_coef(w2, 'pearson')
w3_corr_sp = SPF.corr_coef(w3, 'pearson')

sig_corr_sp = SPF.corr_coef(signal_data, mode='pearson')

compare = w1_corr_sp - sig_corr_sp

for i in range(64):
    for j in range(64):
        if compare[i,j] < 0.03:
            compare[i,j] = 0

#%% Binarization (if neccessary)
compare_corr = SPF.bina_corr(w1_corr_sp, sig_corr_sp, th=0.03)


#%% Inter-channel correlation analysis: canonical correlation analysis (CCA)
            

#%% Divide input&output data for model according to correlation

# pick input channels: Cz, CP6, TP8, P4, P6, P8
# choose output channels: Oz

# w1 model data: 0-500ms
w1_i = w1[:,:,[27,40,41,49,50,51],:]
w1_o = w1[:,:,61,:]
w1_total = w1[:,:,[27,40,41,49,50,51,61],:]

# w2 model data: 0-250ms
w2_i = w2[:,:,[27,40,41,49,50,51],:]
w2_o = w2[:,:,61,:]
#w2_total = w2[:,:,[27,40,41,49,50,51,61],:]

# w3 model data: 250-500ms
w3_i = w3[:,:,[27,40,41,49,50,51],:]
w3_o = w3[:,:,61,:]
#w3_total = w3[:,:,[27,40,41,49,50,51,61],:]

# signal part data: 500ms-1250ms
sig_i = signal_data[:,:,[27,40,41,49,50,51],:]
sig_o = signal_data[:,:,61,:]
sig_total = signal_data[:,:,[27,40,41,49,50,51,61],:]


#%% Prepare for checkboard plot (Spearman method)
w1_pick_corr_sp = SPF.corr_coef(w1_total, 'spearman')
#w2_pick_corr_sp = SPF.corr_coef(w2_total, 'spearman')
#w3_pick_corr_sp = SPF.corr_coef(w3_total, 'spearman')

sig_pick_corr_sp = SPF.corr_coef(sig_total, 'spearman')


#%% Prepare for checkboard plot (Pearson method)
w1_pick_corr_sp = SPF.corr_coef(w1_total, 'pearson')
#w2_pick_corr_sp = SPF.corr_coef(w2_total, 'pearson')
#w3_pick_corr_sp = SPF.corr_coef(w3_total, 'pearson')

sig_pick_corr_sp = SPF.corr_coef(sig_total, 'pearson')

        
#%% Spatial filter: multi-linear regression method
# regression coefficient, intercept, R^2
rc_w1, ri_w1, r2_w1 = SPF.mlr_analysis(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_mes_w1, w1_mex_w1 = SPF.sig_extract(rc_w1, w1_i, w1_o, ri_w1)

# the same but w2 part data:
rc_w2, ri_w2, r2_w2 = SPF.mlr_analysis(w2_i, w2_o, w2_i, 0)
w2_mes_w2, w2_mex_w2 = SPF.sig_extract(rc_w2, w2_i, w2_o, ri_w2)

# the same but w3 part data (use w2)
w2_mes_w3, w2_mex_w3 = SPF.sig_extract(rc_w2, w3_i, w3_o, ri_w2)

# the same but w3 part data (use w3)
rc_w3, ri_w3, r2_w3 = SPF.mlr_analysis(w3_i, w3_o, w3_i, 0)
w3_mes_w3, w3_mex_w3 = SPF.sig_extract(rc_w3, w3_i, w3_o, ri_w3)

# signal part data (use w1):
s_mes_w1, s_mex_w1 = SPF.sig_extract(rc_w1, sig_i, sig_o, ri_w1)
# signal part data (use w2):
s_mes_w2, s_mex_w2 = SPF.sig_extract(rc_w2, sig_i, sig_o, ri_w2)
# signal part data (use w3): 
s_mes_w3, s_mex_w3 = SPF.sig_extract(rc_w3, sig_i, sig_o, ri_w3)


#%% Spatial filter: inverse array method
# filter coefficient
sp_w1 = SPF.inv_spa(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_ies_w1, w1_iex_w1 = SPF.sig_extract(sp_w1, w1_i, w1_o, 0)
# w1 model's goodness of fit
gf_w1 = SPF.fit_goodness(w1_o, w1_ies_w1, chans=5)

# the same but w2 part data:
sp_w2 = SPF.inv_spa(w2_i, w2_o)
w2_ies_w2, w2_iex_w2 = SPF.sig_extract(sp_w2, w2_i, w2_o, 0)
gf_w2 = SPF.fit_goodness(w2_o, w2_ies_w2, chans=5)

# the same but w3 part data (use w2):
w2_ies_w3, w2_iex_w3 = SPF.sig_extract(sp_w2, w3_i, w3_o, 0)

# the same but w3 part data (use w3):
sp_w3 = SPF.inv_spa(w3_i, w3_o)
w3_ies_w3, w3_iex_w3 = SPF.sig_extract(sp_w3, w3_i, w3_o, 0)
gf_w3 = SPF.fit_goodness(w3_o, w3_ies_w3, chans=5)

# signal part data (use w1):
s_ies_w1, s_iex_w1 = SPF.sig_extract(sp_w1, sig_i, sig_o, 0)
# signal part data (use w2):
s_ies_w2, s_iex_w2 = SPF.sig_extract(sp_w2, sig_i, sig_o, 0)
# signal part data (use w3):
s_ies_w3, s_iex_w3 = SPF.sig_extract(sp_w3, sig_i, sig_o, 0)


#%% Cosine similarity (background part): normal
# w1 estimate (w1 model) & w1 original, mlr, normal similarity, the same below
w1_w1_m_nsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='normal')
w2_w2_m_nsim = SPF.cos_sim(w2_o, w2_mes_w2, mode='normal')
w2_w3_m_nsim = SPF.cos_sim(w3_o, w2_mes_w3, mode='normal')
w3_w3_m_nsim = SPF.cos_sim(w3_o, w3_mes_w3, mode='normal')

w1_w1_i_nsim = SPF.cos_sim(w1_o, w1_ies_w1, mode='normal')
w2_w2_i_nsim = SPF.cos_sim(w2_o, w2_ies_w2, mode='normal')
w2_w3_i_nsim = SPF.cos_sim(w3_o, w2_ies_w3, mode='normal')
w3_w3_i_nsim = SPF.cos_sim(w3_o, w3_ies_w3, mode='normal')


#%% Cosine similarity (background part): Tanimoto (generalized Jaccard)
# w1 estimate (w1 model) & w1 original, mlr, Tanimoto, the same below
w1_w1_m_tsim = SPF.cos_sim(w1_o, w1_mes_w1, mode='tanimoto')
w2_w2_m_tsim = SPF.cos_sim(w2_o, w2_mes_w2, mode='tanimoto')
w2_w3_m_tsim = SPF.cos_sim(w3_o, w2_mes_w3, mode='tanimoto')
w3_w3_m_tsim = SPF.cos_sim(w3_o, w3_mes_w3, mode='tanimoto')

w1_w1_i_tsim = SPF.cos_sim(w1_o, w1_ies_w1, mode='tanimoto')
w2_w2_i_tsim = SPF.cos_sim(w2_o, w2_ies_w2, mode='tanimoto')
w2_w3_i_tsim = SPF.cos_sim(w3_o, w2_ies_w3, mode='tanimoto')
w3_w3_i_tsim = SPF.cos_sim(w3_o, w3_ies_w3, mode='tanimoto')


#%% Power spectrum density
w1_p, f = SPF.welch_p(s_iex_w1, sfreq=250, fmin=0, fmax=50, n_fft=2048,
                      n_overlap=250, n_per_seg=500)
w2_p, f = SPF.welch_p(s_iex_w2, sfreq=250, fmin=0, fmax=50, n_fft=2048,
                      n_overlap=250, n_per_seg=500)
w3_p, f = SPF.welch_p(s_iex_w3, sfreq=250, fmin=0, fmax=50, n_fft=2048,
                      n_overlap=250, n_per_seg=500)
sig_p, f = SPF.welch_p(sig_o, sfreq=250, fmin=0, fmax=50, n_fft=2048,
                       n_overlap=250, n_per_seg=500)


#%% Precise FFT transform

#%% Variance
# original signal variance
var_o_t = var_estimation(sig_o)

# extract signal variance (w1 model) 
var_w1_m_t = var_estimation(w1_mex_w1)
var_w1_i_t = var_estimation(w1_iex_w1)

# extract signal variance (w2 model) 
var_w2_m_t = var_estimation(w2_mex_w2)
var_w2_i_t = var_estimation(w2_iex_w2)

# extract signal variance (w3 model) 
var_w3_m_t = var_estimation(w3_mex_w3)
var_w3_i_t = var_estimation(w3_iex_w3)


#%% SNR in time domain
# original signal snr
snr_o_t = SPF.snr_time(sig_o, mode='time')

# extract signal snr (w1 model) 
#snr_w1_m_t = snr_time(s_mex_w1, mode='time')
snr_w1_i_t = SPF.snr_time(s_iex_w1, mode='time')

# extract signal snr (w2 model) 
#snr_w2_m_t = snr_time(s_mex_w2, mode='time')
snr_w2_i_t = SPF.snr_time(s_iex_w2, mode='time')

# extract signal snr (w3 model) 
#snr_w3_m_t = snr_time(s_mex_w3, mode='time')
snr_w3_i_t = SPF.snr_time(s_iex_w3, mode='time')


#%% SNR in frequency domain


#%%*************************Part II: plot figures*************************
#%% Model descrpition (Comoplex)
fig = plt.figure(figsize=(24,24))
fig.suptitle(r'$\ Model\ Description$', fontsize=30, fontweight='bold')
gs = GridSpec(6, 7, figure=fig)

# 1. Boxplot of R^2 
X = gf_w1.flatten()
Y = gf_w2.flatten()
Z = gf_w3.flatten()

xmin = min(np.min(X), np.min(Y), np.min(Z)) - 0.05

R2 = np.zeros((720))
R2[0:240] = X
R2[240:480] = Y
R2[480:720] = Z
model = ['w1' for i in range(240)]+['w2' for i in range(240)]+['w3' for i in range(240)]
R2 = pd.DataFrame({r'$\ model$': model, r'$\ R^2$': R2})

order=['w1', 'w2', 'w3']
sns.set(style="whitegrid")

ax1 = fig.add_subplot(gs[0:4, 0:4])
ax1.set_title(r"$\ 3\ model's\ R^2$", fontsize=26)
ax1.tick_params(axis='both', labelsize=22)
ax1.set_xlim((xmin, 1.05))
ax1 = sns.boxplot(x=r'$\ R^2$', y=r'$\ model$', data=R2, notch=True,
                  linewidth=2.5, orient='h', fliersize=10)
ax1 = sns.swarmplot(x=r'$\ R^2$', y=r'$\ model$', data=R2, color='dimgrey',
                    orient='h', size=5)
ax1.set_xlabel(r'$\ R^2\ values$', fontsize=22)
ax1.set_ylabel(r'$\ Models$', fontsize=22)


# 2. Histogram of R^2
ax2 = fig.add_subplot(gs[4:, 0:4])
ax2.set_title(r'$\ Distribution\ of\ R^2$', fontsize=26)
ax2.set_xlabel(r'$\ R^2\ values$', fontsize=22)
ax2.set_ylabel(r'$\ Frequence$', fontsize=22)
ax2.tick_params(axis='both', labelsize=22)
ax2.set_xlim((xmin, 1.05))
ax2 = sns.kdeplot(X, shade=True, label=r'$\ w1$')
ax2 = sns.kdeplot(Y, shade=True, label=r'$\ w2$')
ax2 = sns.kdeplot(Z, shade=True, label=r'$\ w3$')
ax2.legend(loc='best', fontsize=16)

del X, Y, Z


# 3. Inter-channel correlation (2 parts + compare)
X = w1_pick_corr_sp
Y = sig_pick_corr_sp
Z = X - Y
pick_chans = ['Cz','CP6','TP8','P4','P6','P8','Oz']  # change each time

vmin = min(np.min(X), np.min(Y))
vmax = max(np.max(X), np.max(Y))

inter_chan = cm.get_cmap('Blues', 64)

ax3 = fig.add_subplot(gs[0:2, 4:])
mesh = ax3.pcolormesh(X, cmap=inter_chan, vmin=vmin, vmax=vmax)
ax3.set_title(r'$\ Rest\ part$', fontsize=26)
ax3.set_xticks(np.arange(len(pick_chans)))
ax3.set_xticklabels(pick_chans)
ax3.set_xlabel(r'$\ Channels$', fontsize=22)
ax3.set_ylabel(r'$\ Channels$', fontsize=22)
#ax3.tick_params(axis='both', labelsize=22)
fig.colorbar(mesh, ax=ax3)

ax4 = fig.add_subplot(gs[2:4, 4:])
mesh = ax4.pcolormesh(Y, cmap=inter_chan, vmin=vmin, vmax=vmax)
ax4.set_title(r'$\ Signal\ part$', fontsize=26)
ax4.set_xlabel(r'$\ Channels$', fontsize=22)
ax4.set_ylabel(r'$\ Channels$', fontsize=22)
#ax4.tick_params(axis='both', labelsize=22)
fig.colorbar(mesh, ax=ax4)

ax5 = fig.add_subplot(gs[4:, 4:])
mesh = ax5.pcolormesh(Z, cmap=inter_chan, vmin=np.min(X-Y), vmax=np.max(X-Y))
ax5.set_title(r'$\ Rest\ -\ Signal$', fontsize=26)
ax5.set_xlabel(r'$\ Channels$', fontsize=22)
ax5.set_ylabel(r'$\ Channels$', fontsize=22)
#ax5.tick_params(axis='both', labelsize=22)
fig.colorbar(mesh, ax=ax5)

fig.subplots_adjust(top=0.949, bottom=0.05, left=0.049, right=0.990, 
                    hspace=1.000, wspace=1.000)

#plt.savefig(r'E:\fuck.png', dpi=600)


#%% Signal waveform plot
# 1. w1 part (w1 model): original & estimation & extraction
# 2. w2 part (w2 model)
# 3. w3 part (w2 model)
# 4. w3 part (w3 model)
# 5. signal part ()
fig, axes = plt.subplots(2,1, figsize=(16,16))

axes[0].set_title('signal', fontsize=20)
axes[0].set_xlabel('time/ms', fontsize=20)
axes[0].set_ylabel('SNR', fontsize=20)
axes[0].plot(np.mean(sig_o[7,:,:], axis=0), label='origin:125-1375')
axes[0].plot(np.mean(s_iex_w1[7,:,:], axis=0), label='w1:0-125')
axes[0].plot(np.mean(s_iex_w2[7,:,:], axis=0), label='w2:0-63')
axes[0].plot(np.mean(s_iex_w3[7,:,:], axis=0), label='w3:63-125')
axes[0].tick_params(axis='both', labelsize=20)
axes[0].legend(loc='upper right', fontsize=20)

axes[1].set_title('time snr', fontsize=20)
axes[1].set_xlabel('time/ms', fontsize=20)
axes[1].set_ylabel('SNR', fontsize=20)
axes[1].plot(snr_o_t[7,:], label='origin:125-1375')
axes[1].plot(snr_w1_i_t[7,:], label='w1:0-125')
axes[1].plot(snr_w2_i_t[7,:], label='w2:0-63')
axes[1].plot(snr_w3_i_t[7,:], label='w3:63-125')
axes[1].tick_params(axis='both', labelsize=20)
axes[1].legend(loc='best', fontsize=20)


#%% plot PSD
plt.title('signal psd', fontsize=20)
plt.xlabel('frequency/Hz', fontsize=20)
plt.plot(f[1,1,:], np.mean(sig_p[7,:,:], axis=0), label='origin:125-1375')
plt.plot(f[1,1,:], np.mean(w1_p[7,:,:], axis=0), label='w1:0-125')
plt.plot(f[1,1,:], np.mean(w2_p[7,:,:], axis=0), label='w2:0-63')
plt.plot(f[1,1,:], np.mean(w3_p[7,:,:], axis=0), label='w3:63-125')
plt.legend(loc='best', fontsize=20)

#%%
def strain(X):
    strain = np.zeros((40,50))
    
    for i in range(40):
        k=0
        for j in range(50):
            strain[i,j] = np.mean(X[i,k:k+25])
            k += 25
    return strain

#%%
snr1 = strain(snr_o_t)
snr2 = strain(snr_w1_i_t)
snr3 = strain(snr_w2_i_t)
snr4 = strain(snr_w3_i_t)

#%%
plt.plot(snr1[7,:], label='origin:125-1375')
plt.plot(snr2[7,:], label='w1:0-125')
plt.plot(snr3[7,:], label='w2:0-63')
plt.plot(snr4[7,:], label='w3:63-125')
plt.tick_params(axis='both', labelsize=20)
plt.legend(loc='best', fontsize=20)
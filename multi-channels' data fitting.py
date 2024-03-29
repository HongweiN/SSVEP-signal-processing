# -*- coding: utf-8 -*-
"""
SSVEP data processing program
Created on Tue Oct 22 16:03:11 2019

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
from mne.io import concatenate_raws
from mne import Epochs, pick_types, find_events
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet, psd_array_welch)
from mne.baseline import rescale
from sklearn.linear_model import LinearRegression

#%% load data
filepath = r'E:\Documents\SSVEP瞬态特性的研究\data'

subjectlist = ['wangruiyan']

filefolders = []
for subindex in subjectlist:
    filefolder = os.path.join(filepath, subindex)
    filefolders.append(filefolder)

filelist = []
for filefolder in filefolders:
    for file in os.listdir(filefolder):
        filefullpath = os.path.join(filefolder, file)
        filelist.append(filefullpath)

raw_cnts = []
for file in filelist:
    montage = mne.channels.read_montage('standard_1020')
    raw_cnt = mne.io.read_raw_cnt(file, montage=montage,
            eog=['HEO', 'VEO'], emg=['EMG'], ecg=['EKG'],
            preload=True, verbose=False, stim_channel='True')
    raw_cnts.append(raw_cnt)

raw = concatenate_raws(raw_cnts)

del raw_cnts, raw_cnt

#%% prepare for data extraction
raw.filter(l_freq=5, h_freq=40., method='fir', phase='zero',
           fir_window='hamming', fir_design='firwin', skip_by_annotation='edge')
events = mne.find_events(raw, output='onset')

raw.pick_types(raw.info, emg=False, eeg=True, stim=False, eog=False)

# drop channels
drop_chans = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 
              'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 
              'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 
              'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'M2']
raw.drop_channels(drop_chans)

# define labels
event_id = dict(f8d6p0=1, f10d6p0=2, f15d6p0=3,
                f8d6p05=4, f10d6p05=5, f15d6p05=6,
                f8d6p1=7, f10d6p1=8, f15d6p1=9,
                f8d18p0=10, f10d18p0=11, f15d18p0=12,
                f8d18p05=13, f10d18p05=14, f15d18p05=15,
                f8d18p1=16, f10d18p1=17, f15d18p1=18,
                f8d30p0=19, f10d30p0=20, f15d30p0=21,
                f8d30p05=22, f10d30p05=23, f15d30p05=24,
                f8d30p1=25, f10d30p1=26, f15d30p1=27)

#%% get data for linear regression
tmin_m, tmax_m = -0.8, 0  # set the time range for linear regression

# prepare for high-dimension regression data
pick_chans = ['O1','O2','OZ','POZ','PO3','PO4']

sfreq = 1000                               # sampling frequency
tfoc = int(events.shape[0]/len(event_id))  # number of trails for one test condition
npc = len(pick_chans)                      # number of picked channels
tp_m = int(1 + (tmax_m-tmin_m) * sfreq)    # time points
ne = int(len(event_id))                    # number of events

model_data = np.zeros((ne, tfoc, npc, tp_m))

for i in range(ne):
    epochs = Epochs(raw, events=events, event_id=i+1, tmin=tmin_m,
                    tmax=tmax_m, baseline=None, preload=True)
    epochs_copy = epochs.copy().pick_channels(pick_chans)  # pick channels
    model_data[i,:,:,:] = epochs_copy.get_data()           # get the 4D array
    # (n_events, n_trials, n_channels, timepoints) at here 1 trial just has 1 epoch

model_data *= 1e6             # reset the units

# create two model in timing
model_w1 = model_data[:,:,:,:400]
model_w2 = model_data[:,:,:,400:]

del model_data  # release RAM

#%% get data for model comparision
tmin_s, tmax_s = -0.8, 1    # time range for model comparision (signal data)
baseline = (-0.2, 0)        # define baseline  

# prepare for high-dimension signal data                     
tp_s = int(1 + (tmax_s-tmin_s) * sfreq)          

signal_data = np.zeros((ne, tfoc, npc, tp_s))

for i in range(ne):
    epochs = Epochs(raw, events=events, event_id=i+1, tmin=tmin_s,
                    tmax=tmax_s, baseline=baseline, preload=True)
    epochs_copy = epochs.copy().pick_channels(pick_chans)
    signal_data[i,:,:,:] = epochs_copy.get_data()
    # (n_events, n_trials, n_channels, timepoints)
    
signal_data *= 1e6
# O1, O2, OZ, POZ, PO3, PO4
del raw

#%% flatten model data into 2-D array to do LRA in SPSS
ttp_w1 = model_w1.shape[3] * ne * tfoc
ttp_w2 = model_w2.shape[3] * ne * tfoc
w1_data = np.zeros((npc, ttp_w1))  # prepare for flattening
w2_data = np.zeros((npc, ttp_w2))
    
for i in range(npc):
    w1_data[i,:] = model_w1[:,:,i,:].flatten()
    w2_data[i,:] = model_w2[:,:,i,:].flatten()
    
#%% save model data into excel files for spss
# remember to transform the array before write into .csv
csv_w1 = pd.DataFrame(columns=pick_chans, data=w1_data.T)
csv_w2 = pd.DataFrame(columns=pick_chans, data=w2_data.T)  
csv_w1.to_csv(r'E:\Documents\医学工程与转化医学研究院\研究生课题\program\w1_data.csv')
csv_w2.to_csv(r'E:\Documents\医学工程与转化医学研究院\研究生课题\program\w2_data.csv')

del csv_w1, csv_w2
#del w1_data, w2_data  # release RAM

#%% save data for matlab
w1_path = r'E:\Documents\医学工程与转化医学研究院\研究生课题\program\w1_data'
w2_path = r'E:\Documents\医学工程与转化医学研究院\研究生课题\program\w2_data'
io.savemat(w1_path, {'w1_data':w1_data})
io.savemat(w2_path, {'w2_data':w2_data})

del w1_path, w2_path

#%% prepare for plotting
# w1 model data
w1_i = model_w1[:,:,0:5,:]  # input channels: O1,O2,OZ,POZ,PO3 (in order)
w1_o = model_w1[:,:,5,:]    # output channel: PO4 

# w2 model data
w2_i = model_w2[:,:,0:5,:]  # input channels: O1,O2,OZ,POZ,PO3
w2_o = model_w2[:,:,5,:]    # output channel: PO4

del model_w1, model_w2  # release RAM

#%% plot origin data(background and signal)
fig = plt.figure(figsize=(18,16))

x1 = [i for i in range(-800, -400, 1)]
x2 = [i for i in range(-400, 1, 1)]
x3 = [i for i in range(-200, 1001, 1)]

gs = GridSpec(3, 2, figure=fig)

ax1 = fig.add_subplot(gs[:2, :])
ax1.set_title(r'$\ Original\ Signal\ Data$', fontsize=28)
ax1.set_xlabel(r'$\ Time/ms$', fontsize=24)
ax1.set_ylabel(r'$\ Amplitude/μV$', fontsize=24)
ax1.plot(x3, np.mean(signal_data[23,:,0,:], axis=0), label=r'$\ O1$', linewidth=1)
ax1.plot(x3, np.mean(signal_data[23,:,1,:], axis=0), label=r'$\ O2$', linewidth=1)
ax1.plot(x3, np.mean(signal_data[23,:,2,:], axis=0), label=r'$\ OZ$', linewidth=1)
ax1.plot(x3, np.mean(signal_data[23,:,3,:], axis=0), label=r'$\ POZ$', linewidth=1)
ax1.plot(x3, np.mean(signal_data[23,:,4,:], axis=0), label=r'$\ PO3$', linewidth=1)
ax1.plot(x3, np.mean(signal_data[23,:,5,:], axis=0), label=r'$\ PO4$', linewidth=3)
ax1.vlines(8.3, -20, 20, colors='black', linestyles='dashed',
           label=r'$\ start\ point$', linewidth=2)
ax1.tick_params(axis='both', labelsize=18)
ax1.legend(loc='upper right', fontsize=18)

ax2 = fig.add_subplot(gs[2, 0])
ax2.set_title(r'$\ Part\ I\ Background\ Data$', fontsize=28)
ax2.set_xlabel(r'$\ Time/ms$', fontsize=24)
ax2.set_ylabel(r'$\ Amplitude/μV$', fontsize=24)
ax2.plot(x1, w1_i[6,15,0,:], label=r'$\ O1$', linewidth=1)
ax2.plot(x1, w1_i[6,15,1,:], label=r'$\ O2$', linewidth=1)
ax2.plot(x1, w1_i[6,15,2,:], label=r'$\ OZ$', linewidth=1)
ax2.plot(x1, w1_i[6,15,3,:], label=r'$\ POZ$', linewidth=1)
ax2.plot(x1, w1_i[6,15,4,:], label=r'$\ PO3$', linewidth=1)
ax2.plot(x1, w1_o[6,15,:], label=r'$\ PO4$', linewidth=3)
ax2.tick_params(axis='both', labelsize=18)
ax2.legend(loc='lower right', fontsize=16)

ax3 = fig.add_subplot(gs[2, 1])
ax3.set_title(r'$\ Part\ II\ Background\ Data$', fontsize=28)
ax3.set_xlabel(r'$\ Time/ms$', fontsize=24)
ax3.set_ylabel(r'$\ Amplitude/μV$', fontsize=24)
ax3.plot(x2, w2_i[6,15,0,:], label=r'$\ O1$', linewidth=1)
ax3.plot(x2, w2_i[6,15,1,:], label=r'$\ O2$', linewidth=1)
ax3.plot(x2, w2_i[6,15,2,:], label=r'$\ OZ$', linewidth=1)
ax3.plot(x2, w2_i[6,15,3,:], label=r'$\ POZ$', linewidth=1)
ax3.plot(x2, w2_i[6,15,4,:], label=r'$\ PO3$', linewidth=1)
ax3.plot(x2, w2_o[6,15,:], label=r'$\ PO4$', linewidth=3)
ax3.tick_params(axis='both', labelsize=18)
ax3.legend(loc='lower right', fontsize=16)

fig.subplots_adjust(left=0.08, bottom=0.070, right=0.950, top=0.950, hspace=0.5, wspace=0.25)
                    
#plt.savefig(r'E:\origin_data-mqf.png', dpi=600)
plt.show()

#%% linear regression analysis and target channel data estimation
# w1 part:-800~-400ms; w2 part:-400~0ms; s part:-200~1000ms
# R^2 & w1 estimate data: n_events, n_trials, n_times
R2_w1, w1_PO4_w1 = SPF.mlr_analysis(w1_i, w1_o, w1_i)
# w1 PO4-only data: n_events, n_trials, n_times
w1_PO4_only_w1 = w1_o - w1_PO4_w1

# R^2 & w2 estimate data(use w2): n_events, n_trials, n_times
R2_w2, w2_PO4_w2 = SPF.mlr_analysis(w2_i, w2_o, w2_i)
# R^2 & w2 estimate data(use w1): n_events, n_trials, n_times
R2_w1, w2_PO4_w1 = SPF.mlr_analysis(w1_i, w1_o, w2_i)
# w2 PO4-only data(use w1): n_events, n_trials, n_times
w2_PO4_only_w1w2 = w2_o - w2_PO4_w1
# w2 PO4-only data(use w2): n_events, n_trials, n_times
w2_PO4_only_w2w2 = w2_o - w2_PO4_w2

# s estimate data(use w1): n_events, n_trials, n_times
s_PO4_w1 = SPF.mlr_analysis(w1_i, w1_o, signal_data[:,:,0:5,:], regression=False)
# s estimate data(use w2): n_events, n_trials, n_times
s_PO4_w2 = SPF.mlr_analysis(w2_i, w2_o, signal_data[:,:,0:5,:], regression=False)               
# s PO4-only data(use w1): n_events, n_trials, n_times
s_PO4_only_w1 = signal_data[:,:,5,:] - s_PO4_w1
# s PO4-only data(use w2): n_events, n_trials, n_times
s_PO4_only_w2 = signal_data[:,:,5,:] - s_PO4_w2

#%% use SPSS to test the model
X = w2_i[14,5,:,:]
Y = w2_o[14,5,:]

csv_X = pd.DataFrame(columns=['O1','O2','OZ','POZ','PO3'], data=X.T)
csv_Y = pd.DataFrame(columns=['PO4'], data=Y.T)  
csv_X.to_csv(r'E:\Documents\医学工程与转化医学研究院\研究生课题\program\X.csv')
csv_Y.to_csv(r'E:\Documents\医学工程与转化医学研究院\研究生课题\program\Y.csv')

#%% plot estimate comparison data (estimate and original)
'''
line1:
    fig1: contains original background EEG, estimate data using w1 model and 
        target-channel-only data in -800~-400ms(w1 part)
    fig2: contains original background EEG, estimate data using w1 model and 
        target-channel-only data in -400~-0ms(w2 part)
    fig3: contains original background EEG, estimate data using w2 model and
        target-channel-only data in w2 part
line2:
    fig4: contains original SSVEP data(the mean of multi-trials'data), estimate
        SSVEP using w1 model(the mean of multi-trials'data) and target-channel-only
        SSVEP in -200~1000ms (the stimulus begin at around 8.3ms)
    fig5: contains original SSVEP data(mean), estimate SSVEP using w2 model(mean)
        and target-channel-only SSVEP in -200~1000ms
'''
fig = plt.figure(figsize=(18,10))

x1 = [i for i in range(-800, -400, 1)]
x2 = [i for i in range(-400, 1, 1)]
x3 = [i for i in range(-200, 1001, 1)]

gs = GridSpec(5, 6, figure=fig)

ax1 = fig.add_subplot(gs[:2, :2])
ax1.set_title(r'$w1\ Comparison\ (w1\ model)$', fontsize=18)
ax1.set_xlabel(r'$Time/ms$', fontsize=14)
ax1.set_ylabel(r'$Amplitude/μV$', fontsize=14)
ax1.plot(x1, np.mean(w1_o[23,:,:], axis=0), label=r'$Original$',
         linewidth=1.5, color='dodgerblue')
ax1.plot(x1, np.mean(w1_PO4_w1[23,:,:], axis=0), label=r'$Estimate$',
         linewidth=1.5, color='coral')
ax1.plot(x1, np.mean(w1_PO4_only_w1[23,:,:], axis=0), label=r'$Separate$',
         linewidth=2, color='red')
ax1.tick_params(axis='both', labelsize=14)
ax1.legend(loc='lower left', fontsize=14)

ax2 = fig.add_subplot(gs[:2, 2:4])
ax2.set_title(r'$w2\ Comparison\ (w1\ model)$', fontsize=18)
ax2.set_xlabel(r'$Time/ms$', fontsize=14)
ax2.set_ylabel(r'$Amplitude/μV$', fontsize=14)
ax2.plot(x2, np.mean(w2_o[23,:,:], axis=0), label=r'$Original$',
         linewidth=1.5, color='dodgerblue')
ax2.plot(x2, np.mean(w2_PO4_w1[23,:,:], axis=0), label=r'$Estimate$',
         linewidth=1.5, color='coral')
ax2.plot(x2, np.mean(w2_PO4_only_w1w2[23,:,:], axis=0), label=r'$Separate$',
         linewidth=2, color='red')
ax2.tick_params(axis='both', labelsize=14)
ax2.legend(loc='lower right', fontsize=14)

ax3 = fig.add_subplot(gs[:2, 4:6])
ax3.set_title(r'$w2\ Comparison\ (w2\ model)$', fontsize=18)
ax3.set_xlabel(r'$Time/ms$', fontsize=14)
ax3.set_ylabel(r'$Amplitude/μV$', fontsize=14)
ax3.plot(x2, np.mean(w2_o[23,:,:], axis=0), label=r'$Original$',
         linewidth=1.5, color='dodgerblue')
ax3.plot(x2, np.mean(w2_PO4_w2[23,:,:], axis=0), label=r'$Estimate$',
         linewidth=1.5, color='coral')
ax3.plot(x2, np.mean(w2_PO4_only_w2w2[23,:,:], axis=0), label=r'$Separate$',
         linewidth=2, color='red')
ax3.tick_params(axis='both', labelsize=14)
ax3.legend(loc='lower right', fontsize=14)

ax4 = fig.add_subplot(gs[2:, :3])
ax4.set_title(r'$Signal\ Comparison\ (w1\ model)$', fontsize=18)
ax4.set_xlabel(r'$Time/ms$', fontsize=14)
ax4.set_ylabel(r'$Amplitude/μV$', fontsize=14)
ax4.plot(x3, np.mean(signal_data[23,:,5,:], axis=0), label=r'$Original$',
         linewidth=1.5, color='dodgerblue')
ax4.plot(x3, np.mean(s_PO4_w1[23,:,:], axis=0), label=r'$Estimate$',
         linewidth=1.5, color='coral')
ax4.plot(x3, np.mean(s_PO4_only_w1[23,:,:], axis=0), label=r'$Separate$',
         linewidth=2, color='red')
ax4.vlines(8.3, -20, 20, colors='black', linestyles='dashed',
           label=r'$Start Point$', linewidth=2)
ax4.tick_params(axis='both', labelsize=14)
ax4.legend(loc='upper right', fontsize=14)

ax5 = fig.add_subplot(gs[2:, 3:])
ax5.set_title(r'$Signal\ Comparison\ (w2\ model)$', fontsize=18)
ax5.set_xlabel(r'$Time/ms$', fontsize=14)
ax5.set_ylabel(r'$Amplitude/μV$', fontsize=14)
ax5.plot(x3, np.mean(signal_data[23,:,5,:], axis=0), label=r'$Original$',
         linewidth=1.5, color='dodgerblue')
ax5.plot(x3, np.mean(s_PO4_w2[23,:,:], axis=0), label=r'$Estimate$',
         linewidth=1.5, color='coral')
ax5.plot(x3, np.mean(s_PO4_only_w2[23,:,:], axis=0), label=r'$Separate$',
         linewidth=2, color='red')
ax5.vlines(8.3, -20, 20, colors='black', linestyles='dashed',
           label=r'$Start Point$', linewidth=2)
ax5.tick_params(axis='both', labelsize=14)
ax5.legend(loc='upper right', fontsize=14)

fig.subplots_adjust(left=0.050, bottom=0.070, right=0.980, top=0.960, hspace=1.000, wspace=0.800)
                    
plt.savefig(r'E:\est_bg_data-mqf.png', dpi=600)
plt.show()

#%% model parameter discription (in figure)
'''
1. Boxplot: description of regression model coefficient
2. Histogram: description of the distribution of data
3. Error Bar: description of overall degreee of dispersion
4. 3-D Surface: description of 3D data's trend in space
5. 
'''

#%% Notched-boxplot of R^2 (w1 & w2) & Histogram
RCw1 = R2_w1.flatten()
RCw2 = R2_w2.flatten()

fig = plt.figure(figsize=(24,12))

gs = GridSpec(5, 8, figure=fig)

ax1 = fig.add_subplot(gs[0:2, 0:4])
ax1.set_title(r"$w1\ model's\ R^2$", fontsize=26)
ax1.set_xlabel(r'$\ 1350(27*50)\ models$', fontsize=20)
ax1.set_ylabel(r'$\ R^2\ value$', fontsize=20)
ax1.tick_params(axis='both', labelsize=18)
ax1.set_xlim((0.60, 1.03))
ax1 = sns.boxplot(data=RCw1, notch=True, linewidth=2.5, orient='h')
ax1 = sns.swarmplot(data=RCw1, color='dimgrey', orient='h', size=3, label=r'$Actual\ Point$')
ax1.legend(loc='upper left', fontsize=16)

ax2 = fig.add_subplot(gs[0:2, 4:8])
ax2.set_title(r"$w2\ model's\ R^2$", fontsize=26)
ax2.set_xlabel(r'$\ 1350(27*50)\ models$', fontsize=20)
ax2.set_ylabel(r'$\ R^2\ value$', fontsize=20)
ax2.tick_params(axis='both', labelsize=18)
ax2.set_xlim((0.60, 1.03))
ax2 = sns.boxplot(data=RCw2, notch=True, linewidth=2.5, orient='h')
ax2 = sns.swarmplot(data=RCw2, color='dimgrey', orient='h', size=3, label=r'$Actual\ Point$')
ax2.legend(loc='upper left', fontsize=16)

ax3 = fig.add_subplot(gs[2:, 0:4])
ax3.set_xlabel(r'$\ R^2\ value$', fontsize=20)
ax3.set_ylabel(r'$\ Statistical\ frequency$', fontsize=20)
ax3.tick_params(axis='both', labelsize=18)
ax3.set_xlim((0.60, 1.03))
ax3 = sns.distplot(RCw1, rug=True, kde_kws={'color':'dimgrey', 'lw':2})

ax4 = fig.add_subplot(gs[2:, 4:8])
ax4.set_xlabel(r'$\ R^2\ value$', fontsize=20)
ax4.set_ylabel(r'$\ Statistical\ frequency$', fontsize=20)
ax4.tick_params(axis='both', labelsize=18)
ax4.set_xlim((0.60, 1.03))
ax4 = sns.distplot(RCw2, rug=True, kde_kws={'color':'dimgrey', 'lw':2})

fig.subplots_adjust(top=0.940, bottom=0.070, left=0.060, right=0.950, hspace=0.800, wspace=1.000)

plt.savefig(r'E:\boxplot&histogram-mqf.png', dpi=600)
plt.show()

#%% zero mean signal sequence
# baseline correction
zm_signal_data = SPF.zero_mean(signal_data)
zm_s_po4_only_w1 = SPF.zero_mean(s_PO4_only_w1)
zm_s_po4_only_w2 = SPF.zero_mean(s_PO4_only_w2)

#%% extract signal's SNR comparison (using superimposed average method)
'''
compare the SNR of original SSVEP signal and the target-only-channel's data
now we're going to compute the SNR of 50 trials' data of 27 stimuli conditions
compare the w1 & w2 model
'''
SNR_o = SPF.snr_sa(signal_data[:,:,5,:])
SNR_w1 = SPF.snr_sa(s_PO4_only_w1)
SNR_w2 = SPF.snr_sa(s_PO4_only_w2)

#%% linear regression model time variation analysis
'''
use w1 model to estimate w1 part's data, then compute the fitting deviation
use w1 model to estimate w2 part's data, and do so
use w2 model to estimate w2 part's data, and do so
'''
#%% power spectral density estimation
'''
Use welch method to compute PSD, target is PO4-only data array
'''
# algorithm initialization
sfreq = 1000
fmin, fmax = 0, 50. 
n_fft = 2048
n_overlap = 0
n_per_seg = 2048

# psd computation
s_o_psds, s_o_freqs = SPF.welch_p(signal_data[:,:,5,:], sfreq=sfreq, fmin=fmin,
                fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)
s_w1_psds, s_w1_freqs = SPF.welch_p(s_PO4_w1, sfreq=sfreq, fmin=fmin,
                fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)
s_w2_psds, s_w2_freqs = SPF.welch_p(s_PO4_w2, sfreq=sfreq, fmin=fmin,
                fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)
s_e_w1_psds, s_e_w1_freqs = SPF.welch_p(s_PO4_only_w1, sfreq=sfreq, fmin=fmin,
                fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)
s_e_w2_psds, s_e_w2_freqs = SPF.welch_p(s_PO4_only_w2, sfreq=sfreq, fmin=fmin,
                fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)

#%% extract signal's SNR comparison (in frequency domain)
# X (n_events, n_epochs, n_times)
# subscript:0,3,6,9,12,15,18,21,24 to 8Hz
# subscript:1,4,7,10,13,16,19,22,25 to 10Hz
# subscript:2,5,8,11,14,17,20,23,26 to 15Hz

def snr_F(X):
    snr = np.zeros((X.shape[0], X.shape[1]))
    K = 101
    for i in range(X.shape[0]):
        if i==0 or i==3 or i==6 or i==9 or i==12 or i==15 or i==18 or i==21 or i==24:
            for j in range(X.shape[1]):  # 8Hz
                target = X[i,j,17] + X[i,j,18]
                snr[i,j] = K * (target / (np.sum(X[i,j,:]) - target))
                
        if i==1 or i==4 or i==7 or i==10 or i==13 or i==16 or i==19 or i==22 or i==25:
            for j in range(X.shape[1]):  # 10Hz
                target = X[i,j,21] + X[i,j,22]
                snr[i,j] = K * (target / (np.sum(X[i,j,:]) - target))
                
        if i==2 or i==5 or i==8 or i==11 or i==14 or i==17 or i==20 or i==23 or i==26:
            for j in range(X.shape[1]):  # 15Hz
                target = X[i,j,30] + X[i,j,31]
                snr[i,j] = K * (target / (np.sum(X[i,j,:]) - target))
    return snr

#%%
SNR_o_freq = 20*np.log10(snr_F(s_o_psds))
SNR_w1_freq = 20*np.log10(snr_F(s_e_w1_psds))
SNR_w2_freq = 20*np.log10(snr_F(s_e_w2_psds))

#%%
def snr_single(X):
    '''
    compute single trial signal's SNR in frequency domain (multi-conditions)
    :param X: input signal' spectrum sequence (n_events, )
    '''
    snr = np.zeros((X.shape[0]))
    K= 101
    for i in range(X.shape[0]):
        if i==0 or i==3 or i==6 or i==9 or i==12 or i==15 or i==18 or i==21 or i==24:
            # 8Hz
            target = (X[i,17] + X[i,18]) / 2
            snr[i] = K * (target / (np.sum(X[i,:]) - target))
                
        if i==1 or i==4 or i==7 or i==10 or i==13 or i==16 or i==19 or i==22 or i==25:
            # 10Hz
            target = (X[i,21] + X[i,22]) / 2
            snr[i] = K * (target / (np.sum(X[i,:]) - target))
                
        if i==2 or i==5 or i==8 or i==11 or i==14 or i==17 or i==20 or i==23 or i==26:
            # 15Hz
            target = (X[i,30] + X[i,31]) / 2
            snr[i] = K * (target / (np.sum(X[i,:]) - target))
    
    return snr

#%%
SNR_mean_o = snr_single(np.mean(s_o_psds, axis=1))
SNR_mean_e1_w1 = snr_single((np.mean(s_o_psds, axis=1)-np.mean(s_w1_psds, axis=1)))
SNR_mean_e1_w2 = snr_single((np.mean(s_o_psds, axis=1)-np.mean(s_w2_psds, axis=1)))
SNR_mean_e_w1 = snr_single(np.mean(s_e_w1_psds, axis=1))
SNR_mean_e_w2 = snr_single(np.mean(s_e_w2_psds, axis=1))

#%% plot psd comparision
fig, axes = plt.subplots(3, 2, figsize=(24,18))

axes[0,0].set_xlabel(r'$\ Frequency/Hz$', fontsize=20)
axes[0,0].set_ylabel(r'$\ Power$', fontsize=20)
axes[0,0].tick_params(axis='both', labelsize=20)
axes[0,0].set_title(r'$\ Power\ Spectrum\ Density\ (8Hz&0.5pi&0.5s)$', fontsize=24)
axes[0,0].plot(s_o_freqs[0,0,:], np.mean(s_o_psds[21,:,:], axis=0),
    color='red', linewidth=2, label=r'$\ Origin$')
axes[0,0].plot(s_o_freqs[0,0,:], np.mean(s_w1_psds[21,:,:], axis=0),
    color='coral', linewidth=2, label=r'$\ w1\ Estimation$')
axes[0,0].plot(s_o_freqs[0,0,:], np.mean(s_w2_psds[21,:,:], axis=0),
    color='dodgerblue', linewidth=2, label=r'$\ w2\ Estimation$')
axes[0,0].legend(loc='best', fontsize=20)

axes[0,1].set_xlabel(r'$\ Frequency/Hz$', fontsize=20)
axes[0,1].set_ylabel(r'$\ Power$', fontsize=20)
axes[0,1].tick_params(axis='both', labelsize=20)
axes[0,1].set_title(r'$\ Power\ Spectrum\ Density\ (8Hz&0.5pi&0.5s)$', fontsize=24)
axes[0,1].plot(s_o_freqs[0,0,:], np.mean(s_e_w1_psds[21,:,:], axis=0),
    color='coral', linewidth=2, label=r'$\ w1\ Extraction$')
axes[0,1].plot(s_o_freqs[0,0,:], np.mean(s_e_w2_psds[21,:,:], axis=0),
    color='dodgerblue', linewidth=2, label=r'$\ w2\ Extraction$')
axes[0,1].legend(loc='best', fontsize=20)

axes[1,0].set_xlabel(r'$\ Frequency/Hz$', fontsize=20)
axes[1,0].set_ylabel(r'$\ Power$', fontsize=20)
axes[1,0].tick_params(axis='both', labelsize=20)
axes[1,0].set_title(r'$\ Power\ Spectrum\ Density\ (10Hz&0.5pi&0.5s)$', fontsize=24)
axes[1,0].plot(s_o_freqs[0,0,:], np.mean(s_o_psds[22,:,:], axis=0),
    color='red', linewidth=2, label=r'$\ Origin$')
axes[1,0].plot(s_o_freqs[0,0,:], np.mean(s_w1_psds[22,:,:], axis=0),
    color='coral', linewidth=2, label=r'$\ w1\ Estimation$')
axes[1,0].plot(s_o_freqs[0,0,:], np.mean(s_w2_psds[22,:,:], axis=0),
    color='dodgerblue', linewidth=2, label=r'$\ w2\ Estimation$')
axes[1,0].legend(loc='best', fontsize=20)

axes[1,1].set_xlabel(r'$\ Frequency/Hz$', fontsize=20)
axes[1,1].set_ylabel(r'$\ Power$', fontsize=20)
axes[1,1].tick_params(axis='both', labelsize=20)
axes[1,1].set_title(r'$\ Power\ Spectrum\ Density\ (10Hz&0.5pi&0.5s)$', fontsize=24)
axes[1,1].plot(s_o_freqs[0,0,:], np.mean(s_e_w1_psds[22,:,:], axis=0),
    color='coral', linewidth=2, label=r'$\ w1\ Extraction$')
axes[1,1].plot(s_o_freqs[0,0,:], np.mean(s_e_w2_psds[22,:,:], axis=0),
    color='dodgerblue', linewidth=2, label=r'$\ w2\ Extraction$')
axes[1,1].legend(loc='best', fontsize=20)

axes[2,0].set_xlabel(r'$\ Frequency/Hz$', fontsize=20)
axes[2,0].set_ylabel(r'$\ Power$', fontsize=20)
axes[2,0].tick_params(axis='both', labelsize=20)
axes[2,0].set_title(r'$\ Power\ Spectrum\ Density\ (15Hz&0.5pi&0.5s)$', fontsize=24)
axes[2,0].plot(s_o_freqs[0,0,:], np.mean(s_o_psds[23,:,:], axis=0),
    color='red', linewidth=2, label=r'$\ Origin$')
axes[2,0].plot(s_o_freqs[0,0,:], np.mean(s_w1_psds[23,:,:], axis=0),
    color='coral', linewidth=2, label=r'$\ w1\ Estimation$')
axes[2,0].plot(s_o_freqs[0,0,:], np.mean(s_w2_psds[23,:,:], axis=0),
    color='dodgerblue', linewidth=2, label=r'$\ w2\ Estimation$')
axes[2,0].legend(loc='best', fontsize=20)

axes[2,1].set_xlabel(r'$\ Frequency/Hz$', fontsize=20)
axes[2,1].set_ylabel(r'$\ Power$', fontsize=20)
axes[2,1].tick_params(axis='both', labelsize=20)
axes[2,1].set_title(r'$\ Power\ Spectrum\ Density\ (15Hz&0.5pi&0.5s)$', fontsize=24)
axes[2,1].plot(s_o_freqs[0,0,:], np.mean(s_e_w1_psds[23,:,:], axis=0),
    color='coral', linewidth=2, label=r'$\ w1\ Extraction$')
axes[2,1].plot(s_o_freqs[0,0,:], np.mean(s_e_w2_psds[23,:,:], axis=0),
    color='dodgerblue', linewidth=2, label=r'$\ w2\ Extraction$')
axes[2,1].legend(loc='best', fontsize=20)

fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95,
                    hspace=0.5, wspace=0.2)
plt.savefig(r'E:/psd-comparision-mqf.png', dpi=600)
plt.show()

#%% plot SNR comparision
fig, axes = plt.subplots(2, 2, figsize=(20,15))

axes[0,0].set_xlabel(r'$\ Trials$', fontsize=20)
axes[0,0].set_ylabel(r'$\ SNR/dB$', fontsize=20)
axes[0,0].tick_params(axis='both', labelsize=20)
axes[0,0].set_title(r'$\ Promotion\ of\ SNR(inter-conditions)$', fontsize=24)
axes[0,0].plot(np.mean(SNR_w1_freq-SNR_o_freq, axis=0), linewidth=2,
    label=r'$\ w1-Origin$')
axes[0,0].plot(np.mean(SNR_w2_freq-SNR_o_freq, axis=0), linewidth=2,
    label=r'$\ w2-Origin$')
axes[0,0].hlines(0, 0, 50, colors='dimgrey', linestyles='dashed', linewidth=3)
axes[0,0].legend(loc='best', fontsize=20)

axes[0,1].set_xlabel(r'$\ Trials$', fontsize=20)
axes[0,1].set_ylabel(r'$\ SNR/dB$', fontsize=20)
axes[0,1].tick_params(axis='both', labelsize=20)
axes[0,1].set_title(r'$\ Comparision\ of\ models(inter-conditions)$', fontsize=24)
axes[0,1].plot(np.mean(SNR_w2_freq-SNR_w1_freq, axis=0), linewidth=2, label=r'$\ w2-w1$')
axes[0,1].hlines(0, 0, 50, colors='dimgrey', linestyles='dashed', linewidth=3)
axes[0,1].legend(loc='best', fontsize=20)

axes[1,0].set_xlabel(r'$\ Conditions$', fontsize=20)
axes[1,0].set_ylabel(r'$\ SNR/dB$', fontsize=20)
axes[1,0].tick_params(axis='both', labelsize=20)
axes[1,0].set_title(r'$\ Promotion\ of\ SNR(inter-trials)$', fontsize=24)
axes[1,0].plot(np.mean(SNR_w1_freq-SNR_o_freq, axis=1), linewidth=2,
    label=r'$\ w1-Origin$')
axes[1,0].plot(np.mean(SNR_w2_freq-SNR_o_freq, axis=1), linewidth=2,
    label=r'$\ w2-Origin$')
axes[1,0].hlines(0, 0, 26, colors='dimgrey', linestyles='dashed', linewidth=3)
axes[1,0].legend(loc='best', fontsize=20)

axes[1,1].set_xlabel('Conditions', fontsize=20)
axes[1,1].set_ylabel('SNR/dB', fontsize=20)
axes[1,1].tick_params(axis='both', labelsize=20)
axes[1,1].set_title(r'$\ Comparision\ of\ models(inter-trials)$', fontsize=24)
axes[1,1].plot(np.mean(SNR_w2_freq-SNR_w1_freq, axis=1), linewidth=2, label=r'$\ w2-w1$')
axes[1,1].hlines(0, 0, 26, colors='dimgrey', linestyles='dashed', linewidth=3)
axes[1,1].legend(loc='best', fontsize=20)

fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.95,
                    hspace=0.33, wspace=0.13)
plt.savefig(r'E:/snr-comparision-mqf.png', dpi=600)
plt.show()

#%% time-frequency transform
'''
Use morlet wavelet method to compute TFR
Inside params n_cyles varies for each frequency condition
'''
# frequency initialization for tfr
freqs = np.arange(1., 30, 0.05)
n_cycles = freqs/2.
times = np.linspace(-200, 1000, 1200)

# power of extract PO4 data using w1 model (return expansion, correct manually)
pew1_po4 = SPF.tfr_analysis(s_PO4_only_w1)
# power of extract PO4 data using w1 model (return expansion, correct manually)
pew2_po4 = SPF.tfr_analysis(s_PO4_only_w2)

vmin = np.min()
vmax = np.max()

fig, axes = plt.subplots(3, 3, figsize=(18,15))

mesh = axes[0,0].pcolormesh(times * 1000, freqs, power_f8d6[1], cmap='RdBu_r',
                    vmin=0, vmax=vmax)
axes[0,0].set_title('8Hz & 100ms', fontsize=24)
axes[0,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,0].set_ylabel('ITC/Frequency', fontsize=20)
axes[0,0].tick_params(axis='both', labelsize=20)
axes[0,0].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[0,0])


#%%
def fangcha(X):
    '''
    X:()
    '''
    EX = np.mean(X,axis=0)
    for i in range(X.shape[1]):
        X[:,i] = X[:,i] - EX[i]

    X = X**2
    return X
#%%
def snr(X):
    '''
    X: (n_epochs, n_times)
    '''
    EX = np.mean(X, axis=0)
    EX = EX**2
    Y = np.zeros(())
    for i in range(EX.shape[0]):
        Y[:,i] = X[:,i]-EX[i]
    Y = Y**2
    EX1 = np.mean(Y, axis=0)
    snr = EX/EX1
    return snr

#%%
X1 = signal_data[26,:,5,:]
X2 = s_PO4_only_w1[26,:,:]
X3 = s_PO4_only_w2[26,:,:]

#%%
snr1 = snr(X1)
snr2 = snr(X2)
snr3 = snr(X3)

#%%
Y1 = np.mean(fangcha(X1), axis=0)
Y2 = np.mean(fangcha(X2), axis=0)
Y3 = np.mean(fangcha(X3), axis=0)

#%%
fig, axes = plt.subplots(2,1)
axes[0].plot(Y1, label='origin')

axes[1].plot(Y3, label='w2 model')
#plt.plot(Y2, label='w1 model')
#plt.plot(Y3, label='w2 model')
#plt.legend(loc='best')
#%%
fig, axes = plt.subplots(3,1)

#axes[0].plot(np.log10(snr1), label='origin', color='red')
axes[0].plot(snr1, label='origin', color='red')
axes[0].set_xlabel('time/ms')
axes[0].set_ylabel('SNR')
axes[0].vlines(808.3, 0, 1, color='black', linestyles='dashed', label='start point')
axes[0].legend(loc='best')

#xes[1].plot(np.log10(snr2), label='w1', color='blue')
axes[1].plot(snr2, label='w1', color='blue')
axes[1].set_xlabel('time/ms')
axes[1].set_ylabel('SNR')
axes[1].vlines(808.3, 0, 1, color='black', linestyles='dashed', label='start point')
axes[1].legend(loc='best')

#axes[2].plot(np.log10(snr3), label='w2', color='green')
axes[2].plot(snr3, label='w2', color='green')
axes[2].set_xlabel('time/ms')
axes[2].set_ylabel('SNR')
axes[2].vlines(808.3, 0, 1, color='black', linestyles='dashed', label='start point')
axes[2].legend(loc='best')
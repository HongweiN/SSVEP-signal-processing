# -*- coding: utf-8 -*-
"""
Created on Mon March 25 16:00:31 2019

@author: Brynhildr
"""
#%%
import os
import numpy as np
import pylab as pl
import scipy.io as io
import mne
from mne.io import concatenate_raws
from mne import Epochs, pick_types, find_events
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mne.time_frequency import (tfr_multitaper, tfr_stockwell,
                tfr_morlet, tfr_array_morlet, psd_array_welch)
from mne.baseline import rescale

# %% load data
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
    # misc=['CB1', 'CB2', 'M1', 'M2'],
    raw_cnts.append(raw_cnt)

raw = concatenate_raws(raw_cnts)

del raw_cnts

#%% prepare for average data
# raw.set_eeg_reference(ref_channels='average', projection=True)
raw.filter(l_freq=0.5, h_freq=20., method='fir', phase='zero',
           fir_window='hamming', fir_design='firwin',
           skip_by_annotation='edge')
events = mne.find_events(raw, output='onset')
# stim_channel='STI 014',

raw.pick_types(raw.info, emg=False, eeg=True,
               stim=False, eog=False)

# drop channels
drop_chans = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5',
    'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5',
    'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'M2']
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

baseline = (-0.2, 0)    # define baseline
tmin, tmax = -0.2, 1    # set the time range

# make averaged data for plotting
averaged_data = np.zeros((27, 1201))

#%% store data into array
pick_chans = ['O1','O2','OZ','POZ','PO3','PO4']    # pick the channel to show

for i in range(len(event_id)):
    
    #event_choose = event_id[IDname]
    
    # make epoch data
    epochs = Epochs(raw, events=events, event_id=i+1, tmin=tmin,
                    tmax=tmax, baseline=baseline, preload=True)
    epochs_copy = epochs.copy().pick_channels(pick_chans)  # pich channels
    evoked_data = epochs_copy.get_data()  # get the 3D array of data
    # (n_trials, n_channels, timepoints)
    # evoked_data[0:49, :, :]  # choose the first testee
    
    # store the evoke data array into averaged_data
    averaged_data[i,:] = np.mean(np.mean(evoked_data, axis=1), axis=0)
    del epochs, epochs_copy, evoked_data
del i
#%%
pick_chans = ['O1','O2','OZ','POZ','PO3','PO4']
epochs = Epochs(raw, events=events, event_id=24, tmin=tmin,
                    tmax=tmax, baseline=baseline, preload=True)
epochs_copy = epochs.copy().pick_channels(pick_chans)  # pich channels
evoked_data = epochs_copy.get_data()  # get the 3D array of data
test_data = np.mean(evoked_data,axis=0)
#%%
test_data *= 1e6
#%%
plt.plot(test_data[0,:],label='O1',linewidth='1')
plt.plot(test_data[1,:],label='O2',linewidth='1')
plt.plot(test_data[2,:],label='OZ',linewidth='1')
plt.legend(loc='upper right', fontsize=15)

#%%
averaged_data = averaged_data*1e6  # transform the amplitude

#%% make ssvep data (fundamental frequency) & erp data
ssvep_data = np.zeros((9, 1201))

ssvep_data[0,:] = (averaged_data[0,:]-averaged_data[6,:])/2   # 8Hz 100ms
ssvep_data[1,:] = (averaged_data[9,:]-averaged_data[15,:])/2   # 8Hz 300ms
ssvep_data[2,:] = (averaged_data[18,:]-averaged_data[24,:])/2   # 8Hz 500ms
ssvep_data[3,:] = (averaged_data[1,:]-averaged_data[7,:])/2   # 10Hz 100ms
ssvep_data[4,:] = (averaged_data[10,:]-averaged_data[16,:])/2   # 10Hz 300ms
ssvep_data[5,:] = (averaged_data[19,:]-averaged_data[25,:])/2   # 10Hz 500ms
ssvep_data[6,:] = (averaged_data[2,:]-averaged_data[8,:])/2   # 15Hz 100ms
ssvep_data[7,:] = (averaged_data[11,:]-averaged_data[17,:])/2   # 15Hz 300ms
ssvep_data[8,:] = (averaged_data[20,:]-averaged_data[26,:])/2   # 15Hz 500ms

erp_data = np.zeros((9, 1201))
erp_data[0,:] = (averaged_data[0,:]+averaged_data[6,:])/2   # 8Hz 100ms
erp_data[1,:] = (averaged_data[9,:]+averaged_data[15,:])/2   # 8Hz 300ms
erp_data[2,:] = (averaged_data[18,:]+averaged_data[24,:])/2   # 8Hz 500ms
erp_data[3,:] = (averaged_data[1,:]+averaged_data[7,:])/2   # 10Hz 100ms
erp_data[4,:] = (averaged_data[10,:]+averaged_data[16,:])/2   # 10Hz 300ms
erp_data[5,:] = (averaged_data[19,:]+averaged_data[25,:])/2   # 10Hz 500ms
erp_data[6,:] = (averaged_data[2,:]+averaged_data[8,:])/2   # 15Hz 100ms
erp_data[7,:] = (averaged_data[11,:]+averaged_data[17,:])/2   # 15Hz 300ms
erp_data[8,:] = (averaged_data[20,:]+averaged_data[26,:])/2   # 15Hz 500ms

#%% prepare for plotting: Nomalization
def Normalization(x, y, z):
    line = np.zeros((3,1201))
    line[0,:] = x  # 8Hz
    line[1,:] = y  # 10Hz
    line[2,:] = z  # 15Hz
    Min = np.min(line)
    Max = np.max(line)
    for i in range(0,3):
        line[i,:] = -1 + 2*(line[i,:]-Min)/(Max-Min)
    del i, Min, Max
    return line

def AntiphaseNormalization(x, y):
    line = np.zeros((2,1201))
    line[0,:] = x
    line[1,:] = y
    Min = np.min(line)
    Max = np.max(line)
    for i in range(0,2):
        line[i,:] = -1 + 2*(line[i,:]-Min)/(Max-Min)
    del i, Min, Max
    return line

def SSVEP_Normalization(x):
    Min = np.min(x)
    Max = np.max(x)
    x = -1 + 2*(x-Min)/(Max-Min)
    del Min, Max
    return x

#%% task 1: same time&phase, different frequency
# 0.5pi initial phase
line1 = Normalization(averaged_data[3,:], averaged_data[4,:],
                      averaged_data[5,:])
line2 = Normalization(averaged_data[12,:], averaged_data[13,:],
                      averaged_data[14,:])
line3 = Normalization(averaged_data[21,:], averaged_data[22,:],
                     averaged_data[23,:])

#%% 0 initial phase
line1 = Normalization(averaged_data[3,:], averaged_data[4,:],
                      averaged_data[5,:])
line2 = Normalization(averaged_data[12,:], averaged_data[13,:],
                      averaged_data[14,:])
line3 = Normalization(averaged_data[21,:], averaged_data[22,:],
                     averaged_data[23,:])

#%% pi initial phase
line1 = Normalization(averaged_data[3,:], averaged_data[4,:],
                      averaged_data[5,:])
line2 = Normalization(averaged_data[12,:], averaged_data[13,:],
                      averaged_data[14,:])
line3 = Normalization(averaged_data[21,:], averaged_data[22,:],
                     averaged_data[23,:])

#%%
fig, axes = plt.subplots(3, 1, figsize=(15,21))
x=[i for i in range(-200, 1001, 1)]

# plotting
axes[0].set(xlim=[-250,1050], ylim=[-1.15,1.15]) 
axes[0].set_xlabel('Time/ms', fontsize=26)
axes[0].set_ylabel('Normalized Amplitude', fontsize=26)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_title('100ms & 0.5pi initial phase', fontsize=30)
axes[0].plot(x, line1[0,:], color='red', label='8Hz', linewidth=2)
axes[0].plot(x, line1[1,:], color='blue', label='10Hz', linewidth=2)
axes[0].plot(x, line1[2,:], color='green', label='15Hz', linewidth=2)
axes[0].vlines(8.3, -1., 1., colors='black', linestyles='dashed',
            label='start point', linewidth=2)
axes[0].legend(loc='1', fontsize=20)

axes[1].set(xlim=[-250,1050], ylim=[-1.15,1.15]) 
axes[1].set_xlabel('Time/ms', fontsize=26)
axes[1].set_ylabel('Normalized Amplitude', fontsize=26)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_title('300ms & 0.5pi initial phase', fontsize=30)
axes[1].plot(x, line2[0,:], color='red', label='8Hz', linewidth=2)
axes[1].plot(x, line2[1,:], color='blue', label='10Hz', linewidth=2)
axes[1].plot(x, line2[2,:], color='green', label='15Hz', linewidth=2)
axes[1].vlines(8.3, -1., 1., colors='black', linestyles='dashed',
            label='start point', linewidth=2)
axes[1].legend(loc='1', fontsize=20)

axes[2].set(xlim=[-250,1050], ylim=[-1.15,1.15]) 
axes[2].set_xlabel('Time/ms', fontsize=26)
axes[2].set_ylabel('Normalized Amplitude', fontsize=26)
axes[2].tick_params(axis='both', labelsize=20)
axes[2].set_title('500ms & 0.5pi initial phase', fontsize=30)
axes[2].plot(x, line3[0,:], color='red', label='8Hz', linewidth=2)
axes[2].plot(x, line3[1,:], color='blue', label='10Hz', linewidth=2)
axes[2].plot(x, line3[2,:], color='green', label='15Hz', linewidth=2)
axes[2].vlines(8.3, -1., 1., colors='black', linestyles='dashed',
            label='start point', linewidth=2)
axes[2].legend(loc='1', fontsize=20)  # 显示图例

# adjust position and paddings
fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, hspace=0.3)
plt.savefig(r'E:/same time&phase.png', dpi=600)

plt.show()

#%% task 2: same frequency&phase, different time
line1 = Normalization(averaged_data[3,:], averaged_data[12,:],
                      averaged_data[21,:])
line2 = Normalization(averaged_data[4,:], averaged_data[13,:],
                      averaged_data[22,:])
line3 = Normalization(averaged_data[5,:], averaged_data[14,:],
                     averaged_data[23,:])

x=[i for i in range(-200, 1001, 1)]
fig, axes = plt.subplots(3, 1, figsize=(15,21))

# plotting
axes[0].set(xlim=[-250,1050], ylim=[-1.15,1.15]) 
axes[0].set_xlabel('Time/ms', fontsize=26)
axes[0].set_ylabel('Normalized Amplitude', fontsize=26)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_title('8Hz & 0.5pi initial phase', fontsize=30)
axes[0].plot(x, line1[0,:], color='red', label='100ms', linewidth=2)
axes[0].plot(x, line1[1,:], color='blue', label='300ms', linewidth=2)
axes[0].plot(x, line1[2,:], color='green', label='500ms', linewidth=2)
axes[0].vlines(8.3, -1., 1., colors='black', linestyles='dashed',
            label='start point', linewidth=2)
axes[0].legend(loc='upper right', fontsize=20)

axes[1].set(xlim=[-250,1050], ylim=[-1.15,1.15]) 
axes[1].set_xlabel('Time/ms', fontsize=26)
axes[1].set_ylabel('Normalized Amplitude', fontsize=26)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_title('10Hz & 0.5pi initial phase', fontsize=30)
axes[1].plot(x, line2[0,:], color='red', label='100ms', linewidth=2)
axes[1].plot(x, line2[1,:], color='blue', label='300ms', linewidth=2)
axes[1].plot(x, line2[2,:], color='green', label='500ms', linewidth=2)
axes[1].vlines(8.3, -1., 1., colors='black', linestyles='dashed',
            label='start point', linewidth=2)
axes[1].legend(loc='upper right', fontsize=20)

axes[2].set(xlim=[-250,1050], ylim=[-1.15,1.15]) 
axes[2].set_xlabel('Time/ms', fontsize=26)
axes[2].set_ylabel('Normalized Amplitude', fontsize=26)
axes[2].tick_params(axis='both', labelsize=20)
axes[2].set_title('15Hz & 0.5pi initial phase', fontsize=30)
axes[2].plot(x, line3[0,:], color='red', label='100ms', linewidth=2)
axes[2].plot(x, line3[1,:], color='blue', label='300ms', linewidth=2)
axes[2].plot(x, line3[2,:], color='green', label='500ms', linewidth=2)
axes[2].vlines(8.3, -1., 1., colors='black', linestyles='dashed',
            label='start point', linewidth=2)
axes[2].legend(loc='upper right', fontsize=20)  # 显示图例

# adjust position and paddings
fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, hspace=0.3)
plt.savefig(r'E:/same freq&phase.png', dpi=600)

plt.show()

#%% task 3: antiphase method
line = AntiphaseNormalization(1, 7)

# initialization
plt.figure(figsize=(16,9))
plt.xlabel('Time/ms', fontsize=24)
plt.ylabel('Normalized Amplitude', fontsize=24)
plt.tick_params(axis='both', labelsize=16)

plt.title('Raw data:8Hz&100ms', fontsize=24)

plt.plot(line[0,:], color='red', label='0 initial phase')
plt.plot(line[1,:], color='blue', label='π initial phase')
plt.vlines(208.3, -1., 1., colors='black',
           linestyles='dashed',label='start point')
plt.legend(fontsize=24)  # 显示图例
plt.show()

#%% task3(2): comparision
# Normalization
line = SSVEP_Normalization(ssvep_data[2,:])

# initialization
plt.figure(figsize=(16,9))
plt.xlabel('Time/ms', fontsize=24)
plt.ylabel('Normalized Amplitude', fontsize=24)
plt.tick_params(axis='both', labelsize=16)

plt.title('SSVEP:8Hz&500ms', fontsize=24)

plt.vlines(208.3, -1., 1., colors='black',
           linestyles='dashed',label='start point')
plt.plot(line, label='ssvep signal')
plt.legend(fontsize=24)  # 显示图例

plt.show()

#plt.plot(ssvep_data[0,:])   # 8Hz 100ms
#plt.plot(ssvep_data[1,:])   # 8Hz 300ms
#plt.plot(ssvep_data[2,:])   # 8Hz 500ms
#plt.plot(ssvep_data[3,:])   # 10Hz 100ms
#plt.plot(ssvep_data[4,:])   # 10Hz 300ms
#plt.plot(ssvep_data[5,:])   # 10Hz 500ms
#plt.plot(ssvep_data[6,:])   # 15Hz 100ms
#plt.plot(ssvep_data[7,:])   # 15Hz 300ms
#plt.plot(ssvep_data[8,:])   # 15Hz 500ms

#%% task3(3):ERP
# Normalization
line = SSVEP_Normalization(erp_data[8,:])

# initialization
plt.figure(figsize=(16,9))
plt.xlabel('Time/ms', fontsize=24)
plt.ylabel('Normalized Amplitude', fontsize=24)
plt.tick_params(axis='both', labelsize=16)

plt.title('ERP:15Hz&500ms', fontsize=24)

plt.vlines(208.3, -1., 1., colors='black',
           linestyles='dashed',label='start point')
plt.plot(erp_data[8,:], label='erp signal')
plt.legend(fontsize=24)  # 显示图例

plt.show()

#%% ssvep's psd plot, using welch method(use subplot)
# prepare for psd data
psds = np.zeros((103,27))
freqs = np.zeros((103,27))
for i in range(0,27):
    psds[:,i], freqs[:,i] = psd_array_welch(averaged_data[i,:], sfreq=1000,
                    fmin=0, fmax=50, n_fft=2048, n_overlap=300, n_per_seg=600)
    
del i

# initialization
fig, axes = plt.subplots(3, 3, figsize=(18,15))

# plotting
axes[0,0].set(xlim=[0,25], ylim=[0,3])
axes[0,0].set_title('100ms & 0 initial phase', fontsize=24)
axes[0,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,0].set_ylabel('Power/Frequency', fontsize=20)
axes[0,0].plot(freqs[:,0], psds[:,0], label='8Hz', color='red', linewidth=2)
axes[0,0].plot(freqs[:,1], psds[:,1], label='10Hz', color='blue', linewidth=2)
axes[0,0].plot(freqs[:,2], psds[:,2], label='15Hz', color='green', linewidth=2)
axes[0,0].tick_params(axis='both', labelsize=20)
axes[0,0].legend(fontsize=18)

axes[0,1].set(xlim=[0,25], ylim=[0,3])
axes[0,1].set_title('100ms & 0.5π initial phase', fontsize=24)
axes[0,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,1].set_ylabel('Power/Frequency', fontsize=20)
axes[0,1].plot(freqs[:,3], psds[:,3], label='8Hz', color='red', linewidth=2)
axes[0,1].plot(freqs[:,4], psds[:,4], label='10Hz', color='blue', linewidth=2)
axes[0,1].plot(freqs[:,5], psds[:,5], label='15Hz', color='green', linewidth=2)
axes[0,1].tick_params(axis='both', labelsize=20)
axes[0,1].legend(fontsize=18)

axes[0,2].set(xlim=[0,25], ylim=[0,3])
axes[0,2].set_title('100ms & π initial phase', fontsize=24)
axes[0,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,2].set_ylabel('Power/Frequency', fontsize=20)
axes[0,2].plot(freqs[:,6], psds[:,6], label='8Hz', color='red', linewidth=2)
axes[0,2].plot(freqs[:,7], psds[:,7], label='10Hz', color='blue', linewidth=2)
axes[0,2].plot(freqs[:,8], psds[:,8], label='15Hz', color='green', linewidth=2)
axes[0,2].tick_params(axis='both', labelsize=20)
axes[0,2].legend(fontsize=18)

axes[1,0].set(xlim=[0,25], ylim=[0,3])
axes[1,0].set_title('300ms & 0 initial phase', fontsize=24)
axes[1,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,0].set_ylabel('Power/Frequency', fontsize=20)
axes[1,0].plot(freqs[:,9], psds[:,9], label='8Hz', color='red', linewidth=2)
axes[1,0].plot(freqs[:,10], psds[:,10], label='10Hz', color='blue', linewidth=2)
axes[1,0].plot(freqs[:,11], psds[:,11], label='15Hz', color='green', linewidth=2)
axes[1,0].tick_params(axis='both', labelsize=20)
axes[1,0].legend(fontsize=18)

axes[1,1].set(xlim=[0,25], ylim=[0,3])
axes[1,1].set_title('300ms & 0.5π initial phase', fontsize=24)
axes[1,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,1].set_ylabel('Power/Frequency', fontsize=20)
axes[1,1].plot(freqs[:,12], psds[:,12], label='8Hz', color='red', linewidth=2)
axes[1,1].plot(freqs[:,13], psds[:,13], label='10Hz', color='blue', linewidth=2)
axes[1,1].plot(freqs[:,14], psds[:,14], label='15Hz', color='green', linewidth=2)
axes[1,1].tick_params(axis='both', labelsize=20)
axes[1,1].legend(fontsize=18)

axes[1,2].set(xlim=[0,25], ylim=[0,3])
axes[1,2].set_title('300ms & π initial phase', fontsize=24)
axes[1,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,2].set_ylabel('Power/Frequency', fontsize=20)
axes[1,2].plot(freqs[:,15], psds[:,15], label='8Hz', color='red', linewidth=2)
axes[1,2].plot(freqs[:,16], psds[:,16], label='10Hz', color='blue', linewidth=2)
axes[1,2].plot(freqs[:,17], psds[:,17], label='15Hz', color='green', linewidth=2)
axes[1,2].tick_params(axis='both', labelsize=20)
axes[1,2].legend(fontsize=18)

axes[2,0].set(xlim=[0,25], ylim=[0,3])
axes[2,0].set_title('500ms & 0 initial phase', fontsize=24)
axes[2,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[2,0].set_ylabel('Power/Frequency', fontsize=20)
axes[2,0].plot(freqs[:,18], psds[:,18], label='8Hz', color='red', linewidth=2)
axes[2,0].plot(freqs[:,19], psds[:,19], label='10Hz', color='blue', linewidth=2)
axes[2,0].plot(freqs[:,20], psds[:,20], label='15Hz', color='green', linewidth=2)
axes[2,0].tick_params(axis='both', labelsize=20)
axes[2,0].legend(fontsize=18)

axes[2,1].set(xlim=[0,25], ylim=[0,3])
axes[2,1].set_title('500ms & 0.5π initial phase', fontsize=24)
axes[2,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[2,1].set_ylabel('Power/Frequency', fontsize=20)
axes[2,1].plot(freqs[:,21], psds[:,21], label='8Hz', color='red', linewidth=2)
axes[2,1].plot(freqs[:,22], psds[:,22], label='10Hz', color='blue', linewidth=2)
axes[2,1].plot(freqs[:,23], psds[:,23], label='15Hz', color='green', linewidth=2)
axes[2,1].tick_params(axis='both', labelsize=20)
axes[2,1].legend(fontsize=18)

axes[2,2].set(xlim=[0,25], ylim=[0,3])
axes[2,2].set_title('500ms & π initial phase', fontsize=24)
axes[2,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[2,2].set_ylabel('Power/Frequency', fontsize=20)
axes[2,2].plot(freqs[:,24], psds[:,24], label='8Hz', color='red', linewidth=2)
axes[2,2].plot(freqs[:,25], psds[:,25], label='10Hz', color='blue', linewidth=2)
axes[2,2].plot(freqs[:,26], psds[:,26], label='15Hz', color='green', linewidth=2)
axes[2,2].tick_params(axis='both', labelsize=20)
axes[2,2].legend(fontsize=18)

# adjust position and paddings
fig.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.96,
                    hspace=0.4, wspace=0.3)
plt.savefig(r'E:/time&phase.png', dpi=600)
plt.show()

#%% antiphase method & erp's psd plotting
# prepare for psd data
psd_ssvep = np.zeros((103,9))
psd_erp = np.zeros((103,9))
freq_ssvep = np.zeros((103,9))
freq_erp = np.zeros((103,9))

for i in range(0,9):
    psd_ssvep[:,i], freq_ssvep[:,i] = psd_array_welch(ssvep_data[i,:], sfreq=1000,
                    fmin=0, fmax=50, n_fft=2048, n_overlap=300, n_per_seg=600)
del i

for i in range(0,9):
    psd_erp[:,i], freq_erp[:,i] = psd_array_welch(erp_data[i,:], sfreq=1000,
                    fmin=0, fmax=50, n_fft=2048, n_overlap=300, n_per_seg=600)
del i    

# initialization
fig, axes = plt.subplots(2, 3, figsize=(18,10))

# plotting
axes[0,0].set(xlim=[0,25], ylim=[0,2])
axes[0,0].set_title('8Hz SSVEP', fontsize=24)
axes[0,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,0].set_ylabel('Power/Frequency', fontsize=20)
axes[0,0].plot(freq_ssvep[:,0], psd_ssvep[:,0], label='100ms',color='red', linewidth=2)
axes[0,0].plot(freq_ssvep[:,1], psd_ssvep[:,1], label='300ms', color='blue', linewidth=2)
axes[0,0].plot(freq_ssvep[:,2], psd_ssvep[:,2], label='500ms', color='green', linewidth=2)
axes[0,0].tick_params(axis='both', labelsize=20)
axes[0,0].legend(fontsize=18)

axes[0,1].set(xlim=[0,25], ylim=[0,2])
axes[0,1].set_title('10Hz SSVEP', fontsize=24)
axes[0,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,1].set_ylabel('Power/Frequency', fontsize=20)
axes[0,1].plot(freq_ssvep[:,3], psd_ssvep[:,3], label='100ms', color='red', linewidth=2)
axes[0,1].plot(freq_ssvep[:,4], psd_ssvep[:,4], label='300ms', color='blue', linewidth=2)
axes[0,1].plot(freq_ssvep[:,5], psd_ssvep[:,5], label='500ms', color='green', linewidth=2)
axes[0,1].tick_params(axis='both', labelsize=20)
axes[0,1].legend(fontsize=18)

axes[0,2].set(xlim=[0,25], ylim=[0,2])
axes[0,2].set_title('15Hz SSVEP', fontsize=24)
axes[0,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,2].set_ylabel('Power/Frequency', fontsize=20)
axes[0,2].plot(freq_ssvep[:,6], psd_ssvep[:,6], label='100ms', color='red', linewidth=2)
axes[0,2].plot(freq_ssvep[:,7], psd_ssvep[:,7], label='300ms', color='blue', linewidth=2)
axes[0,2].plot(freq_ssvep[:,8], psd_ssvep[:,8], label='500ms', color='green', linewidth=2)
axes[0,2].tick_params(axis='both', labelsize=20)
axes[0,2].legend(fontsize=18)

axes[1,0].set(xlim=[0,25], ylim=[0,0.25])
axes[1,0].set_title('ERP (8Hz Stimulation)', fontsize=22)
axes[1,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,0].set_ylabel('Power/Frequency', fontsize=20)
axes[1,0].plot(freq_erp[:,0], psd_erp[:,0], label='100ms', color='red', linewidth=2)
axes[1,0].plot(freq_erp[:,1], psd_erp[:,1], label='300ms', color='blue', linewidth=2)
axes[1,0].plot(freq_erp[:,2], psd_erp[:,2], label='500ms', color='green', linewidth=2)
axes[1,0].tick_params(axis='both', labelsize=20)
axes[1,0].legend(fontsize=18)

axes[1,1].set(xlim=[0,25], ylim=[0,0.25])
axes[1,1].set_title('ERP (10Hz Stimulation)', fontsize=22)
axes[1,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,1].set_ylabel('Power/Frequency', fontsize=20)
axes[1,1].plot(freq_erp[:,3], psd_erp[:,3], label='100ms', color='red', linewidth=2)
axes[1,1].plot(freq_erp[:,4], psd_erp[:,4], label='300ms', color='blue', linewidth=2)
axes[1,1].plot(freq_erp[:,5], psd_erp[:,5], label='500ms', color='green', linewidth=2)
axes[1,1].tick_params(axis='both', labelsize=20)
axes[1,1].legend(fontsize=18)

axes[1,2].set(xlim=[0,25], ylim=[0,0.25])
axes[1,2].set_title('ERP (15Hz Stimulation)', fontsize=22)
axes[1,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,2].set_ylabel('Power/Frequency', fontsize=20)
axes[1,2].plot(freq_erp[:,6], psd_erp[:,6], label='100ms', color='red', linewidth=2)
axes[1,2].plot(freq_erp[:,7], psd_erp[:,7], label='300ms', color='blue', linewidth=2)
axes[1,2].plot(freq_erp[:,8], psd_erp[:,8], label='500ms', color='green', linewidth=2)
axes[1,2].tick_params(axis='both', labelsize=20)
axes[1,2].legend(fontsize=18)

# adjust position and paddings
fig.subplots_adjust(left=0.06, bottom=0.08, right=0.99, top=0.955,
                    hspace=0.4, wspace=0.3)
plt.savefig(r'E:/ssvep&erp.png', dpi=600)
plt.show()

#%% Antiphase comparision
fig = plt.figure(figsize=(18,15))
x=[i for i in range(-200, 1001, 1)]

gs = GridSpec(3, 2, figure=fig)

ax1 = fig.add_subplot(gs[:2, :])
ax1.set_title('Raw Data:15Hz&500ms', fontsize=30)
ax1.set_xlabel('Time/ms',fontsize=26)
ax1.set_ylabel('Amplitude/μV',fontsize=26)
ax1.plot(x, averaged_data[20,:], label='0 initial phase',
         color='red', linewidth=2)
ax1.plot(x, averaged_data[26,:], label='π initial phase',
         color='blue', linewidth=2)
ax1.vlines(8.3, -6., 6., colors='black', linestyles='dashed',
           label='start point', linewidth=2)
ax1.tick_params(axis='both', labelsize=20)
ax1.legend(fontsize=20)

ax2 = fig.add_subplot(gs[2, 0])
ax2.set_title('SSVEP data', fontsize=30)
ax2.set_xlabel('Time/ms',fontsize=26)
ax2.set_ylabel('Amplitude/μV',fontsize=26)
ax2.plot(x, ssvep_data[8,:], linewidth=2)
ax2.vlines(8.3, -5., 5., colors='black', linestyles='dashed',
           label='start point', linewidth=2)
ax2.tick_params(axis='both', labelsize=20)

ax3 = fig.add_subplot(gs[2, 1])
ax3.set_title('ERP data', fontsize=30)
ax3.set_xlabel('Time/ms',fontsize=26)
ax3.set_ylabel('Amplitude/μV',fontsize=26)
ax3.plot(x, erp_data[8,:], color='green', linewidth=2)
ax3.vlines(8.3, -5., 5., colors='black', linestyles='dashed',
           label='start point', linewidth=2)
ax3.tick_params(axis='both', labelsize=20)
ax3.legend(fontsize=20)

#format_axes(fig)
fig.subplots_adjust(left=0.08, bottom=0.095, right=0.960, top=0.945,
                    hspace=0.4, wspace=0.2)
plt.savefig(r'E:/15Hz&100ms.png', dpi=600)
plt.show()
#%% ssvep's tfr plotting(using array method)
# initialization
freqs = np.arange(1., 30, 0.05)
#vmin, vmax = -3., 3.  # Define our color limits.
n_cycles = freqs/2.

#%% 8Hz data preparation
epochs1 = Epochs(raw, events=events, event_id=1, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs1.pick_channels(pick_chans)
epochs_data1=1e4 * epochs1.get_data()
power1 = tfr_array_morlet(epochs_data1, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs1

epochs2 = Epochs(raw, events=events, event_id=10, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs2.pick_channels(pick_chans)
epochs_data2=1e4 * epochs2.get_data()
power2 = tfr_array_morlet(epochs_data2, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs2

epochs3 = Epochs(raw, events=events, event_id=19, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs3.pick_channels(pick_chans)
epochs_data3=1e4 * epochs3.get_data()
power3 = tfr_array_morlet(epochs_data3, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs3

epochs4 = Epochs(raw, events=events, event_id=4, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs4.pick_channels(pick_chans)
epochs_data4=1e4 * epochs4.get_data()
power4 = tfr_array_morlet(epochs_data4, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs4

epochs5 = Epochs(raw, events=events, event_id=13, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs5.pick_channels(pick_chans)
epochs_data5=1e4 * epochs5.get_data()
power5 = tfr_array_morlet(epochs_data5, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs5

epochs6 = Epochs(raw, events=events, event_id=22, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs6.pick_channels(pick_chans)
epochs_data6=1e4 * epochs6.get_data()
power6 = tfr_array_morlet(epochs_data6, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs6

epochs7 = Epochs(raw, events=events, event_id=7, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs7.pick_channels(pick_chans)
epochs_data7=1e4 * epochs7.get_data()
power7 = tfr_array_morlet(epochs_data7, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs7

epochs8 = Epochs(raw, events=events, event_id=16, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs8.pick_channels(pick_chans)
epochs_data8=1e4 * epochs8.get_data()
power8 = tfr_array_morlet(epochs_data8, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs8

epochs9 = Epochs(raw, events=events, event_id=25, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs9.pick_channels(pick_chans)
epochs_data9=1e4 * epochs9.get_data()
power9 = tfr_array_morlet(epochs_data9, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
Times = epochs9.times
del epochs9

#%% 10Hz preparation
epochs1 = Epochs(raw, events=events, event_id=2, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs1.pick_channels(pick_chans)
epochs_data1=1e4 * epochs1.get_data()
power1 = tfr_array_morlet(epochs_data1, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs1

epochs2 = Epochs(raw, events=events, event_id=11, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs2.pick_channels(pick_chans)
epochs_data2=1e4 * epochs2.get_data()
power2 = tfr_array_morlet(epochs_data2, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs2

epochs3 = Epochs(raw, events=events, event_id=20, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs3.pick_channels(pick_chans)
epochs_data3=1e4 * epochs3.get_data()
power3 = tfr_array_morlet(epochs_data3, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs3

epochs4 = Epochs(raw, events=events, event_id=5, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs4.pick_channels(pick_chans)
epochs_data4=1e4 * epochs4.get_data()
power4 = tfr_array_morlet(epochs_data4, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs4

epochs5 = Epochs(raw, events=events, event_id=14, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs5.pick_channels(pick_chans)
epochs_data5=1e4 * epochs5.get_data()
power5 = tfr_array_morlet(epochs_data5, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs5

epochs6 = Epochs(raw, events=events, event_id=23, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs6.pick_channels(pick_chans)
epochs_data6=1e4 * epochs6.get_data()
power6 = tfr_array_morlet(epochs_data6, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs6

epochs7 = Epochs(raw, events=events, event_id=8, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs7.pick_channels(pick_chans)
epochs_data7=1e4 * epochs7.get_data()
power7 = tfr_array_morlet(epochs_data7, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs7

epochs8 = Epochs(raw, events=events, event_id=17, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs8.pick_channels(pick_chans)
epochs_data8=1e4 * epochs8.get_data()
power8 = tfr_array_morlet(epochs_data8, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
del epochs8

epochs9 = Epochs(raw, events=events, event_id=26, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs9.pick_channels(pick_chans)
epochs_data9=1e4 * epochs9.get_data()
power9 = tfr_array_morlet(epochs_data9, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='itc')
Times = epochs9.times
del epochs9
#%% 15Hz preparation
epochs1 = Epochs(raw, events=events, event_id=3, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs1.pick_channels(pick_chans)
epochs_data1=1e4 * epochs1.get_data()
power1 = tfr_array_morlet(epochs_data1, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs1

epochs2 = Epochs(raw, events=events, event_id=12, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs2.pick_channels(pick_chans)
epochs_data2=1e4 * epochs2.get_data()
power2 = tfr_array_morlet(epochs_data2, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs2

epochs3 = Epochs(raw, events=events, event_id=21, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs3.pick_channels(pick_chans)
epochs_data3=1e4 * epochs3.get_data()
power3 = tfr_array_morlet(epochs_data3, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs3

epochs4 = Epochs(raw, events=events, event_id=6, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs4.pick_channels(pick_chans)
epochs_data4=1e4 * epochs4.get_data()
power4 = tfr_array_morlet(epochs_data4, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs4

epochs5 = Epochs(raw, events=events, event_id=15, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs5.pick_channels(pick_chans)
epochs_data5=1e4 * epochs5.get_data()
power5 = tfr_array_morlet(epochs_data5, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs5

epochs6 = Epochs(raw, events=events, event_id=24, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs6.pick_channels(pick_chans)
epochs_data6=1e4 * epochs6.get_data()
power6 = tfr_array_morlet(epochs_data6, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs6

epochs7 = Epochs(raw, events=events, event_id=9, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs7.pick_channels(pick_chans)
epochs_data7=1e4 * epochs7.get_data()
power7 = tfr_array_morlet(epochs_data7, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs7

epochs8 = Epochs(raw, events=events, event_id=18, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs8.pick_channels(pick_chans)
epochs_data8=1e4 * epochs8.get_data()
power8 = tfr_array_morlet(epochs_data8, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
del epochs8

epochs9 = Epochs(raw, events=events, event_id=27, tmin=tmin, tmax=tmax,
                baseline=baseline, preload=True)
epochs9.pick_channels(pick_chans)
epochs_data9=1e4 * epochs9.get_data()
power9 = tfr_array_morlet(epochs_data9, sfreq=1000, freqs=freqs, n_cycles=n_cycles,
                         output='avg_power')
Times = epochs9.times
del epochs9
#%% colorbar normalization
Vmax_1 = max(np.max(power_f8d6), np.max(power_f8d18), np.max(power_f8d30))
Vmax_2 = max(np.max(power_f10d6), np.max(power_f10d18), np.max(power_f10d30))
Vmax_3 = max(np.max(power_f15d6), np.max(power_f15d18), np.max(power_f15d30))

#%% SSVEP&ERP data prepare (15Hz)
ssvep_f15d6 = (epochs_data1 - epochs_data7)/2
ssvep_f15d18 = (epochs_data2 - epochs_data8)/2
ssvep_f15d30 = (epochs_data3 - epochs_data9)/2
#%% 10Hz
ssvep_f10d6 = (epochs_data1 - epochs_data7)/2
ssvep_f10d18 = (epochs_data2 - epochs_data8)/2
ssvep_f10d30 = (epochs_data3 - epochs_data9)/2
#%% 8Hz
ssvep_f8d6 = (epochs_data1 - epochs_data7)/2
ssvep_f8d18 = (epochs_data2 - epochs_data8)/2
ssvep_f8d30 = (epochs_data3 - epochs_data9)/2
#%% plot tfr
power_f15d6 = tfr_array_morlet(ssvep_f15d6, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f15d18 = tfr_array_morlet(ssvep_f15d18, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f15d30 = tfr_array_morlet(ssvep_f15d30, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f10d6 = tfr_array_morlet(ssvep_f10d6, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f10d18 = tfr_array_morlet(ssvep_f10d18, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f10d30 = tfr_array_morlet(ssvep_f10d30, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f8d6 = tfr_array_morlet(ssvep_f8d6, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f8d18 = tfr_array_morlet(ssvep_f8d18, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
power_f8d30 = tfr_array_morlet(ssvep_f8d30, sfreq=1000, freqs=freqs,
                               n_cycles=n_cycles, output='itc')
#%%
fig, axes = plt.subplots(3, 3, figsize=(18,15))

mesh = axes[0,0].pcolormesh(Times * 1000, freqs, power_f8d6[1], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_1)
axes[0,0].set_title('8Hz & 100ms', fontsize=24)
axes[0,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,0].set_ylabel('ITC/Frequency', fontsize=20)
axes[0,0].tick_params(axis='both', labelsize=20)
axes[0,0].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[0,0])

mesh = axes[0,1].pcolormesh(Times * 1000, freqs, power_f8d18[1], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_1)
axes[0,1].set_title('8Hz & 300ms', fontsize=24)
axes[0,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,1].set_ylabel('ITC/Frequency', fontsize=20)
axes[0,1].tick_params(axis='both', labelsize=20)
axes[0,1].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[0,1])

mesh = axes[0,2].pcolormesh(Times * 1000, freqs, power_f8d30[1], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_1)
axes[0,2].set_title('8Hz & 500ms', fontsize=24)
axes[0,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[0,2].set_ylabel('ITC/Frequency', fontsize=20)
axes[0,2].tick_params(axis='both', labelsize=20)
axes[0,2].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[0,2])

mesh = axes[1,0].pcolormesh(Times * 1000, freqs, power_f10d6[0], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_2)
axes[1,0].set_title('10Hz & 100ms', fontsize=24)
axes[1,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,0].set_ylabel('ITC/Frequency', fontsize=20)
axes[1,0].tick_params(axis='both', labelsize=20)
axes[1,0].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[1,0])

mesh = axes[1,1].pcolormesh(Times * 1000, freqs, power_f10d18[0], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_2)
axes[1,1].set_title('10Hz & 300ms', fontsize=24)
axes[1,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,1].set_ylabel('ITC/Frequency', fontsize=20)
axes[1,1].tick_params(axis='both', labelsize=20)
axes[1,1].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[1,1])

mesh = axes[1,2].pcolormesh(Times * 1000, freqs, power_f10d30[0], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_2)
axes[1,2].set_title('10Hz & 500ms', fontsize=24)
axes[1,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[1,2].set_ylabel('ITC/Frequency', fontsize=20)
axes[1,2].tick_params(axis='both', labelsize=20)
axes[1,2].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[1,2])

mesh = axes[2,0].pcolormesh(Times * 1000, freqs, power_f15d6[0], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_3)
axes[2,0].set_title('15Hz & 100ms', fontsize=24)
axes[2,0].set_xlabel('Frequency/Hz', fontsize=20)
axes[2,0].set_ylabel('ITC/Frequency', fontsize=20)
axes[2,0].tick_params(axis='both', labelsize=20)
axes[2,0].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[2,0])

mesh = axes[2,1].pcolormesh(Times * 1000, freqs, power_f15d18[0], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_3)
axes[2,1].set_title('15Hz & 300ms', fontsize=24)
axes[2,1].set_xlabel('Frequency/Hz', fontsize=20)
axes[2,1].set_ylabel('ITC/Frequency', fontsize=20)
axes[2,1].tick_params(axis='both', labelsize=20)
axes[2,1].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[2,1])

mesh = axes[2,2].pcolormesh(Times * 1000, freqs, power_f15d30[0], cmap='RdBu_r',
                    vmin=0, vmax=Vmax_3)
axes[2,2].set_title('15Hz & 500ms', fontsize=24)
axes[2,2].set_xlabel('Frequency/Hz', fontsize=20)
axes[2,2].set_ylabel('ITC/Frequency', fontsize=20)
axes[2,2].tick_params(axis='both', labelsize=20)
axes[2,2].set(ylim=freqs[[0, -1]], xlabel='Times/ms')
fig.colorbar(mesh, ax=axes[2,2])


# adjust position and paddings
fig.subplots_adjust(left=0.05, bottom=0.06, right=1.0, top=0.96,
                    hspace=0.5, wspace=0.2)

plt.savefig(r'E:/SSVEP ITC.png', dpi=600)
plt.show()

#del power1, power2, power3, power4, power5, power6, power7, power8, power9

#%% ERP comparison
fig, axes = plt.subplots(3, 1, figsize=(15,21))
x=[i for i in range(-200, 1001, 1)]

axes[0].set(xlim=[-250, 1050], ylim=[-5., 5.])
axes[0].set_xlabel('Time/ms', fontsize=26)
axes[0].set_ylabel('Amplitude/μV', fontsize=26)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_title('100ms Stimulation', fontsize=30)
axes[0].plot(x, erp_data[0,:], color='red', label='8Hz', linewidth=2)
axes[0].plot(x, erp_data[3,:], color='blue', label='10Hz', linewidth=2)
axes[0].plot(x, erp_data[6,:], color='green', label='15Hz', linewidth=2)
axes[0].vlines(8.3, -4., 4., colors='black', linestyles='dashed',
    label='start point', linewidth=2)
axes[0].legend(loc='1', fontsize=20)

axes[1].set(xlim=[-250, 1050], ylim=[-5., 5.])
axes[1].set_xlabel('Time/ms', fontsize=26)
axes[1].set_ylabel('Amplitude/μV', fontsize=26)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_title('300ms Stimulation', fontsize=30)
axes[1].plot(x, erp_data[1,:], color='red', label='8Hz', linewidth=2)
axes[1].plot(x, erp_data[4,:], color='blue', label='10Hz', linewidth=2)
axes[1].plot(x, erp_data[7,:], color='green', label='15Hz', linewidth=2)
axes[1].vlines(8.3, -4., 4., colors='black', linestyles='dashed',
    label='start point', linewidth=2)
axes[1].legend(loc='1', fontsize=20)

axes[2].set(xlim=[-250, 1050], ylim=[-5., 5.])
axes[2].set_xlabel('Time/ms', fontsize=26)
axes[2].set_ylabel('Amplitude/μV', fontsize=26)
axes[2].tick_params(axis='both', labelsize=20)
axes[2].set_title('500ms Stimulation', fontsize=30)
axes[2].plot(x, erp_data[2,:], color='red', label='8Hz', linewidth=2)
axes[2].plot(x, erp_data[5,:], color='blue', label='10Hz', linewidth=2)
axes[2].plot(x, erp_data[8,:], color='green', label='15Hz', linewidth=2)
axes[2].vlines(8.3, -4., 4., colors='black', linestyles='dashed',
    label='start point', linewidth=2)
axes[2].legend(loc='1', fontsize=20)

fig.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, hspace=0.3)
plt.savefig(r'E:/ERP Comparison.png', dpi=600)

plt.show()

#%% try to use matlab to plot psd
# save np.array into .mat
average_path = r'E:\Matlab_ex\averaged1_data'
ssvep_path = r'E:\Matlab_ex\ssvep1_data'
erp_path = r'E:\Matlab_ex\erp1_data'
io.savemat(average_path, {'averaged1_data':averaged_data})
io.savemat(ssvep_path, {'ssvep1_data':ssvep_data})
io.savemat(erp_path, {'erp1_data':erp_data})



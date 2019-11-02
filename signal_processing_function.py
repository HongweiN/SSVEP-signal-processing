# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:41:33 2019

This module contains some useful functions to help processing eeg data
Continuously updating...

2019-10-30:
    1. Multi-linear regression estimation
    2. SNR computing based on superimposed average method
    3. Baseline correction based on zero mean
    
2019-10-31:
    4. Time-frequency transform analysis using Morlet wavelet
    
2019-11-3:
    5. Welch power spectrum analysis

@author: Brynhildr
"""

import numpy as np
import math
import mne
from mne.time_frequency import tfr_array_morlet, psd_array_welch

#%%
def mlr_estimate(X, C, I):
    '''
    use input data array to estimate one-channel data
    :param X: 4D input array (n_events, n_trials, n_chans, n_times)
    :param C: 3D coefficient array (n_chans, n_events, n_trials)
    :param I: 2D intercept array (n_events, n_trials)
    :param target: 3D signal data array (n_events, n_trials, n_times)
    estimate equation: y = a + b1*x1 + b2*x2 + ... + bn*xn
    use mat() to transform array to matrix to apply linear computation
    use .A to transform matrix to array to apply normal functions
    '''
    target = np.zeros((X.shape[0], X.shape[1], X.shape[3]))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):  # i for events, j for trials
            target[i,j,:] = (np.mat(C[:,i,j].T)*np.mat(X[i,j,:,:])).A
            target[i,j,:] += I[i,j]
            
    return target

#%%
def snr_sa(X):
    '''
    in MLR condition
    use superimposed average method to compute SNR
    :param X: input 3D signal array (n_events, n_trials, n_times)
    assume yi refers to an observation from a single experiment
    assume E(y) refers to the mean of 50 trials' signal: E(y) = (y1+y2+...+yN)/N
    and the random noise are considered to be decreased during the average process
    So the SNR of pure SSVEP is computed like: SNR = sum((E(y))^2)/E(y^2)
    '''
    e_X = np.mean(X, axis=1)
    p_op_X = np.zeros((X.shape[0]))
    p_on_X = np.zeros((X.shape[0], X.shape[1]))
    
    for i in range(X.shape[0]):
        p_op_X[i] = np.sum(e_X[i,:]**2)
        
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p_on_X[i,j] = (np.sum(X[i,j,:]**2))/X.shape[1]
    
    snr = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            snr[i,j] = 10 * math.log((p_op_X[i]/p_on_X[i,j]), 10)
            
    return snr

#%%
def zero_mean(X):
    '''
    :param: input signal array, 4D (n_events, n_trials, n_chans, n_times)
        or 3D(n_events, n_trials, n_times)
    zero mean a signal sequence
    '''
    if X.ndim == 4:
        Y = np.zeros((X.shape[0],X.shape[1],X.shape[2],X.shape[3]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    Y[i,j,k,:]=X[i,j,k,:]-np.mean(X[i,j,k,:])
                
    if X.ndim == 3:
        Y = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i,j,:]=X[i,j,:]-np.mean(X[i,j,:])            
    return Y

#%%
def tfr_analysis(X, sfreq, freqs, n_cycles, mode):
    '''
    basic library is mne
    use Morlet wavelet to do time-frequency transform
    default choice: preload=True
    :param X: input data array (n_events, n_trials, n_chans, n_times)
    :param sfreq: sampling frequency
    :param freqs: list, define the frequencies used in time-frequency transform
    :param n_cycles: number of cycles in the Morlet wavelet; 
                    fixed number or one per frequency
    :param mode: complex, power, phase, avg_power, itc, avg_power_itc
        (1)complex: single trial complex (n_events, n_trials, n_chans, n_freqs, n_times)
        (2)power: single trial power (n_events, n_trials, n_chans, n_freqs, n_times)
        (3)phase: single trial phase (n_events, n_trials, n_chans, n_freqs, n_times)
        (4)avg_power: average of single trial power (n_events, n_chans, n_freqs, n_times)
        (5)itc: inter-trial coherence (n_events, n_chans, n_freqs, n_times)
        (6)avg_power_itc: average of singel trial power and inter-trial coherence
            across trials :avg_power+i*itc (n_events, n_chans, n_freqs, n_times)
    expand data array in channel's dimension to fit tfr_array_morlet if necessary
    '''
    if X.ndim < 4:
        data = np.zeros((X.shape[0], X.shape[1], 2, X.shape[2]))
        for i in range(2):
            data[:,:,i,:] = X
    else:
        data = X
        
    if mode == 'complex':
        C = np.zeros((data.shape[0], data.shape[1], data.shape[2], freqs.shape[0], data.shape[3]))
        for i in range(data.shape[0]):
            C[i,:,:,:,:] = tfr_array_morlet(data[i,:,:,:], sfreq=sfreq,
                         freqs=freqs, n_cycles=n_cycles, output='complex')
        return C
    
    elif mode == 'power':
        PO = np.zeros((data.shape[0], data.shape[1], data.shape[2], freqs.shape[0], data.shape[3]))
        for i in range(data.shape[0]):
            PO[i,:,:,:,:] = tfr_array_morlet(data[i,:,:,:], sfreq=sfreq,
                          freqs=freqs, n_cycles=n_cycles, output='power')
        return PO
    
    elif mode == 'phase':
        PH = np.zeros((data.shape[0], data.shape[1], data.shape[2], freqs.shape[0], data.shape[3]))
        for i in range(data.shape[0]):
            PH[i,:,:,:,:] = tfr_array_morlet(data[i,:,:,:], sfreq=sfreq,
                          freqs=freqs, n_cycles=n_cycles, output='phase')
        return PH
    
    elif mode == 'avg_power':
        AP = np.zeros((data.shape[0], data.shape[2], freqs.shape[0], data.shape[3]))
        for i in range(data.shape[0]):
            AP[i,:,:,:] = tfr_array_morlet(data[i,:,:,:], sfreq=sfreq,
                          freqs=freqs, n_cycles=n_cycles, output='avg_power')
        return AP
    
    elif mode == 'itc':
        ITC = np.zeros((data.shape[0], data.shape[2], freqs.shape[0], data.shape[3]))
        for i in range(data.shape[0]):
            ITC[i,:,:,:] = tfr_array_morlet(data[i,:,:,:], sfreq=sfreq,
                          freqs=freqs, n_cycles=n_cycles, output='itc')
        return ITC
    
    elif mode == 'avg_power_itc':
        API = np.zeros((data.shape[0], data.shape[2], freqs.shape[0], data.shape[3]))
        for i in range(data.shape[0]):
            API[i,:,:,:] = tfr_array_morlet(data[i,:,:,:], sfreq=sfreq,
                          freqs=freqs, n_cycles=n_cycles, output='avg_power_itc')
        return API
    
#%%
def welch_p(X):
    '''
    use welch method to estimate signal power spectrum
    
    '''
    
    
    return 
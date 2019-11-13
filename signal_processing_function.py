# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:41:33 2019

This module contains some useful functions to help processing EEG data
Continuously updating...

1. Spatial filter: 
    (1) Multi-linear regression
    (2) Inverse array
2. Signal estimate & extraction
3. SNR computarion: 
    (1) Superimposed average
    (2) SSVEP frequency domain
    (3) SSVEP time domain
4. Baseline correction: (zero mean method)
5. Time-frequency transform:
    (1) Morlet wavelet
    (2) Short-time Fourier transform (STFT)
6. Power spectral density: (Welch method)
7. Precise FFT transform
8. Cosine similarity:
    (1) Normal
    (2) Tanimoto (Generalized Jaccard)
9. Residual analysis
 

@author: Brynhildr
"""
#%% import modules
import numpy as np
import math
import mne
from mne.time_frequency import tfr_array_morlet, psd_array_welch, stft
from sklearn.linear_model import LinearRegression
from scipy import signal

#%% spatial filter
def inv_spa(data, target):
    '''
    Use inverse array to create spatial filter A
    Y=AX, Y (X.T) ((XX.T)^(-1))= A
    Use mat() to transform array to matrix to apply linear computation
    Use .A to transform matrix to array to apply normal functions
    :param data: input model (n_stims, n_epochs, n_chans, n_times)
    :param target: output model (n_stims, n_epochs, n_times)
    :param coef: bool, if True, return filter coefficients A
    '''
    a = np.zeros((data.shape[0], data.shape[1], data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # transform array to matrix
            y = np.mat(target[i,j,:])     # (1,T)
            x = np.mat(data[i,j,:,:])     # (N,T)
            xt = np.mat(data[i,j,:,:].T)  # (T,N)

            xxt = x * xt                  # matrix multiplication (N,N)
            ixxt = xxt.I                  # inverse matrix (N,N)
            a[i,j,:] = y * xt * ixxt      # A =Y*(X.T)*((XX.T)^(-1)): (1,N)

    a = a.A                               # transform matrix to array

    return a

def mlr_analysis(data, target):
    '''
    Do multi-linear regression repeatedly
    Model = LinearRegression().fit(X,Y): X(n_chans, n_times) & Y(n_chans, n_times)
    :param data: input model: 4D array (n_events, n_epochs, n_chans, n_times)
    :param target: output model: output model: 3D array (n_events, n_epochs, n_times)
    Return R^2(after correction), coefficient, intercept
    R^2 here is a corrected version: 
        new R^2 = 1-(RSS/TSS)*((n-1)/(n-k-1)) = 1-(1-R^2)*((n-1)/(n-k-1))
        (n: time points;   k: input channels)
        which will be a little less than original R^2 but more accurate
    Expected to add in future: F-score, T-score, collinear diagnosis, ANOVA,
        correlation, coefficient correlation, RSS analysis, 95% confidence interval
    '''
    # R^2: R2 (n_events, n_epochs)
    R2 = np.zeros((data.shape[0], data.shape[1]))

    # R^2 adjustment coefficient
    correc_co = (data.shape[3]-1) / (data.shape[3]-data.shape[2]-1)
    
    # regression coefficient: RC (n_events, n_epochs, n_chans)
    RC = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    
    # regression intercept: RI (n_events, n_epochs): 
    RI = np.zeros((data.shape[0], data.shape[1]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):  # i for events, j for epochs
            # linear regression, remember to transform the array
            L = LinearRegression().fit(data[i,j,:,:].T, target[i,j,:].T)
            # the coefficient of derterminated R^2 of the prediction
            R2[i,j] = 1 - (1 - L.score(data[i,j,:,:].T, target[i,j,:])) * correc_co
            # the intercept of the model
            RI[i,j] = L.intercept_
            # the regression coefficient of the model
            RC[i,j,:] = L.coef_

    return RC, RI, R2


#%% signal extraction
def sig_extract(coef, data, target, intercept=None):
    '''
    :param coef: from spatial filter or regression (n_events, n_epochs, n_chans)
    :param data: input data (n_events, n_epochs, n_chans, n_times)
    :param target: original data (one-channel)
    :param intercept: regression intercept (n_events, n_epochs)
    estimate & extract: one-channel data
    '''
    estimate = np.zeros((data.shape[0], data.shape[1], data.shape[3]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            estimate[i,j,:] = (np.mat(coef[i,j,:]) * np.mat(X2[i,j,:,:])).A
            if intercept == None:
                continue
            else:
                estimate[i,j,:] += intercept[i,j]
    
    extract =  target - estimate

    return estimate, extract


#%% SNR computation
def snr_sa(X):
    '''
    Use superimposed average method to compute SNR in MLR condition
    :param X: input 3D signal array (n_events, n_epochs, n_times)
    Assume yi refers to an observation from a single experiment
    Assume E(y) refers to the mean of signals: E(y) = (y1+y2+...+yN)/N
        and the random noise are decreased during the average process
    So the SNR of pure SSVEP is computed like: SNR = sum((E(y))^2)/E(y^2)
    '''
    e_X = np.mean(X, axis=2)
    p_s_X = np.zeros((X.shape[0]))
    p_sn_X = np.zeros((X.shape[0], X.shape[1]))
    
    for i in range(X.shape[0]):
        p_s_X[i] = (np.mean(e_X[i,:]))**2
        
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p_sn_X[i,j] = e_X[i,j]**2
    
    snr = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            snr[i,j] = 10 * math.log((p_s_X[i]/(p_sn_X[i,j]-p_s_X[i])), 10)
            
    return snr

def snr_freq(X):
    '''
    Compute SNR of SSVEP in frequency domain
    Define SNR of each frequency point as the sum of power of its
        surrounding points in 1Hz's range, which is :
            SNR(freq) = Power(freq)/sum(Power((freq-1):(freq+1)))
    For 1st, 2nd, last and penultimate point, make estimations according to
        edge rules, which consider not-included points are equal to the edge values
    :param X: input spectrum data array (n_events, n_epochs, n_times)
    '''
    snr = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

    for i in range(X.shape[0]):         # i for events
        for j in range(X.shape[1]):     # j for epochs
            for k in range(X.shape[2]):
                if k==0:
                    snr[i,j,k] = 20 * math.log10(X[i,j,k]/(np.sum(X[i,j,k:k+3])+2*X[i,j,k]))
                if k==1:
                    snr[i,j,k] = 20 * math.log10(X[i,j,k]/(np.sum(X[i,j,k-1:k+3])+X[i,j,k-1]))
                if k==(X.shape[2]-1):
                    snr[i,j,k] = 20 * math.log10(X[i,j,k]/(np.sum(X[i,j,k-2:])+2*X[i,j,k]))
                if k==(X.shape[2]-2):
                    snr[i,j,k] = 20 * math.log10(X[i,j,k]/(np.sum(X[i,j,k-2:])+X[i,j,k+1]))
                else:
                    snr[i,j,k] = 20 * math.log10(X[i,j,k]/np.sum(X[i,j,k-2:k+3]))
    
    return snr

def snr_time(X):
    '''
    '''


#%% baseline correction
def zero_mean(X):
    '''
    :param X: input signal array, 4D (n_events, n_epochs, n_chans, n_times)
        or 3D(n_events, n_epochs, n_times)
    Zero mean a signal sequence
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


#%% time-frequency transform
def tfr_morlet(X, sfreq, freqs, n_cycles, mode):
    '''
    Basic library is mne
    Use Morlet wavelet to do time-frequency transform
    Default choice: preload=True
    :param X: input data array (n_events, n_epochs, n_chans, n_times)
    :param sfreq: sampling frequency
    :param freqs: list, define the frequencies used in time-frequency transform
    :param n_cycles: number of cycles in the Morlet wavelet; 
                    fixed number or one per frequency
    :param mode: complex, power, phase, avg_power, itc, avg_power_itc
        (1)complex: single trial complex (n_events, n_epochs, n_chans, n_freqs, n_times)
        (2)power: single trial power (n_events, n_epochs, n_chans, n_freqs, n_times)
        (3)phase: single trial phase (n_events, n_epochs, n_chans, n_freqs, n_times)
        (4)avg_power: average of single trial power (n_events, n_chans, n_freqs, n_times)
        (5)itc: inter-trial coherence (n_events, n_chans, n_freqs, n_times)
        (6)avg_power_itc: average of singel trial power and inter-trial coherence
            across trials :avg_power+i*itc (n_events, n_chans, n_freqs, n_times)
    Expand data array in channel's dimension to fit tfr_array_morlet if necessary
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
    
def tfr_stft(X, sfreq, freqs, mode):
    '''
    Basic library is mne
    Use STFT(short-time fourier transform) to do time-frequency transform
    '''


#%% power spectral density
def welch_p(X, sfreq, fmin, fmax, n_fft, n_overlap, n_per_seg):
    '''
    Use welch method to estimate signal power spectral density
    Basic function is mne.psd_array_welch
    :param X: input data array (n_events, n_epochs, n_times)
    :param sfreq: the sampling frequency
    :param fmin, fmax: the lower(upper) frequency of interest
    :param n_fft: the length of FFT used, must be >= n_per_seg
    :param n_overlap: the number of points of overlap between segments
    :param n_per_seg: length of each welch segment, usually = n_fft
    :param psds: power spectral density array (n_events, n_epochs, n_freqs)
    :param freqs: frequencies used in psd analysis
    '''
    #num_freqs = (np.arange(n_fft//2+1, dtype=float)*(sfreq/n_fft)).shape[0]
    psds = np.zeros((X.shape[0], X.shape[1], 103))
    freqs = np.zeros((X.shape[0], X.shape[1], 103))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            psds[i,j,:], freqs[i,j,:] = psd_array_welch(X[i,j,:], sfreq=sfreq, fmin=fmin,
                    fmax=fmax, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg)

    return psds, freqs


#%% frequency spectrum
def precise_fft(X, ):
    '''
    Compute & plot frequency spectrum of signal
    :param X: input signal data array (n_events, n_epochs, n_times)
    :param n_fft: fft points
    6
    '''


#%% cosine similarity
def cos_sim(origin, estimate):
    '''
    Compute one kind of cosine similarity of two signal sequence
    sim(x,y) = xy / (||x|| * ||y||), x&y are multivariate vectors
    :param origin: origin data array (n_events, n_epochs, n_times)
    :param estimate: estimate data using MLR (n_events, n_epochs, n_times)
    '''

    return SIM

def tanimoto_sim(origin, estimate):
    '''
    Compute another cosine similarity of two signal sequence
    also called generalized Jaccard coefficient
    sim(x,y) = xy / (x*x + y*y - x*y), x&y are multivariate vectors
    '''

    return SIM


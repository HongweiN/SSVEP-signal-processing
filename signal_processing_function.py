# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:41:33 2019

This module contains some useful functions to help processing EEG data
Continuously updating...

2019-10-30:
    1. Multi-linear regression estimation
    2. SNR computing based on superimposed average method
    3. Baseline correction based on zero mean method
    
2019-10-31:
    4. Time-frequency transform analysis using Morlet wavelet
    
2019-11-3:
    5. Welch power spectral density analysis
    6. Multi-linear regression computation (combined with No.1)

2019-11-10:
    7. Time-frequency transform using STFT (combined with No.4)
    8. Precise FFT transform
    9. SNR of SSVEP signal (in frequency domain) (combined with No.2)
    10. Find spectrum peaks (combined with No.9)

2019-11-:
    11. Cosine similarity of two signal sequence
    12. Residual analysis of estimate signal and original signal
    13. 

@author: Brynhildr
"""
#%% import modules
import numpy as np
import math
import mne
from mne.time_frequency import tfr_array_morlet, psd_array_welch, stft
from sklearn.linear_model import LinearRegression
from numpy import signal

#%% MLR analysis (computation & estimation)
def mlr_analysis(X1, Y, X2, regression=True):
    '''
    Do multi-linear regression repeatedly and estimate one-channel signal
    Model = LinearRegression().fit(X,Y): X(n_chans, n_times) & Y(n_chans, n_times)
    :param X1: input model: 4D array (n_events, n_epochs, n_chans, n_times)
    :param Y: output model: output model: 3D array (n_events, n_epochs, n_times)
    :param X2: input signal: 4D array (n_events, n_epochs, n_chans, n_times)
    :param target: 3D signal data array (n_events, n_epochs, n_times)

    Part I:
        Return R^2(after correction), coefficient, intercept
        R^2 here is a corrected version: 
            new R^2 = 1-(RSS/TSS)*((n-1)/(n-k-1)) = 1-(1-R^2)*((n-1)/(n-k-1))
            (n: time points;   k: input channels' number)
            which is a little less than original R^2 
            and more accurate for MLR than LR(linear regression)
    
    Part II:
        Estimate equation: y = a + b1*x1 + b2*x2 + ... + bn*xn
        Use mat() to transform array to matrix to apply linear computation
        Use .A to transform matrix to array to apply normal functions
    
    Expected to add in future: F-score, T-score, collinear diagnosis, ANOVA,
        correlation, coefficient correlation, RSS analysis, 95% confidence interval
    '''
    # R^2 array: R2 (n_events, n_epochs)
    R2 = np.zeros((X1.shape[0], X1.shape[1]))

    # R^2 adjustment coefficient
    correc_co = (X1.shape[3] - 1) / (X1.shape[3] - X1.shape[2] - 1)
    
    # regression coefficient array: RC (n_events, n_epochs, rc array)
    RC = np.zeros((X1.shape[2], X1.shape[0], X1.shape[1]))
    
    # regression intercept array: RI (n_events, n_epochs): 
    RI = np.zeros((X1.shape[0], X1.shape[1]))
    
    # estimate data array
    Target = np.zeros((X2.shape[0], X2.shape[1], X2.shape[3]))

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):  # i for events, j for epochs
            # linear regression, remember to transform the array
            L = LinearRegression().fit(X1[i,j,:,:].T, Y[i,j,:].T)
            # the coefficient of derterminated R^2 of the prediction
            R2[i,j] = 1 - (1 - L.score(X1[i,j,:,:].T, Y[i,j,:])) * correc_co
            # the intercept of the model
            RI[i,j] = L.intercept_
            # the regression coefficient of the model
            RC[:,i,j] = L.coef_

    for i in range(X2.shape[0]):
        for j in range(X2.shape[1]):
            Target[i,j,:] = (np.mat(RC[:,i,j].T) * np.mat(X2[i,j,:,:])).A
            Target[i,j,:] += RI[i,j]
    
    if regression == True:
        return R2, Target
    elif regression == False:
        return Target

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

#%% baseline correction
def zero_mean(X):
    '''
    :param: input signal array, 4D (n_events, n_epochs, n_chans, n_times)
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
    
def tfr_stft(X, sfreq, freqs):
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

#%% Frequency spectrum
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
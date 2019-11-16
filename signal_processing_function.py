# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:41:33 2019

This module contains some useful functions to help processing EEG data
Continuously updating...

1. Spatial filter: 
    (1) Multi-linear regression
    (2) Inverse array
    (3) Goodness of fit
2. Variance computation
3. Signal estimate & extraction
4. SNR computarion: 
    (1) Superimposed average
    (2) SSVEP frequency domain
    (3) SSVEP time domain
5. Baseline correction: (zero mean method)
6. Time-frequency transform:
    (1) Morlet wavelet
    (2) Short-time Fourier transform (STFT)
7. Power spectral density: (Welch method)
8. Precise FFT transform
9. Cosine similarity:
    (1) Normal
    (2) Tanimoto (Generalized Jaccard)
10. Canonical correlation analysis (CCA)
11. Inter-channel correlation
    (1) Spearman method
    (2) Pearson method
    (3) Binarizaiton of compare correlation array
12. Residual analysis
 

@author: Brynhildr
"""
#%% import modules
import numpy as np
import math
import mne
from mne.time_frequency import tfr_array_morlet, psd_array_welch, stft
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA
from scipy import signal
import pandas as pd

#%% spatial filter
def inv_spa(data, target):
    '''
    Use inverse array to create spatial filter A
    Y=AX, Y (X.T) ((XX.T)^(-1))= A
    Use mat() to transform array to matrix to apply linear computation
    Use .A to transform matrix to array to apply normal functions
    :param data: input model (n_events, n_epochs, n_chans, n_times)
    :param target: output model (n_events, n_epochs, n_times)
    '''
    # spatial filter coefficients (n_events, n_epochs, n_chans)
    A = np.zeros((data.shape[0], data.shape[1], data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # transform array to matrix
            y = np.mat(target[i,j,:])     # (1,T)
            x = np.mat(data[i,j,:,:])     # (N,T)
            xt = np.mat(data[i,j,:,:].T)  # (T,N)

            xxt = x * xt                  # matrix multiplication (N,N)
            ixxt = xxt.I                  # inverse matrix (N,N)
            A[i,j,:] = y * xt * ixxt      # A =Y*(X.T)*((XX.T)^(-1)): (1,N)
            
    return A


def fit_goodness(X, Y, chans):
    '''
    Compute goodness of fit in non-linear-regression occasion
    :param X: original signal (n_events, n_epochs, n_times)
    :param Y: estimate signal (n_events, n_epochs, n_times)
    :param chans: number of regression data' channels
    '''
    # R^2: R2 (n_events, n_epochs)
    R2 = np.zeros((X.shape[0], X.shape[1]))
    correc_co = (X.shape[2] - 1) / (X.shape[2] - chans -1)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            RSS = np.sum((X[i,j,:] - Y[i,j,:])**2)
            TSS = np.sum((X[i,j,:] - np.mean(X[i,j,:]))**2)
            R2[i,j] = 1 - (RSS/TSS) * correc_co
    
    return R2
    
    
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
def sig_extract(coef, data, target, intercept):
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
            estimate[i,j,:] = (np.mat(coef[i,j,:]) * np.mat(data[i,j,:,:])).A
            estimate[i,j,:] += intercept[i,j]
    
    extract =  target - estimate

    return estimate, extract


#%% variance computation
def var_estimation(X):
    '''
    Use superimposed method to compute multi-dimension data's variance
    :param X: input data (n_events, n_epochs, n_times)
    '''
    var = np.zeors((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):                 # i for n_events
        ex = np.mat(np.mean(X[i,:,:], axis=0))  # (1, n_times)
        temp = np.mat(np.ones((1, X.shape[1]))) # (1, n_epochs)
        minus = (temp.T * ex).A                 # (n_epochs, n_times)
        var[i,:] = np.mean((X[i,:,:] - minus)**2, axis=0)

    return var # (n_events, n_times)


#%% SNR computation
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


def snr_time(X, mode):
    '''
    Two method for SNR computation
        (1) Compute SNR and return time sequency
        (2) Use superimposed average method:
    :param X: input data (one-channel) (n_events, n_epochs, n_times) 
    :param mode: choose method
    '''
    if mode == 'time':
        snr = np.zeros((X.shape[0], X.shape[2]))    # (n_events, n_times)
        # basic object's size: (n_epochs, n_times)
        for i in range(X.shape[0]):                 # i for n_events
            ex = np.mat(np.mean(X[i,:,:], axis=0))  # (1, n_times)
            temp = np.mat(np.ones((1, X.shape[1]))) # (1, n_epochs)
            minus = (temp.T * ex).A                 # (n_epochs, n_times)
            ex = (ex.A) ** 2
            var = np.mean((X[i,:,:] - minus)**2, axis=0)
            snr[i,:] = ex/var
    
    if mode == 'average':
        snr = np.zeros((X.shape[0]))

        for i in range(X.shape[0]):
            ex = np.mat(np.mean(X[i,:,:], axis=0))
            temp = np.mat(np.ones((1, X.shape[1])))
            minus = (temp.T * ex).A
            var = np.mean((X[i,:,:] - minus)**2, axis=0)
            snr[i] = 20 * np.log10(np.sum(ex)/np.sum(var))

    return snr


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
    psds = np.zeros((X.shape[0], X.shape[1], 251))
    freqs = np.zeros((X.shape[0], X.shape[1], 251))
    
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
def cos_sim(X, Y, mode):
    '''
    Compute two kind of cosine similarity of two signal sequence:
        (1) sim(x,y) = xy / (||x|| * ||y||), x&y are multivariate vectors
        (2) tanimoto(x,t) = xy / (x*x + y*y - x*y)
    :param X: origin data array (n_events, n_epochs, n_times)
    :param Y: estimate data using MLR (n_events, n_epochs, n_times)
    :param mode: str, choose method
    '''
    sim = np.zeros((X.shape[0], X.shape[1]))
    # (n_events, n_epochs)
    if mode == 'normal':
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xy = np.sum(X[i,j,:] * Y[i,j,:])
                mx = (np.sum(X[i,j,:]**2)) ** 0.5
                my = (np.sum(Y[i,j,:]**2)) ** 0.5
                sim[i,j] = xy / (mx * my)
    
    if mode == 'tanimoto':
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                xy = np.sum(X[i,j,:] * Y[i,j,:])
                xx = np.sum(X[i,j,:]**2)
                yy = np.sum(Y[i,j,:]**2)
                sim[i,j] = xy / (xx + yy - xy)
                
    return sim


#%% Canonical correlation analysis (CCA)
def cca_coef(X, Y):
    '''
    Apply CCA method to compute inter_channel correlation
    :param X: data 1 (n_events, n_epochs, n_chans, n_times)
    :param Y: data 2 (actually equal to data 1)
    '''
    cca = CCA(n_components=1)


#%% inter-channel correlation coefficient
def corr_coef(X, mode):
    '''
    Contains Spearman & Pearson correlation coefficient
    :param X: input data (n_events, n_epochs, n_chans, n_times)
    :param mode: str, choose which method to use
    '''
    corr = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[2]))

    if mode == 'pearson':
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                corr[i,j,:,:] = np.corrcoef(X[i,j,:,:])
                
        corr = np.mean(np.mean(corr, axis=0), axis=0)

    if mode == 'spearman':
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                temp = pd.DataFrame(X[i,j,:,:].T)
                corr[i,j,:,:] = temp.corr('spearman')
                
        corr = np.mean(np.mean(corr, axis=0), axis=0)
    
    return corr


def bina_corr(X,Y, th):
    '''
    Compare two correlation array and do binarization
    :param X&Y: two input array (n_chans, n_chans)
    :param th: threshold
    '''
    compare = X-Y
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if compare[i,j] > th:
                compare[i,j] = 1
            else:
                compare[i,j] = 0
    
    return compare
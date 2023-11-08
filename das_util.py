import os
import h5py
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib

from obspy import UTCDateTime
from datetime import datetime
from datetime import timedelta
from functools import partial
from scipy.signal import butter
from scipy.signal import detrend
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal import filtfilt, butter
from scipy.signal.windows import tukey
from multiprocessing import Pool
from matplotlib import pyplot as plt


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low < 0:
        Wn = high
        btype = "lowpass"
    elif high < 0:
        Wn = low
        btype = "highpass"
    else:
        Wn = [low, high]
        btype = "bandpass"

    b, a = butter(order, Wn, btype=btype)

    return b, a


def taper_filter(arr, fmin, fmax, samp_DAS):
    b_DAS, a_DAS = butter_bandpass(fmin, fmax, samp_DAS)
    window_time = tukey(arr.shape[1], 0.1)
    arr_wind = arr * window_time
    arr_wind_filt = filtfilt(b_DAS, a_DAS, arr_wind, axis=-1)
    return arr_wind_filt


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def fk_filter_2cones(vsp, w1=0, w2=0, cone1=False, cone2=False):
    n1, n2 = vsp.shape
    nf = next_power_of_2(n1)
    nk = next_power_of_2(n2)

    nf2 = int(nf / 2)
    nk2 = int(nk / 2)

    fk2d = np.fft.fft2(vsp, s=(nf, nk))
    fk2d = np.fft.fftshift(fk2d, axes=(-2, -1))

    nw1 = int(np.ceil(w1 * nk))
    nw2 = int(np.ceil(w2 * nf))

    mask1 = np.ones((nf, nk), dtype=np.float64)
    mask2 = np.ones((nf, nk), dtype=np.float64)

    if cone1:
        for j in np.arange(nk2 - nw1, nk2 + 1):
            th1 = int((j - nk2 + nw1) * nf2 / nw1)

            mask1[:th1, j] = 0
            mask1[nf - th1:, j] = 0
            mask1[:th1, nk - j] = 0
            mask1[nf - th1:, nk - j] = 0

    if cone2:
        for j in np.arange(0, nk2):
            th2 = int(nf2 - (nw2 / nk2) * (nk2 - j))
            mask2[th2:nf - th2 + 1, j] = 0
            if j != 0:
                mask2[th2:nf - th2 + 1, nk - j] = 0

    mask = mask2 * mask1

    filtered_2d = fk2d * mask
    tmp = np.fft.ifftshift(filtered_2d)
    output = np.fft.ifft2(tmp, s=(nk, nf), axes=(-1, -2))

    return output[:n1, :n2], filtered_2d, fk2d


def read_decimate(file_path, dsamp_factor=20, start_ch=0, end_ch=100):
    with h5py.File(file_path,'r') as f:      
        minute_data = f['Acquisition']['Raw[0]']['RawData'][:, start_ch:end_ch].T
    downsample_data = decimate(minute_data, q=dsamp_factor, ftype='fir', zero_phase=True)   
    return downsample_data


# %% extract time from file name (by Ethan Williams)
def get_tstamp(fname):
    datestr = fname.split('_')[1].split('-')
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    timestr = fname.split('_')[2].split('.')
    H = int(timestr[0])
    M = int(timestr[1])
    S = int(timestr[2])
    return UTCDateTime('%04d-%02d-%02dT%02d:%02d:%02d' % (y,m,d,H,M,S))


def get_tstamp_dts(time_label):
    datestr = time_label.split(' ')[0].split('/')
    timestr = time_label.split(' ')[1]
    y = int(datestr[0])
    m = int(datestr[1])
    d = int(datestr[2])
    return UTCDateTime('%04d-%02d-%02dT%08s' % (y,m,d,timestr))


# %% calculate the NFFT for spectrogram (by Dominik Gräff)
def calc_NFFT(trace, sample_rate, power_of_2=True):
    '''calculate meaningful number of fourier samples for spectrogram'''
    NFFT = int(trace.shape[-1]/1000) # results in ~1000 bins
    if power_of_2:
        NFFT = int(2**(math.floor(math.log(NFFT, 2)))) # power of 2 < than original value
    print(r'NFFT={} samples, equivalent to {} seconds'.format(NFFT, NFFT/sample_rate))
    return NFFT, NFFT/sample_rate


# %% use pandas to find the best time-ticks
def x_tick_locs(stime, etime):
    '''calculate where to put x-ticks'''
    fmtfreq_list = {'years':['1YS'],
                    'months':['6MS','1MS'],
                    'days':['10d','5d','1d'],
                    'hours':['12h','6h','3h','1h'],
                    'minutes':['30min','15min','10min','5min','1min'],
                    'seconds':['30s','15s','10s','5s','1s']}

    for key in fmtfreq_list.keys():
        for value in fmtfreq_list[key]:
            daterange = pd.date_range(stime, etime+pd.Timedelta('1d'), 
                          freq=value, normalize=True)
            daterange = [t for t in daterange if t>=stime if t<=etime]
            if len(daterange)<6:
                continue
            else:
                return key, daterange


def x_labels_fmt(key, same_superior):
    '''x-ticks and axis formatting'''
    # if no change of superior unit
    if same_superior:
        fmtlabels_list = {'years':('%Y', ('', '', '[Year]')),
                          'months':('%b', ('of', '%Y', '[Month]')),
                          'days':('%-d %b', ('of', '%Y' '[Day Month]')),
                          'hours':('%H:%M', ('of', '%-d %b %Y', '[Hour:Minute]')),
                          'minutes':('%H:%M', ('of', '%-d %b %Y', '[Hour:Minute]')),
                          'seconds':('%H:%M:%S', ('of', '%-d %b %Y', '[Hour:Minute:Second]'))}
    # if superior unit changes
    if not same_superior:
        fmtlabels_list = {'years':('%Y', ('', '', '[Year]')),
                          'months':('%b %Y', ('', '', '[Month Year]')),
                          'days':('%-d %b', ('in', '%Y', '[Day Month]')),
                          'hours':('%-d %b %H:%M', ('in', '%Y', '[Day Month  Hour:Minute]')),
                          'minutes':('%H:%M', ('of', '%-d %b %Y', '[Hour:Minute]')),
                          'seconds':('%H:%M:%S', ('of', '%-d %b %Y', '[Hour:Minute:Second]'))}
    return fmtlabels_list[key]


def t_array(t):
    '''returns np.array([year,month,day,hour,minute,second])'''
    t_arr = [t.year,t.month,t.day,t.hour,t.minute,t.second]
    return t_arr


def translate_daterange_intervals(daterange, t_keys):
    '''find daterange spacing time unit'''
    t_arrs = [t_array(t) for t in daterange]
    for i, key in enumerate(t_keys):
        if not t_arrs[0][:i+1]==t_arrs[1][:i+1]:
            key = t_keys[i]
            return key


def nice_x_axis(stats, bins, t_int=False):
    stime = stats.starttime.datetime
    etime = stats.endtime.datetime
    t_bins = [(stats.starttime+t).datetime for t in bins]

    # units into which humans subdivide time
    t_keys = ['years','months','days','hours','minutes','seconds']

    # if fixed x-tick interval is set and if valid
    if t_int:
        try:
            daterange = pd.date_range(stime, etime+pd.Timedelta('1d'), freq=t_int, normalize=True)
            daterange = [t for t in daterange if t>=stime if t<=etime]
            key = translate_daterange_intervals(daterange, t_keys)
        except ValueError:
            print('Set "t_int" keyword smaller than time series length')
    else: # automatically choose x-tick interval
        key, daterange = x_tick_locs(stime, etime)

    # ===== x-tick location =====
    x_tickloc = [UTCDateTime(t).timestamp for t in daterange]

    # ===== x-tick format =====
    key_idx = t_keys.index(key) # get index of key in list
    # check if x-tick intervals are over a superior unit (eg. minute x-ticks, but crossing a full hour)
    same_superior = t_array(t_bins[0])[:key_idx] == t_array(t_bins[-1])[:key_idx]
    x_labels_fmt(key, same_superior)
    x_tickformat = x_labels_fmt(key, same_superior)[0]
    x_ticks_str = [t.strftime('{}'.format(x_tickformat)) for t in daterange]

    # set x-axis label
    x_label_time = stime.strftime('{}'.format(x_labels_fmt(key, same_superior)[1][1]))
    x_label = 'Time (UTC)  {} {}  {}'.format(x_labels_fmt(key, same_superior)[1][0],
                                            x_label_time,
                                            x_labels_fmt(key, same_superior)[1][2])
    return x_tickloc, x_ticks_str, x_label


# plot the spectrogram
def plot_spectro(Pxx, freqs, bins, stats, 
                 t_int=False,
                 cmap="viridis",
                 vmax=None,
                 vmin=None, # matplotlib color map
                 ylim=None, 
                 yscale='linear',
                 **kwargs):
    '''plot spectrogram from pre-calculated values in calc_spec'''    
    fig = plt.figure(figsize=(6.4*2,4.8))
    ax = fig.add_axes([0.125, 0.125, 0.76, 0.6])      # main spectrogram
    ax_cbar = fig.add_axes([0.895, 0.125, 0.02, 0.6]) # colorbar
    
    # apply the ylim here - directly reduce the data
    if ylim:
        idx_low = (np.abs(freqs - ylim[0])).argmin()
        idx_high = (np.abs(freqs - ylim[1])).argmin()
        Pxx_cut = Pxx[idx_low:idx_high,:]
        freqs_cut = freqs[idx_low:idx_high]
    else:
        Pxx_cut = Pxx
        freqs_cut = freqs
    
    im = ax.imshow(10*np.log10(Pxx_cut), 
                   aspect='auto', 
                   origin='lower', 
                   cmap=cmap,
                   extent=[stats.starttime.timestamp, stats.endtime.timestamp, freqs_cut[0], freqs_cut[-1]],
                   vmax=vmax, 
                   vmin=vmin)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.ax.locator_params(nbins=5)
    cbar.set_label('Power Spectral Density [dB]', fontsize=12) #colorbar label
    
    # set the x-ticks
    x_tickloc, x_ticks_str, x_label = nice_x_axis(stats, bins, t_int=t_int) # functions to plot nice x-axis
    ax.set_xticks(x_tickloc)
    ax.set_xticklabels(x_ticks_str)
    ax.set_xlabel(x_label, fontsize=12) 
    
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    return fig

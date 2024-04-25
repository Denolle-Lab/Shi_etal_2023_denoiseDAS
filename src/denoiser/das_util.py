import os
import h5py
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib

from obspy import UTCDateTime
from obspy.taup import TauPyModel
from distaz import DistAz
from joblib import Parallel, delayed
from datetime import datetime
from datetime import timedelta
from functools import partial
from scipy.signal import butter
from scipy.signal import detrend
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal import filtfilt, butter
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
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


############################ Functions for offshore Alaska

### Functions to calculate P and S time on arrays
### Travel time for given channels
def array_tpts(i, stla, stlo, evla, evlo, evdp):
    
    distdeg = DistAz(stla[i], stlo[i], evla, evlo).getDelta()
    tp = TauPyModel(model="iasp91").get_travel_times(source_depth_in_km=evdp, 
                                                     distance_in_degree=distdeg, 
                                                     phase_list=['p', 'P'])
    ts = TauPyModel(model="iasp91").get_travel_times(source_depth_in_km=evdp, 
                                                     distance_in_degree=distdeg, 
                                                     phase_list=['s', 'S'])
    return [tp[0].time, ts[0].time]


### Add travel time for AK DAS arrays to dataframe
def akdas_tpts(cat, eid, kkfls, terra, correct_terra):
    ### Event info
    evt = cat[eid]
    mag = evt.magnitudes[0].mag
    lon = evt.origins[0].longitude
    lat = evt.origins[0].latitude
    dep = evt.origins[0].depth/1000 + 2.0  ## too shallow depth+ long distance = bugs
    ort = evt.origins[0].time
    
    mag0,lon0,lat0,dep0,ort0=round(mag,1),round(lon,2),round(lat,2),round(dep,0),ort.strftime('%Y-%m-%d')

    ### array info
    t_kkfls=np.array(Parallel(n_jobs=100)(delayed(array_tpts)(ch,kkfls['lat'].values,kkfls['lon'].values,lat,lon,dep) 
                                           for ch in range(len(kkfls))))
    t_terra=np.array(Parallel(n_jobs=100)(delayed(array_tpts)(ch,terra['lat'].values,terra['lon'].values,lat,lon,dep) 
                                           for ch in range(len(terra))))

    kkfls['tp'] = t_kkfls[:, 0]
    kkfls['ts'] = t_kkfls[:, 1]
    terra['tp'] = t_terra[:, 0]+correct_terra
    terra['ts'] = t_terra[:, 1]+correct_terra
    
    return kkfls, terra, mag0,lon0,lat0,dep0,ort0


### Functions to denoise large-N DAS array
def process_3d_array(arr, len1=1500, len2=1500):
    """convert to numpy array"""
    arr = np.array(arr)
    
    """Ensure the array has at least len1 rows and len2 columns"""
    slices, rows, cols = arr.shape
    arr = arr[:, :min(rows, len1), :min(cols, len2)]
    
    """Pad zeros if it has fewer than len1 rows or len2 columns"""
    if rows < len1 or cols < len2:
        padding_rows = max(len1 - rows, 0)
        padding_cols = max(len2 - cols, 0)
        arr = np.pad(arr, ((0, 0), (0, padding_rows), (0, padding_cols)), 'constant')
    
    return arr


def Denoise_largeDAS(data, model_func, devc, repeat=4, norm_batch=False):
    """ This function do the following (it does NOT filter data):
    1) split into multiple 1500-channel segments
    2) call Denoise function for each segments
    3) merge all segments
    
    data: 2D -- [channel, time]
    output: 2D, but padded 0 to have multiple of 1500 channels
    
    This code was primarily designed for the Alaska DAS, but applicable to other networks
    """ 
    data = np.array(data)
    nchan = data.shape[0]
    ntime = data.shape[1]
    
    if (nchan // 1500) == 0:
        n_seg = nchan // 1500
    else:
        n_seg = nchan // 1500 + 1
        
    full_len = int(n_seg * 1500)
    
    pad_data = process_3d_array(data[np.newaxis,:,:], len1=full_len)
    data3d = pad_data.reshape((-1, 1500, 1500))
    
    oneDenoise, mulDenoise = Denoise(data3d, model_func, devc, repeat=repeat, norm_batch=norm_batch)
    
    oneDenoise2d = oneDenoise.reshape((full_len, 1500))[:nchan, :ntime]
    mulDenoise2d = mulDenoise.reshape((full_len, 1500))[:nchan, :ntime]
    
    return oneDenoise2d, mulDenoise2d
    

def Denoise(data, model_func, devc, repeat=4, norm_batch=False):
    """ This function do the following (it does NOT initialize model):

    1) normalized the data
    2) ensure the data format, precision and size
    3) denoise and scale back the output amplitude
    """ 
    
    """ convert to torch tensors """
    if norm_batch:
        scale = np.std(data[-1]) + 1e-7  ### Avoid potentially bad beginning sub-images
    else:
        scale = np.std(data, axis=(1,2), keepdims=True) + 1e-7
        
    data_norm = data / scale  ## standard scaling
    arr = process_3d_array(data_norm.astype(np.float32))
    X = torch.from_numpy(arr).to(devc)
    
    """ denoise - deploy """
    with torch.no_grad():
        oneDenoise = model_func(X)
        mulDenoise = oneDenoise
        
        for i in range(repeat-1):
            mulDenoise = model_func(mulDenoise)

    """ convert back to numpy """
    oneDenoise = oneDenoise.to('cpu').numpy() * scale
    mulDenoise = mulDenoise.to('cpu').numpy() * scale
    
    return oneDenoise[:, :len(data[0]), :len(data[0][0])], mulDenoise[:, :len(data[0]), :len(data[0][0])]

def vizRawDenoise(in_data, oneDenoise, mulDenoise, sample_rate=25, dchan=10, index=[0,1], model="MAE"):
    """
    in_data, oneDenoise, mulDenoise: 3D -- [event, channel, time]
    index: list, subset of the events
    model: string, descriptions about the model
    """
    len1, len2 = oneDenoise[0].shape[0], oneDenoise[0].shape[1]
    x, y = np.arange(len2)/sample_rate, np.arange(0-len1/2, len1/2)*dchan/1000
    rawdata = process_3d_array(in_data, len1=len1, len2=len2)
    
    matplotlib.rcParams['font.size'] = 20

    for j in index:
        bound = np.percentile(np.fabs(in_data[j]), 80)
        cmp = matplotlib.colormaps['RdBu']
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        img=ax[0].pcolormesh(x, y, rawdata[j], shading='auto', vmin=-bound, vmax=bound, cmap=cmap)
        ax[1].pcolormesh(x, y, oneDenoise[j], shading='auto',  vmin=-bound, vmax=bound, cmap=cmap)
        ax[2].pcolormesh(x, y, mulDenoise[j], shading='auto', vmin=-bound, vmax=bound, cmap=cmap)

        ax[0].set_title("Raw data #"+str(j))
        ax[1].set_title(model+" 1-time denoised")
        ax[2].set_title(model+" multi-time denoised")
        ax[0].set_ylabel('Distance (km)')
        ax[0].set_xlabel('Time (s)')
        ax[1].set_xlabel('Time (s)')
        ax[2].set_xlabel('Time (s)')

        plt.colorbar(img, ax=ax[2], aspect=50)


### Functions to pick large DAS arrays
def process_p(ista,paras_semblance,batch_pred,istart,sfs):
    
        crap = ensemble_semblance(batch_pred[:, ista, :], paras_semblance)
        imax = np.argmax(crap[istart:])
            
        return float((imax)/sfs)+istart/sfs, crap[istart+imax]
    

def apply_elep(DAS_data, list_models, fs, paras_semblance, device):
    
    """"
    This function takes a array of stream, a list of ML models and 
    apply these models to the data, predict phase picks, and
    return an array of picks .
    DAS_data: NDArray of DAS data: [channel,time stamp - 6000]
    """
    
    twin = 6000  ## needed by EQTransformer
    nsta = DAS_data.shape[0]
    bigS = np.zeros(shape=(DAS_data.shape[0], 3, DAS_data.shape[1]))
    for i in range(nsta):   ### same data copied to three components
        bigS[i,0,:] = DAS_data[i,:]
        bigS[i,1,:] = DAS_data[i,:]
        bigS[i,2,:] = DAS_data[i,:]

    # allocating memory for the ensemble predictions
    batch_pred_P = np.zeros(shape=(len(list_models),nsta,twin)) 
    batch_pred_S = np.zeros(shape=(len(list_models),nsta,twin))
        
    ######### Broadband workflow ################
    crap2 = bigS.copy()
    crap2 -= np.mean(crap2, axis=-1, keepdims= True) # demean data
    # original use std norm
    data_std = crap2 / (np.std(crap2) + 1e-7)
    # could use max data
    mmax = np.max(np.abs(crap2), axis=-1, keepdims=True)
    data_max = np.divide(crap2 , mmax,out=np.zeros_like(crap2), where=mmax!=0)
    del bigS
    
    # data to tensor
    data_tt = torch.from_numpy(data_max).to(device, dtype=torch.float32)
    
    for ii, imodel in enumerate(list_models):
        imodel.eval()
        with torch.no_grad():
            batch_pred_P[ii, :, :] = imodel(data_tt)[1].cpu().numpy()[:, :]
            batch_pred_S[ii, :, :] = imodel(data_tt)[2].cpu().numpy()[:, :]
    
    smb_peak = np.zeros([nsta,2,2], dtype = np.float32)

    smb_peak[:,0,:] =np.array(Parallel(n_jobs=100)(delayed(process_p)(ista,paras_semblance,batch_pred_P,0,fs) 
                                                    for ista in range(nsta)))
    smb_peak[:,1,:] =np.array(Parallel(n_jobs=100)(delayed(process_p)(ista,paras_semblance,batch_pred_S,0,fs) 
                                                    for ista in range(nsta)))
    
    return smb_peak


### plotting codes to simplify massive event processing

### plot picked times
def subfig_img(image, pick, ind_p, ind_s, pred, array, colors=['blue', 'green']):
    x = np.arange(image.shape[1])/fs
    y = np.arange(0-image.shape[0]/2,image.shape[0]/2)*ch_itv*dchan/1000
    bound = np.percentile(np.fabs(image), 80)
    
    plt.pcolormesh(x,y,image, shading='auto',vmin=-bound,vmax=bound,cmap=matplotlib.colormaps['RdBu'])
    plt.plot(pred[:,1], array, color='red', linestyle='-', lw=5) 
    # plt.plot(pred[:,0], array, color='orange', linestyle='-', lw=5)   
    # plt.scatter(pick[ind_p,0,0],y[ind_p], s=15,marker='o',c=colors[0],alpha=0.3)
    plt.scatter(pick[ind_s,1,0],y[ind_s], s=15,marker='o',c=colors[1],alpha=0.3)
    plt.ylabel("Distance along cable (km)")

### plot picking likelihood
def subfig_histpick(pick, colors=['blue', 'green'], labels=['P', 'S']):
    plt.hist(pick[:,0,1],bins=20,color=colors[0],label=labels[0],range=(0,0.3))
    plt.hist(pick[:,1,1],bins=20,color=colors[1],label=labels[1],range=(0,0.3))
    plt.title("Picks count")
    plt.xlabel("Probability")
    plt.ylim(0,500)
    plt.xlim(0.05,0.3)

### plot waveforms
def subfig_goodtrace(image, pick, ind_p, ind_s, tax, win):
    snr_p=0
    snr_s=0
    
    for ch in ind_p:
        pt=int(pick[ch,0,0]*fs)
        snr_p+=np.std(image[ch, pt:pt+fs]) / (np.std(image[ch, pt-fs:pt])+1e-7)
    for ch in ind_s: 
        pt=int(pick[ch,1,0]*fs)
        snr_s+=np.std(image[ch, pt:pt+fs]) / (np.std(image[ch, pt-fs:pt])+1e-7)
        plt.plot(tax[win], image[ch, win])
            
    return snr_p/(len(ind_p) + 1e-7), snr_s/ (len(ind_s) + 1e-7)


def fit_series(s1, s2, prob, thr=0.05, vmin=0, vmax=60):
    offsets = s1-s2
    ind = np.where(np.logical_and(np.logical_and(np.logical_and(vmin<s1, s1<vmax), prob > thr), np.fabs(offsets) < 3.0))[0]
    
    if len(ind)>0:
        offsets = offsets[ind]
    else:
        offsets = 0
        
    mean_offset = np.mean(offsets)
    offsets = offsets-mean_offset
    
    return offsets, round(np.std(offsets),3), mean_offset, ind


### Functions for CC-based alignment
def shift2maxcc(wave1, wave2, maxshift=5):
    n1 = np.sum(np.square(wave1))
    n2 = np.sum(np.square(wave2))
    corr = sgn.correlate(wave1, wave2) / np.sqrt(n1 * n2)
    lags = sgn.correlation_lags(len(wave1), len(wave2))

    st_pt = len(wave2) - min(len(wave2), maxshift)
    en_pt = len(wave2) + min(len(wave1), maxshift)

    ind1 = np.argmax(corr[st_pt: en_pt]) + st_pt

    return lags[ind1], corr[ind1]


def shift_pad(wave, shift_pt=0):
    tmp_tr = np.zeros(wave.shape, dtype=np.float32)
    if shift_pt > 0:
        tmp_tr[shift_pt:] = wave[0:0 - shift_pt]
    elif shift_pt < 0:
        tmp_tr[0:shift_pt] = wave[0 - shift_pt:]
    else:
        tmp_tr[:] = wave[:]
    return tmp_tr


### Functions to shift each channel of the 2D data using shift2maxcc
def align_channels_twice(data, ref_ch=0, maxshift=5, cc_thres=0.5):

    nchan, ntime = data.shape

    aligned = np.zeros_like(data)
    shifts = np.zeros(nchan)
    cccs = np.zeros(nchan)

    ## First round of alignment with reference channel
    count = 0
    ref = np.zeros(ntime)
    for i in range(nchan):
        shift, ccc = shift2maxcc(data[ref_ch], data[i], maxshift=maxshift)
        if ccc > cc_thres:
            ref += shift_pad(data[i], shift_pt=int(shift))
            count += 1
    ref /= count

    ## second round of alignment with the stacked aligned data
    for i in range(nchan):
        shifts[i], cccs[i] = shift2maxcc(ref, data[i], maxshift=maxshift)
        aligned[i,:] = shift_pad(data[i], shift_pt=int(shifts[i]))

    return aligned, shifts, cccs



### functions to calculate PDF of multiple channels
def ppsd(data,fs,fmin,fmax):
    """
    data:  2D array, the statistics is calculated along axis=0
    fs: sampling rate
    fmin: minimum frequency for statistics
    fmax: maximum frequency for statictics
    """
    ns = data.shape[1]
    nx = data.shape[0]
    
    ### Demean, detrend
    data -= np.mean(data, axis=1, keepdims=True) 
#     data = sgn.detrend(data, axis=1) 
    
    freq, spec = sgn.periodogram(data, sample_rate, window='hamming', axis=-1)
    freq = np.tile(freq,(nx,1)).flatten()

    ### Generate PDF
    xbins = np.logspace(np.log10(fmin),np.log10(fmax),60)
    ybins = np.logspace(np.log10(np.nanmin(spec)),np.log10(np.nanmax(spec)),90)
    
    H,xe,ye = np.histogram2d(freq.flatten(), spec.flatten(), bins=(xbins,ybins))
    
    return H/np.nansum(H, axis=1, keepdims=True), (xe[1:] + xe[:-1])/2, (ye[1:] + ye[:-1])/2
    
def psd_stats(H,xm,ym):
    ym = np.log10(ym)
    mean = np.zeros(len(xm))
    variance = mean.copy()
    for ix in range(len(xm)):
        mean[ix] = np.average(ym,weights=H[ix,:])
        variance[ix] = np.average((ym-mean[ix])**2,weights=H[ix,:])
    
    return xm,10**mean,variance
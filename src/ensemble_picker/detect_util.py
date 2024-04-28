import sys
sys.path.append('../denoiser/')

import gc
import glob
import h5py
import numpy as np
import pandas as pd

import torch
import seisbench.models as sbm
from ELEP.elep.ensemble_coherence import ensemble_semblance
from ELEP.elep.trigger_func import picks_summary_simple

from datetime import datetime
from das_util import try_gpu
from joblib import Parallel, delayed
from tqdm import tqdm

import obspy
from obspy import UTCDateTime, read_events
from obspy.clients.fdsn.client import Client



### Functions to pick many windows
def apply_elep_smb(data, list_models, paras_semblance, device):
    """
    Input: data of 1 station, 3 components, many time windows
           several eqT models (already to device and eval mode)
    Output: semblance for P or S
    """

    ### Normalize data
    tmp = data - np.mean(data, axis=-1, keepdims= True)
    mmax = np.max(np.abs(tmp), axis=-1, keepdims=True)
    data_max = np.divide(tmp , mmax, out=np.zeros_like(tmp), where=(mmax!=0))
    data_tt = torch.from_numpy(data_max).to(device, dtype=torch.float32)
    
    ### ELEP workflow
    twin = 6000  ## constant for EQTransformer
    nwin = data.shape[0]
   
    # predictions from all models
    batch_pred_P = np.zeros((len(list_models),nwin,twin)) 
    batch_pred_S = np.zeros((len(list_models),nwin,twin))
    smb = np.zeros((nwin,2,twin), dtype = np.float32)
    
    for ii, imodel in enumerate(list_models):
        with torch.no_grad():
            batch_pred_P[ii, :, :] = imodel(data_tt)[1].cpu().numpy()[:, :]
            batch_pred_S[ii, :, :] = imodel(data_tt)[2].cpu().numpy()[:, :]
    
    # semblance of all predictions
    smb[:,0,:] =np.array(Parallel(n_jobs=100)(delayed(ensemble_semblance)(batch_pred_P[:, iwin, :], paras_semblance) 
                                                    for iwin in range(nwin)))
    smb[:,1,:] =np.array(Parallel(n_jobs=100)(delayed(ensemble_semblance)(batch_pred_S[:, iwin, :], paras_semblance) 
                                                    for iwin in range(nwin)))
    
    del batch_pred_P, batch_pred_S, data_tt, data_max, mmax, tmp
    gc.collect()
    torch.cuda.empty_cache()
    
    return smb


def detect_on_fly(network, station, t1, filepath, width, stride, list_models, devcc):
    '''
    The workflow: 
    1) download data for 1 station
    2) window the data with stride
    3) apply ELEP
    4) save picks
    '''
   ########################
   ### Download waveforms
    try:
        sdata = client.get_waveforms(network=network, 
                                     station=station,
                                     location="*", 
                                     channel="BH?", 
                                     starttime=t1,
                                     endtime=t1 + 86400)
        
    except obspy.clients.fdsn.header.FDSNNoDataException:
        print(f"--- No data for {network}.{station} on {t1} ---")
        return
    
    fs_all = [tr.stats.sampling_rate for tr in sdata]
    fs = np.round(fs_all[0])
    if len(np.unique(np.array(fs_all))) > 1:      
        print(f"--- Sampling rates are different for {network}.{station} on {t1} ---")
        sdata = sdata.resample(fs)
    
    sdata.merge(fill_value='interpolate')  # fill gaps
    sdata.filter(type='bandpass',freqmin=0.5,freqmax=12)
    btime = sdata[0].stats.starttime
    

    # align 3 components
    max_b = max([tr.stats.starttime for tr in sdata])
    min_e = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_b, endtime=min_e, nearest_sample=True)    
        
    ########################
    ### Window data
    arr_sdata = np.array(sdata)
    if len(arr_sdata.shape) == 1:
        arr_sdata = arr_sdata[np.newaxis, :] # add dimension
    if arr_sdata.shape[0] == 1:
        arr_sdata = np.repeat(arr_sdata, 3, axis=0)
    elif arr_sdata.shape[0] == 2:
        arr_sdata = np.vstack((arr_sdata, arr_sdata[1]))
    elif arr_sdata.shape[0] > 3:
        arr_sdata = arr_sdata[:3]

    nwin = (arr_sdata.shape[1] - width) // stride
    if nwin < 1: 
        print(f"--- Data too short for {network}.{station} on {t1} ---")
        return
    arr_sdata = arr_sdata[:, :int(nwin * stride + width)]
    
    win_idx = np.zeros(nwin, dtype=np.int32)
    windows = np.zeros((nwin, 3, width), dtype= np.float32)

    for iwin in range(nwin):
        idx = iwin * stride
        win_idx[iwin] = idx
        windows[iwin,:,:] = arr_sdata[:, idx:idx+width]
        
    ########################
    ### Apply ELEP
    paras_semblance = {'dt':1/fs, 
                       'semblance_order':2, 
                       'window_flag':True, 
                       'semblance_win':0.5, 
                       'weight_flag':'max'}
    
    smb = apply_elep_smb(windows, list_models, paras_semblance, devcc)
    smb_all = np.zeros_like(arr_sdata[0:2])

    for iwin in range(nwin):
        idx = iwin * stride
        smb_all[0, idx+stride:idx+width] = smb[iwin,0,stride:]
        smb_all[1, idx+stride:idx+width] = smb[iwin,1,stride:]

    p_picks = picks_summary_simple(smb_all[0], 0.10)
    s_picks = picks_summary_simple(smb_all[1], 0.05)

    ########################
    ### Save picks
    len_picks = len(p_picks + s_picks)
    df = pd.DataFrame({
        'event_id': [' '] * len_picks,
        'source_type': [' '] * len_picks,
        'station_network_code': [network] * len_picks,
        'station_channel_code': [' '] * len_picks,
        'station_code': [station] * len_picks,
        'station_location_code': [sdata[0].stats.location] * len_picks,
        'station_latitude_deg': [inventory[0][0].latitude] * len_picks,
        'station_longitude_deg': [inventory[0][0].longitude] * len_picks,
        'station_elevation_m': [inventory[0][0].elevation] * len_picks,
        'trace_name': [' '] * len_picks,
        'trace_sampling_rate_hz': [sdata[0].stats.sampling_rate] * len_picks,
        'trace_start_time': [sdata[0].stats.starttime] * len_picks,
        'trace_S_arrival_sample': [' '] * len_picks,
        'trace_P_arrival_sample': [' '] * len_picks,
        'trace_S_onset': [' '] * len_picks,
        'trace_P_onset': [' '] * len_picks,
        'trace_snr_db': [' '] * len_picks,
        'trace_s_arrival': [np.nan] * len(p_picks) + [str(btime + idx / fs) for idx in s_picks],
        'trace_p_arrival': [str(btime + idx / fs) for idx in p_picks] + [np.nan] * len(s_picks)
    })

    df.to_csv(filepath+'1month/' + network + '_' + station + '_' + t1.strftime('%Y%m%d') + '.csv')
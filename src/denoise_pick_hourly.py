import sys
sys.path.append('../src/denoiser/')
sys.path.append('../src/ensemble_picker/')

import gc
import glob
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import seisbench.models as sbm
from ELEP.elep.ensemble_coherence import ensemble_semblance
from ELEP.elep.trigger_func import picks_summary_simple

from datetime import datetime, timedelta
from das_util import *
from detect_util import *
from das_denoise_models import unet
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import filtfilt, butter
from scipy.interpolate import interp1d

import obspy
from obspy import UTCDateTime



############################################# Loop for hours
filepath = '/auto/cpu-disk1/qibin_folder/hourly_catalog/'
rawdata_dir = '/mnt/qnap/'
model_dir = '../models/checkpoint_noatt_LRdecays0.8_mask0.5_raw2raw_chmax4500.pt'
kkfls_dir = rawdata_dir + 'KKFL-S_FIberA_25Hz/'
terra_dir = rawdata_dir + 'TERRA_FiberA_25Hz/'
format_part = 'decimator2_%Y-%m-%d_%H.??.??_UTC.h5'
### Bandpass filter
b, a = butter(4, (0.5, 12), fs=25, btype='bandpass')

### Initialize the U-net
devc = try_gpu(i=0)
model_1 = unet(1, 16, 1024, factors=(5, 3, 2, 2), use_att=False)
model_1 = nn.DataParallel(model_1, device_ids=[0])
model_1.to(devc)
### Load the pretrained weights
model_1.load_state_dict(torch.load(model_dir))
model_1.eval() 

### ELEP models
pn_ethz_model = sbm.EQTransformer.from_pretrained("ethz")
pn_neic_model = sbm.EQTransformer.from_pretrained("neic")
pn_scedc_model = sbm.EQTransformer.from_pretrained("scedc")
pn_stead_model = sbm.EQTransformer.from_pretrained("stead")
pn_geofon_model = sbm.EQTransformer.from_pretrained("geofon")
pn_instance_model = sbm.EQTransformer.from_pretrained("instance")
pn_ethz_model.to(devc); pn_ethz_model.eval()
pn_neic_model.to(devc); pn_neic_model.eval()
pn_scedc_model.to(devc); pn_scedc_model.eval()
pn_stead_model.to(devc); pn_stead_model.eval()
pn_geofon_model.to(devc); pn_geofon_model.eval()
pn_instance_model.to(devc); pn_instance_model.eval()
list_models = [pn_ethz_model,
            pn_neic_model,
            pn_scedc_model,
            pn_stead_model,
            pn_geofon_model,
            pn_instance_model]
fs=100; ch_itv=500  # data are downsampled to pick

### Cable coordinates from Ethan Williams. We only use Ch. 500-5000
kkfls = pd.read_csv('cable_geometry/KKFLS_coords.xycz',header=None,names=['lon','lat','cha','dep'],delim_whitespace=True)
terra = pd.read_csv('cable_geometry/TERRA_coords.xycz',header=None,names=['lon','lat','cha','dep'],delim_whitespace=True)


for ihour in range(31*24):
# for ihour in tqdm(range(23,24)):
    ############################################# Read hour data
    t0 = UTCDateTime("2023-12-01") + ihour*3600
    print('---- Read'+t0.strftime('%Y%m%d_%H')+' ----')
    fname = UTCDateTime.strftime(t0, format=format_part)
    kkfls_files = sorted(glob.glob(kkfls_dir+fname))
    terra_files = sorted(glob.glob(terra_dir+fname))

    kkfls_data = np.zeros((len(kkfls_files), 4500, 1500), dtype=np.float32)
    terra_data = np.zeros((len(terra_files), 4500, 1500), dtype=np.float32)
    kkfls_btimes = np.zeros(len(kkfls_files), dtype=object)
    terra_btimes = np.zeros(len(terra_files), dtype=object)

    for i, kkfls_file in enumerate(kkfls_files):
        with h5py.File(kkfls_file, 'r') as f:
            time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 500:5000]
            kkfls_data[i, :time_data.shape[1], :time_data.shape[0]] = time_data.T
            kkfls_btimes[i] = datetime.utcfromtimestamp(f['Acquisition']['Raw[0]']['RawDataTime'][0]/1e6)
        del time_data
        gc.collect()

    for i, terra_file in enumerate(terra_files):
        with h5py.File(terra_file, 'r') as f:
            time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 500:5000]
            terra_data[i, :time_data.shape[1], :time_data.shape[0]] = time_data.T
            terra_btimes[i] = datetime.utcfromtimestamp(f['Acquisition']['Raw[0]']['RawDataTime'][0]/1e6)
        del time_data
        gc.collect()

    ### merge two arrays and filter
    rawdata = np.append(kkfls_data[:, ::-1, :], terra_data[:,:,:], axis=1)
    rawdata = np.nan_to_num(rawdata)
    filt = filtfilt(b, a, rawdata, axis=2)
    rawdata = filt / np.std(filt, axis=(1,2), keepdims=True)
    len_cat = len(rawdata)

    del filt, kkfls_data, terra_data, kkfls_files, terra_files, 
    gc.collect()

    ############################################# Denoise
    print('---- Denoising ----')
    mul_denoised = np.zeros_like(rawdata)
    for imin in np.arange(len_cat):
        _, mul_denoised[imin,:,:] = Denoise_largeDAS(rawdata[imin], model_1, devc, repeat=4, norm_batch=False)
    del rawdata
    gc.collect()
    torch.cuda.empty_cache()
    
    ### Interpolate
    interp_func = interp1d(np.linspace(0, 1, 1500), mul_denoised, axis=-1, kind='linear')
    interpolated_muldenoised = interp_func(np.linspace(0, 1, 6000))

    ############################################# Pick
    print('---- Picking ----')
    ### ELEP parameters
    paras_semblance = {'dt':0.01, 
                    'semblance_order':2, 
                    'window_flag':True, 
                    'semblance_win':0.5, 
                    'weight_flag':'max'}
    
    sel_ch = np.arange(int(ch_itv/2), interpolated_muldenoised.shape[1], ch_itv)
    nsta = len(sel_ch)
    mul_picks = np.zeros([len_cat, nsta, 2, 2], dtype = np.float32)
    for imin in np.arange(len_cat):
        mul_picks[imin,:,:,:] = apply_elep(interpolated_muldenoised[imin,sel_ch,:], 
                                           list_models, fs, paras_semblance, devc)
    del interpolated_muldenoised, interp_func, mul_denoised
    gc.collect()
    torch.cuda.empty_cache()
    
    ############################################# Save
    print('---- Saving ----')
    df_deno = pd.DataFrame(columns=[
            'event_id',
            'source_type',
            'station_network_code',
            'station_channel_code',
            'station_code',
            'station_location_code',
            'station_latitude_deg',
            'station_longitude_deg',
            'station_elevation_m',
            'trace_name',
            'trace_sampling_rate_hz',
            'trace_start_time',
            'trace_S_arrival_sample',
            'trace_P_arrival_sample',
            'trace_S_onset',
            'trace_P_onset',
            'trace_snr_db',
            'trace_s_arrival',
            'trace_p_arrival'])

    for ch in np.arange(nsta):
        if ch >= (nsta/2):  # terra
            ch1 = int(sel_ch[ch] - 4000) 
            longitude = terra.loc[ch1, 'lon']
            latitude = terra.loc[ch1, 'lat']
            elevation = terra.loc[ch1, 'dep']
            b_t = terra_btimes
        else:  # kkfls
            ch1 = int(5000 - sel_ch[ch])
            longitude = kkfls.loc[ch1, 'lon']
            latitude = kkfls.loc[ch1, 'lat']
            elevation = kkfls.loc[ch1, 'dep']
            b_t = kkfls_btimes

        p_den = [b_t[i] + timedelta(seconds=np.float64(mul_picks[i, ch, 0, 0])) if mul_picks[i, ch, 0, 1] > 0.10 else np.nan for i in range(len_cat)]
        s_den = [b_t[i] + timedelta(seconds=np.float64(mul_picks[i, ch, 1, 0])) if mul_picks[i, ch, 1, 1] > 0.05 else np.nan for i in range(len_cat)]
        
        df_deno = pd.concat([df_deno, pd.DataFrame(data={
            'event_id': [' '] * len_cat,
            'source_type': [' '] * len_cat,
            'station_network_code': ['CIDAS'] * len_cat,
            'station_channel_code': [' '] * len_cat,
            'station_code': ['das'+str(ch*100)] * len_cat,
            'station_location_code': [' '] * len_cat,
            'station_latitude_deg': [latitude] * len_cat,
            'station_longitude_deg': [longitude] * len_cat,
            'station_elevation_m': [elevation] * len_cat,
            'trace_name': [' '] * len_cat,
            'trace_sampling_rate_hz': [25] * len_cat,
            'trace_start_time': b_t,
            'trace_S_arrival_sample': [' '] * len_cat,
            'trace_P_arrival_sample': [' '] * len_cat,
            'trace_S_onset': [' '] * len_cat,
            'trace_P_onset': [' '] * len_cat,
            'trace_snr_db': [' '] * len_cat,
            'trace_s_arrival': s_den,
            'trace_p_arrival': p_den})], ignore_index=True)
        
    df_deno.to_csv(filepath+'1hour/' + 'CIDAS_' + t0.strftime('%Y%m%d_%H') + '_deno' + '.csv')
        
    del kkfls_btimes, terra_btimes
    gc.collect()
import sys
sys.path.append('../src/denoiser/')
sys.path.append('../src/ensemble_picker/')
import numpy as np
import h5py
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

from das_util import *
from das_denoise_models import unet

### Plot
# file_list = ['2023-12-02_20.45', '2023-12-04_13.53', '2023-12-09_06.45', '2023-12-11_23.34', '2023-12-21_14.20']
# start_pts = np.array([35, 28, 25, 0, 30]) * 25
file_list = ['2023-12-26_20.18']
start_pts = np.array([7]) * 25
win = 10 * 25
for start_pt, file in zip(start_pts, file_list):
    filename1 = 'decimator2_'+file+'.57_UTC.h5'
    filename2 = 'decimator2_'+file+'.56_UTC.h5' 
    file_dir1 = '/mnt/qnap/KKFL-S_FIberA_25Hz/'
    file_dir2 = '/mnt/qnap/TERRA_FiberA_25Hz/'
    model_dir = '../models/checkpoint_noatt_LRdecays0.8_mask0.5_raw2raw_chmax4500.pt'

    data1 = np.zeros((4500, 1500), dtype=np.float32)
    data2 = np.zeros((4500, 1500), dtype=np.float32)

    with h5py.File(file_dir1 + filename1, 'r') as f:
        time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 500:5000]
        data1[:time_data.shape[1], :time_data.shape[0]] = time_data.T
    with h5py.File(file_dir2 + filename2, 'r') as f:
        time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 500:5000]
        data2[:time_data.shape[1], :time_data.shape[0]] = time_data.T
    rawdata = np.append(data1[::-1, :], data2[:,:], axis=0)

    ### Bandpass filter
    b, a = butter(4, (0.5, 12), fs=25, btype='bandpass')
    filt = filtfilt(b, a, rawdata[np.newaxis,:,:], axis=2)
    rawdata = filt / np.std(filt, axis=(1,2), keepdims=True)


    ### Initialize the U-net
    devc = try_gpu(i=0)
    model_1 = unet(1, 16, 1024, factors=(5, 3, 2, 2), use_att=False)
    model_1 = nn.DataParallel(model_1, device_ids=[0])
    model_1.to(devc)
    ### Load the pretrained weights
    model_1.load_state_dict(torch.load(model_dir))
    model_1.eval() 

    mul_denoised = np.zeros_like(rawdata)
    _, mul_denoised[0,:,:] = Denoise_largeDAS(rawdata[0], model_1, devc, repeat=4, norm_batch=False)


    vizRawDenoise(rawdata[:,:,start_pt:start_pt+win], mul_denoised[:,:,start_pt:start_pt+win]*1.5, mul_denoised[:,:,start_pt:start_pt+win], index=range(1), model="raw-raw")

    plt.savefig(file+'quick_look.png', dpi=300)
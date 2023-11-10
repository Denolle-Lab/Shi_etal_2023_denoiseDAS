
import numpy as np
from utils import *
import torch
import gc
import seisbench.models as sbm
from ELEP.elep.ensemble_statistics import ensemble_statistics
from ELEP.elep.ensemble_coherence import ensemble_semblance 
from ELEP.elep.ensemble_learners import ensemble_regressor_cnn
from ELEP.elep import mbf, mbf_utils
from ELEP.elep import trigger_func
import obspy
from ELEP.elep.mbf_utils import make_LogFq, make_LinFq, rec_filter_coeff
from ELEP.elep.mbf import MB_filter
import matplotlib.pyplot as plt


device = torch.device("cpu")


def apply_mbf(evt_data, list_sta, list_models, MBF_paras, paras_semblance, \
               t_before=15,t_around=5,thr=0.01):
    """"
    This function takes a array of stream, a list of stations, a list of ML
    models and apply these models to the data, predict phase picks, and
    return an array of picks .
    evt_data: obspy stream with all data
    """
    twin = 6000
    nsta = len(list_sta)
    bigS = np.zeros(shape=(len(list_sta), 3, twin))
    stas = []
    for i in range(len(list_sta)):
        stream = evt_data.select(station=list_sta[i])
        if len(stream) < 3:
            # copy stream to 2 components, zero the missing data.
            tr3 = stream[0].copy()# assumed to be the vertical
            tr2 = stream[0].copy(); tr2.stats.channel = stream[0].stats.channel[0:2]+"N"
            tr1 = stream[0].copy(); tr1.stats.channel = stream[0].stats.channel[0:2]+"E"
            tr1.data = np.zeros(len(stream[0].data))
            tr2.data = np.zeros(len(stream[0].data))
            stream = obspy.Stream(traces=[tr1, tr2, tr3])
            # convert Stream into seisbench-friendly array    
            # fill in big array and order data ZNE
        bigS[i,0,:] = stream[2].data[:-1]
        bigS[i,1,:] = stream[1].data[:-1]
        bigS[i,2,:] = stream[0].data[:-1]
        stas.append(list_sta[i])


    # allocating memory for the ensemble predictions
    nwin,twin,nsta=bigS.shape[1],bigS.shape[-1],len(list_sta)
    batch_pred =np.zeros(shape=(len(list_models),nsta,twin)) 
    batch_pred_mbf =np.zeros(shape=(len(list_models),nsta,twin))
    # evaluate
    for imodel in list_models:
        imodel.eval()
    ######### Broadband workflow ################
    crap2 = bigS.copy()
    crap2 -= np.mean(crap2, axis=-1, keepdims= True) # demean data
    # original use std norm
    data_std = crap2 / np.std(crap2) + 1e-10
    # could use max data
    mmax = np.max(np.abs(crap2), axis=-1, keepdims=True)
    data_max = np.divide(crap2 , mmax,out=np.zeros_like(crap2), where=mmax!=0)
    data_tt = torch.Tensor(data_max)
    # batch predict picks.
    for ii, imodel in enumerate(list_models):
        batch_pred[ii, :, :] = imodel(data_tt.to(device))[1].detach().cpu().numpy()[:, :]

    
    ############# Multi-band Workflow ########
    windows_std = np.zeros(shape=(nsta, MBF_paras["nfqs"], 3, twin), dtype= np.float32)
    windows_max = np.zeros(shape=( nsta, MBF_paras["nfqs"], 3, twin), dtype= np.float32)
    _windows = bigS.copy();#np.zeros(shape=(nsta, 3, twin), dtype= np.float32)
    _windows_mb = np.zeros(shape=(nsta, 3, MBF_paras["nfqs"], twin), dtype= np.float32)
    # MB filter
    for ista in range(nsta): # loop over stations, it should be one in this benchmark test
        for icha in range(3): # loop over channel, there should be 3 channels total
            _windows_mb[ista, icha, :, :] = MB_filter(_windows[ista, icha], MBF_paras)
    _windows_mb = _windows_mb.swapaxes(1, 2)
    for ista in range(nsta):
        for ifreq in range(MBF_paras["nfqs"]):
            # original use std norm
            windows_std[ista, ifreq,:, :] = _windows_mb[ista, ifreq, :]  \
                / np.std(_windows_mb[ista, ifreq, :]) + 1e-10
            # others use max norm
            mmax = np.max(np.abs(_windows_mb[ista ,ifreq , :, :]), axis=-1, keepdims=True)
            windows_max[ista, ifreq,:, :] = np.divide(_windows_mb[ista, ifreq, :, :] \
                    ,mmax,out=np.zeros_like(_windows_mb[ista, ifreq, :]), where=mmax!=0)
    # print("now predicting on MBF")
    batch_pred_mbf_freq =np.zeros(shape=(len(list_models),MBF_paras["nfqs"],nsta,twin))
    for ifreq in range(MBF_paras["nfqs"]):
        # convert numpy array to torch tensor
        data_tt_mbf = torch.Tensor(windows_max[:,ifreq,:,:])
        # batch predict picks.
        for ii,imodel in enumerate(list_models):
            batch_pred_mbf_freq[ii,ifreq,:, :] = imodel(data_tt_mbf.to(device))[1].detach().cpu().numpy()[:, :]

    # take the max at each frequency
    for ista in range(nsta):
        for it in range(twin):
            for ii,imodel in enumerate(list_models):
                batch_pred_mbf[ii, ista, it] =  np.max(batch_pred_mbf_freq[ii, :, ista, it])
    
    smb_pred = np.zeros([nsta, twin], dtype = np.float32)
    smb_pred_mbf = np.zeros([nsta, twin], dtype = np.float32)
    smb_peak = np.zeros([nsta], dtype = np.float32)
    smb_peak_mbf = np.zeros([nsta], dtype = np.float32)

    # Pick the phase. 
    # all waveforms are aligned to the reference picks.
    # so all pick measurements will be made relative to the reference pick
    # all waveforms starts - 15s from reference picks
    # allow for +/- 10 seconds around reference picks.
    sfs = MBF_paras["fs"]
    istart = t_before*sfs - t_around*sfs
    iend = np.min((t_before*sfs + t_around*sfs,smb_pred.shape[1]))
    for ista in range(nsta):# should be 1 in this context
        # 0 for P-wave
        smb_pred[ista, :] = ensemble_semblance(batch_pred[:, ista, :],\
                                             paras_semblance)
        imax = np.argmax(smb_pred[ ista,istart:iend]) 
        # print("max probab",smb_pred[ista,imax+istart])
        if smb_pred[ista, imax+istart] > thr:
            smb_peak[ista] = float((imax)/sfs)-t_around

 
        # 0 for P-wave
        smb_pred_mbf[ista, :] = ensemble_semblance(batch_pred_mbf[:, ista, :], paras_semblance)
        imax = np.argmax(smb_pred_mbf[ista, istart : iend])# search for peak in the first 80 seconds
        # print("max probab",smb_pred[ista,imax+istart])
        if smb_pred_mbf[ista, imax+istart] > thr:
            smb_peak_mbf[ista] = float(imax/sfs)-t_around


    # below return the time of the first pick aas a list over stations
    return smb_peak, smb_peak_mbf

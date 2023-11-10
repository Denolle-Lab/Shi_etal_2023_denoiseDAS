import os
import gc
import glob
import h5py
import numpy as np

from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.core.event import Catalog
from obspy.clients.fdsn import Client
from obspy.geodetics.base import locations2degrees, degrees2kilometers

from das_util import next_power_of_2
from das_util import fk_filter_2cones


def ak_catalog(t1, t2, lat0=59.441, lon0=-152.028, a=-1, b=0.65):
    '''
    In:  t1, t2: start and ending timestamps
         lat0,lon0: Reference point of DAS network
         a, b: simple GMM parameters
    Out: cat : USGS AK catalog meeting GMM threshold
         ptimes : absolute P arrival times
    '''
    events = []
    ptimes = []

    # Get local catalog
    catalog = Client('USGS').get_events(
        starttime=t1,
        endtime=t2,
        catalog='ak',
        includeallorigins=True,
        includeallmagnitudes=True)

    for event in catalog:
        lon = event.origins[0]['longitude']
        lat = event.origins[0]['latitude']
        dep = event.origins[0]['depth'] * 1e-3
        mag = event.magnitudes[0]['mag']
        distdeg = locations2degrees(lat0, lon0, lat, lon)
        distkm = degrees2kilometers(distdeg)
        rad = np.sqrt(distkm ** 2 + dep ** 2)

        if (mag - 10 ** (a + b * np.log10(rad)) >= 0):
            model = TauPyModel(model='iasp91')
            arr = model.get_travel_times(
                source_depth_in_km=dep,
                distance_in_degree=distdeg)

            t0 = event.origins[0]['time']
            ptimes.append(t0 + arr[0].time)
            events.append(event)

    return Catalog(events=events), np.array(ptimes)


def ak_record_lists(rec_dir, format_part, format_full, times):
    '''
    In:  rec_dir : path to the raw data
         format_part/full: file name format
         times: event first arrival time
    Out: elist : list of files records events
         nlist : list of files of noises
    '''
    elist = []
    nlist = []

    for t_arrival in times:
        fname = UTCDateTime.strftime(t_arrival, format=format_part)
        print(rec_dir + fname)
        fname = os.path.basename(glob.glob(rec_dir + fname)[0])
        t_file = UTCDateTime.strptime(fname, format=format_full)
        if (t_arrival - t_file) > 0:
            t_eq = t_file
        else:
            t_eq = t_file - 60
        t_no = t_eq - 60

        fname = UTCDateTime.strftime(t_eq, format=format_part)
        eq_file = os.path.basename(glob.glob(rec_dir + fname)[0])
        fname = UTCDateTime.strftime(t_no, format=format_part)
        no_file = os.path.basename(glob.glob(rec_dir + fname)[0])

        elist.append(os.path.join(rec_dir, eq_file))
        nlist.append(os.path.join(rec_dir, no_file))

    return elist, nlist


def dataprep_akdas(outdir, seis_arrays, rec_dirs, format_part, format_full, times):
    for rec_dir, seis_array, f_part, f_full in zip(rec_dirs, seis_arrays, format_part, format_full):
        elist, nlist = ak_record_lists(rec_dir, f_part, f_full, times)
        # if not len(elist) == len(nlist):
        #     print('Inconsistent number of quake and noise files')
        #     raise ValueError

        all_quake = np.zeros((len(elist), 7500, 1500), dtype=np.float32)
        # all_noise = np.zeros((len(nlist), 7500, 1500), dtype=np.float32)
        raw_quake = np.zeros((len(elist), 7500, 1500), dtype=np.float32)

        for i, (eq_file, no_file) in enumerate(zip(elist, nlist)):
            with h5py.File(eq_file, 'r') as f:
                time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 100:7600]

            time_data = (time_data - np.mean(time_data)) / np.std(time_data)

            raw_quake[i, :, :time_data.shape[0]] = time_data.T

            # %% Use FK filter
            filt_cplx, mask_fk, fk2d = fk_filter_2cones(time_data,
                                                        w1=0.005,
                                                        w2=0.25,
                                                        cone1=True,
                                                        cone2=True)
            time_data = filt_cplx.real

            all_quake[i, :, :time_data.shape[0]] = time_data.T

            # with h5py.File(no_file, 'r') as f:
            #     time_data = f['Acquisition']['Raw[0]']['RawData'][:1500, 100:7600]
            #
            # time_data = (time_data - np.mean(time_data)) / np.std(time_data)
            # all_noise[i, :, :time_data.shape[0]] = time_data.T

        today = UTCDateTime.strftime(UTCDateTime.now(), format='%Y_%m_%d')
        with h5py.File(outdir + '/' + seis_array + 'till' + today + '.hdf5', 'w') as f:
            f.create_dataset("fk_quake", data=all_quake)
            f.create_dataset("raw_quake", data=raw_quake)

def main():
    seis_arrays = ['KKFLS', 'TERRA']
    rec_dirs = ['/mnt/qnap/KKFL-S_FIberA_25Hz/', '/mnt/qnap/TERRA_FiberA_25Hz/']
    # seis_arrays = ['TERRA']
    # rec_dirs = ['/mnt/qnap/TERRA_FiberA_25Hz/']
    format_part = ['decimator2_%Y-%m-%d_%H.%M.??_UTC.h5', 'decimator2_%Y-%m-%d_%H.%M.??_UTC.h5']
    format_full = ['decimator2_%Y-%m-%d_%H.%M.%S_UTC.h5', 'decimator2_%Y-%m-%d_%H.%M.%S_UTC.h5']
    outdir = '/mnt/disk2/qibin_data'

    t1 = UTCDateTime("2023-06-10T00:00:00")
    t2 = UTCDateTime("2023-08-15T00:00:00")
    _, ptimes = ak_catalog(t1, t2)

    dataprep_akdas(outdir, seis_arrays, rec_dirs, format_part, format_full, ptimes)


if __name__ == '__main__':
    main()
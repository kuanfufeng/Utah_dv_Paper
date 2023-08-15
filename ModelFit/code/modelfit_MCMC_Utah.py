#!/usr/bin/env python
# coding: utf-8

# # MCMC model fitting
# 2021.11.24 Kurama Okubo
#
# 2022.1.18 update to speed up iteration and clean up the notebook.
#
# 2022.10.5 update AIC and model selection

# This notebook conduct MCMC model fitting to estimate model parameters as well as showning the multimodality.

import datetime
import os
import time

os.environ["OMP_NUM_THREADS"] = "16"

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import h5py
import pickle

from tqdm import tqdm


# For the speed up of integral with Low level calling functoin
import ctypes
from scipy import LowLevelCallable

import emcee # MCMC sampler
import corner

# import functions for MCMC
from MCMC_func import *

from multiprocessing import Pool, cpu_count

#plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
os.environ['TZ'] = 'GMT' # change time zone to avoid confusion in unix_tvec conversion

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

if __name__ == '__main__':

    # fitting start and end time
    starttime = datetime.datetime(2006, 1, 1)
    endtime = datetime.datetime(2023, 3, 31)

    # parameters for stacking; identical to the input file of the stacking
    cc_time_unit=86400 # short-stacking time unit
    averagestack_factor=1 # length of time bin to compute mean and std
    averagestack_step=1
    datefreq = '%dD'%(averagestack_step*cc_time_unit/86400)
    
    # select the file to process
    h5_id = 1

    # select the number of iteration during the MCMC inversion
    nsteps = 6500 #15000#30000

    # set initial value and boundaries of the model parameters
    # format is: (initial value, [vmin, vmax])
    # offset, scale of GWL, delay in GWL, scale of Temp, shift of temp in days, scale in coseimic change, healing duration for SS and PF and linear trend.
    modelparam = {
                "a0"            : (0.0, [-1.0, 1.0]), # offset
                #"p1"            : (0.01, [-np.inf, np.inf]), # scale of GWL
                #"a_{precip}"      : (1e-2, [0, 1.0]), # delay in GWL [1/day]
                "p2"            : (0.01, [0, np.inf]), # scale of Temp
                "t_{shiftdays}"   : (7, [0, 90]), # shift of temp in days
                "b_{lin}"         : (0.0, [-np.inf, np.inf]), # slope of linear trend
                "logf"         : (0.0, [-10, 10]), # magnitude of uncertainity
                }

    # model case
    #modelparam["modelcase"] = "precip_temp" # "temp" "base" or "wlin"
    modelparam["modelcase"] = "temp"

    # MCMC parameters
    modelparam["nwalkers"] =  1024 # number of chains

    output_imgdir = "../figure/MCMC_modelfit"
    output_imgdir_debug = "../figure/MCMC_modelfit_dvvtrace"
    output_datadir = "../processed_data/MCMC_sampler_{}".format(nsteps)

    # set random seed to fix the initial chains
    np.random.seed(seed=20201108)
    #-------------------------------------------#

    if not os.path.exists(output_imgdir_debug):
        os.makedirs(output_imgdir_debug)

    if not os.path.exists(output_datadir):
        os.makedirs(output_datadir)

    #---Read keys from filename---# 
    casename = "UU_test"
    freqband = "2-4"
    dvvmethod = "stretching"

    #---Read csv containing channel-weighted dvv and err---#
    #fi = h5py.File(h5_stats_list[h5_id], "r")
    usecols=["uniform_tvec", "dvv", "dv_err", "temp_prism", "ppt_prism", "soil_nldas", "snow_nldas", "date"]
    root = "../../UU_csv_blank/"
    fn = root+"UU_MPU.csv"
    fi = pd.read_csv(fn,names=usecols,header=0)
    fi['date'] = fi['date'].astype(str)

    #---Compute unix time vector---#
    tolen=len(fi['date'][:])
    btimestamp=time.mktime(time.strptime(str(fi['date'][0]), "%Y-%m-%d"))
    
    #uniform_tvec_unix = np.array(fi['uniform_tvec'])
    uniform_tvec_unix = np.array( [(btimestamp+ 86400*x) for x in range(0,tolen)]) 
    uniform_tvec = np.array([datetime.datetime.utcfromtimestamp(x) for x in uniform_tvec_unix])
    unix_tvec = np.array([x.timestamp() for x in uniform_tvec])

    modelparam["averagestack_step"] = averagestack_step
    modelparam["uniform_tvec"] = uniform_tvec
    modelparam["unix_tvec"] = unix_tvec
    
    #---Trim the time series from the starttime to the endtime---#
    fitting_period_ind = np.where((uniform_tvec >= starttime) & (uniform_tvec <= endtime))[0]
    modelparam["fitting_period_ind"] = fitting_period_ind
    print('fitting_period_ind ',fitting_period_ind)

    #---Read temperature and precipitation data at Parkfield---#
    #df_tandp= pd.read_csv("../data/interped_tempandprecip_longterm.csv", header=0, sep = ',')
    #df_tandp = df_tandp.set_index("t")
    #df_tandp.index = pd.to_datetime(df_tandp.index)
    
    # Synchronize the long-period temperature and precipitation
    #df_tandp_synchronized = df_tandp[(df_tandp.index>starttime) & (df_tandp.index<endtime)]

    # store time history of trimmed precipitation and temperature
    #modelparam["precip"] = df_tandp_synchronized.precip
    #modelparam["CAVG"]   = df_tandp_synchronized.CAVG
    
    #modelparam["precip"] = np.array(fi['ppt_prism']/1000.)
    modelparam["CAVG"]   = np.array(fi['temp_prism'])
     
    # # 2nd way
    #st_center = (averagestack_factor*cc_time_unit/86400)/2
    #date_range_st = starttime + datetime.timedelta(days=st_center) # day
    #uniformdates = pd.date_range(start=date_range_st, end=endtime, freq=datefreq)
    #uniform_tvec = uniformdates.date
    #print(uniform_tvec)
    #uniform_tvec3 = [x.date() for x in uniform_tvec1]
    
    # print(uniform_tvec1)
    # print(uniform_tvec2)
    # print(uniform_tvec3)

    # for i in range(len(uniform_tvec2)):
    #     assert uniform_tvec3[i] == uniform_tvec2[i]

    #---Generate the initial model parameters with chains---#

    pos, ndim, keys = get_init_param(**modelparam)

    modelparam["pos"] = pos
    modelparam["ndim"] = ndim
    modelparam["keys"] = keys

    #---Extract station pairs used for model fitting---#
    #stationpairs = list(fi['dvv'].keys())
    stationpairs = ['MPU-MPU']
    print(stationpairs)

    #------------------------------------------------------#

    # select station ids for debug
    stationids = range(len(stationpairs))
    #stationids = [stationpairs.index("BP.EADB-BP.VCAB")]
    #print(stationids)

    # for stationpair in tqdm(stationpairs):

    for stationpair in tqdm([stationpairs[i] for i in stationids]):

        print("start processing :"+stationpair)

        # search file and skip if it exists.
        foname = output_datadir+"/MCMC_sampler_%s_%s_%s_%s.pickle"%(stationpair, freqband, modelparam["modelcase"], dvvmethod)

        if os.path.exists(foname):
            print(os.path.basename(foname) + "exists. Skipping.")
            # continue;
        #dvv_data = np.array(fi["dvv/{}/dvv".format(stationpair)])
        #err_data = np.array(fi["dvv/{}/err".format(stationpair)])
        dvv_data = np.array(fi["dvv"])
        err_data = np.array(fi["dv_err"])

        #---plot dv/v for the debug---#
        fig, ax = plt.subplots(3, 1, figsize=(8,6))
        ax[0].errorbar(uniform_tvec, dvv_data, yerr = err_data, capsize=3, ls="-", c = "r", ecolor='black')
        ax[1].plot(uniform_tvec, modelparam["CAVG"],  ls="-", c = "orange")
        #ax[2].plot(uniform_tvec,modelparam["precip"],  ls="-", c = "b")
        ax[0].set_title(stationpair)
        ax[1].set_title("temp_prism (C)")
        #ax[2].set_title("precip_prism (m)")
        plt.tight_layout()
        plt.savefig(output_imgdir_debug+"/MCMCdvv_%s_%s_%s.png"%(stationpair, freqband, modelparam["modelcase"]), format="png", dpi=150)
        plt.close()
        plt.clf()
         
        #---Trim the dvv and err time history---#
        modelparam["dvv_data_trim"] =  dvv_data #[fitting_period_ind]
        modelparam["err_data_trim"] =  err_data #[fitting_period_ind]

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                            modelparam["nwalkers"], modelparam["ndim"], log_probability,
                            # moves=[(emcee.moves.StretchMove(), 0.5),
                            #          (emcee.moves.DESnookerMove(), 0.5),], # Reference of Move: https://github.com/lermert/cdmx_dvv/blob/main/m.py
                            moves=emcee.moves.StretchMove(),
                            kwargs=(modelparam),
                            pool=pool)
            start = time.time()
            sampler.run_mcmc(pos, nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        # Save the current state.
        with open(foname, "wb") as f:
            pickle.dump(sampler, f)
            pickle.dump(modelparam, f)
            
        samples = sampler.get_chain( flat=True)
        plt.hist(samples[:, 0], nsteps, color="k", histtype="step")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$p(\theta_1)$")
        plt.gca().set_yticks([]);
        plt.savefig(output_imgdir_debug+"/MCMCdvv_%s_%s_%s_histo.png"%(stationpair, freqband, modelparam["modelcase"]), format="png", dpi=150)
        
        fig, axes = plt.subplots(modelparam["ndim"], figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["a0","p2","t_{shiftdays}", "b_{lin}", "log(f)"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        plt.savefig(output_imgdir_debug+"/MCMCdvv_%s_%s_%s_sampler.png"%(stationpair, freqband, modelparam["modelcase"]), format="png", dpi=150)
        

        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        print(flat_samples.shape)
        x=np.zeros(15)
        y=np.zeros(15)
        for i in range(1,16):
            flat_samples = sampler.get_chain(discard=100, flat=True, thin=i)
            x[i-1]=i
            y[i-1]=len(flat_samples)
    
        import corner
        flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)
        print(flat_samples.shape)
        fig = corner.corner( flat_samples, labels=labels)
        plt.savefig(output_imgdir_debug+"/MCMCdvv_%s_%s_%s_corner.png"%(stationpair, freqband, modelparam["modelcase"]), format="png", dpi=150)

        tau = sampler.get_autocorr_time()
        print(tau)
               
        

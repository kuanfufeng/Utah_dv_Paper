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
import sys
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

def sample_walkers(nsamples,flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = model_soil_temp(i, all=False, **modelparam)
        models.append(mod)
    spread = np.std(models,axis=0)
    med_model = np.median(models,axis=0)
    return med_model,spread


if __name__ == '__main__':
    
    stnm=sys.argv[1]
    
    # fitting start and end time
    starttime = datetime.datetime(2006, 1, 1)
    endtime = datetime.datetime(2023, 3, 31)

    # parameters for stacking; identical to the input file of the stacking
    cc_time_unit=86400 # short-stacking time unit
    averagestack_factor=1 # length of time bin to compute mean and std
    averagestack_step=1
    datefreq = '%dD'%(averagestack_step*cc_time_unit/86400)
    

    # select the number of iteration during the MCMC inversion
    nsteps = 12000 #30000
    burnin=200
    thin=20
    
    # set initial value and boundaries of the model parameters
    # format is: (initial value, [vmin, vmax])
    # offset, scale of GWL, delay in GWL, scale of Temp, shift of temp in days, scale in coseimic change, healing duration for SS and PF and linear trend.
                
    modelparam = {
                "a0"            : (0.0, [-1.0, 1.0]), # offset
                "p3"            : (-1.0, [-np.inf, np.inf]), # scale of GWL mimic by soil moisture equivalent water thickness
                "p2"            : (0.01, [0, np.inf]), # scale of Temp
                "t_{shiftdays}"   : (1, [0, 120]), # shift of temp in days
                "b_{lin}"         : (0.0, [-np.inf, np.inf]), # slope of linear trend
                "logf"         : (0.0, [-10, 10]), # magnitude of uncertainity
                }

    # model case
    modelparam["modelcase"] = "soil_temp" # "temp" "base" or "wlin"

    # MCMC parameters
    modelparam["nwalkers"] =  32 # number of chains

    #output_imgdir = "../figure/MCMC_modelfit"
    output_imgdir = "../figure/MCMC_test"
    output_imgdir_debug = "../figure/MCMC_modelfit_dvvtrace"
    output_datadir = "../processed_data/MCMC_sampler_{}".format(nsteps)

    # set random seed to fix the initial chains
    np.random.seed(seed=20121115)
    #-------------------------------------------#
    if not os.path.exists(output_imgdir):
        os.makedirs(output_imgdir)
    
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
    root = "../../UU_csv_blank_remean/"
    fn = root+"UU_"+stnm+".csv"
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
    # Synchronize the long-period temperature and precipitation
    # store time history of trimmed precipitation and temperature
    #modelparam["precip"] = df_tandp_synchronized.precip
    #modelparam["CAVG"]   = df_tandp_synchronized.CAVG
    
    modelparam["CAVG"]   = np.array(fi['temp_prism'])
    modelparam["soil"]   = np.array(fi['soil_nldas']) 
     

    #---Generate the initial model parameters with chains---#

    pos, ndim, keys = get_init_param(**modelparam)

    modelparam["pos"] = pos
    modelparam["ndim"] = ndim
    modelparam["keys"] = keys

    #---Extract station pairs used for model fitting---#
    stationpairs = [stnm+'-'+stnm]
    print(stationpairs)

    #------------------------------------------------------#

    # select station ids for debug
    stationids = range(len(stationpairs))

    for stationpair in tqdm([stationpairs[i] for i in stationids]):

        print("start processing :"+stationpair)

        # search file and skip if it exists.
        foname = output_datadir+"/MCMC_sampler_%s_%s_%s_%s.pickle"%(stationpair, freqband, modelparam["modelcase"], dvvmethod)

        if os.path.exists(foname):
            print(os.path.basename(foname) + "exists. Skipping.")
            # continue;
        dvv_data = np.array(fi["dvv"])
        err_data = np.array(fi["dv_err"])

        #---plot dv/v for the debug---#
        nax=int((ndim-3)/2)+1
        fig, ax = plt.subplots(3, 1, figsize=(12,9))
        ax[0].errorbar(uniform_tvec, dvv_data, yerr = err_data, capsize=3, ls="-", c = "r", ecolor='black')
        ax[1].plot(uniform_tvec, modelparam["CAVG"],  ls="-", c = "orange")
        ax[0].set_title(stationpair)
        ax[1].set_title("Temperature (C) from PRISM")
        ax[2].plot(uniform_tvec,modelparam["soil"],  ls="-", c = "b")
        ax[2].set_title("Soil Moisture EWT (m) from NLDAS")
        plt.tight_layout()
        plt.savefig(output_imgdir_debug+"/MCMCdvv_%s_%s_%s.png"%(stationpair, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.grid(True)
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
 

        
        labels = ["a0","p3","p2","t_{shiftdays}","b_{lin}", "log(f)"]

        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        
        print("theta_max:  ",theta_max)
        print(samples.shape)
        
        ### --- plotting corner
        #flat_samples = sampler.flatchain
        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        print(flat_samples.shape)
        fig = corner.corner( flat_samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], truths=theta_max,title_kwargs={"fontsize": 10})
        plt.savefig(output_imgdir+"/MCMCdvv_%s_%s_%s_corner.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.close()
        plt.clf()
        

        med_model, spread = sample_walkers(nsteps,samples)
        best_fit_model = model_soil_temp(theta_max, all=False, **modelparam)

        fig, ax = plt.subplots(3, 1, figsize=(12,9))
        for theta in samples[np.random.randint(len(samples),size=(nsteps-burnin))]:
            ax[0].plot(uniform_tvec, model_soil_temp(theta, all=False, **modelparam), color="r", alpha=0.1)
        ax[0].plot(uniform_tvec,best_fit_model, c='b', label='Highest Likelihood Model')
        ax[0].plot(uniform_tvec, dvv_data, label='Observed dvv', c='k')
        ax[0].set_title(stationpair+" and theta_max: "+str(theta_max))
        ax[1].fill_between(uniform_tvec,med_model-spread,med_model+spread,color='gold',alpha=0.5,label=r'$1\sigma$ Posterior Spread')
        ax[1].plot(uniform_tvec, med_model, c='orange',label='Highest Likelihood Model')
        ax[1].plot(uniform_tvec, dvv_data, label='observed dvv',c='k')
        ax[1].plot(uniform_tvec, best_fit_model, c='b', label='Highest Likelihood Model')
        ax[2].plot(uniform_tvec, dvv_data, label='observed dvv',c='k')
        ax[2].plot(uniform_tvec, best_fit_model, c='b', label='Highest Likelihood Model')
        ax[2].plot(uniform_tvec, dvv_data-best_fit_model, label='Residual',c='r')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        plt.savefig(output_imgdir+"/MCMCdvv_%s_%s_%s_samplers.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.close()
        plt.clf()        

                
        fig, axes = plt.subplots( modelparam["ndim"], figsize=(10, 7), sharex=True)
        samples = sampler.get_chain(discard=burnin, thin=thin)
        print("samples plot : ", samples.shape, len(samples),nsteps)
        
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.savefig(output_imgdir+"/MCMCdvv_%s_%s_%s_steps.png"%(stnm, freqband, modelparam["modelcase"]), format="png", dpi=100)
        plt.close()
        plt.clf()
    
        
               



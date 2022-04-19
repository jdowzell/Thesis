#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np

from astroquery.mast import Observations
from astroquery.mast import Catalogs

from astropy.io import fits
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries
from astropy.stats import sigma_clipped_stats
from astropy.timeseries import aggregate_downsample

import lightkurve

################################
# General imports
################################
import csv, math, io, os, os.path, sys, random, time, json, gc
from datetime import datetime
import pandas as pd
import seaborn as sb
from collections import Counter

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

################################
# Initialisers
################################

# Load the Data files
fitsarr = np.load("fitslist.npy")
C = 0

################################
# Functions
################################

# Python3 implementation to find elements
# that appeared only once
# Function to find the elements that
# appeared only once in the array
def OccurredOnce(in_arr, out_arr, count=1):
    
    n = len(in_arr)
    #counting frequency of every element using Counter
    mp=Counter(in_arr)
    # Traverse the map and print all
    # the elements with occurrence 1
    for it in mp:
        if mp[it] == count:
            out_arr.append(int(it))
            #print(it, end = " ")
    return


def MakeTimeSeriesNEW(i, L, f, binSize=0.2, timedBinSize=0.003):
    
    print("Processing File {} of {}".format(i+1,L))
    
    ts = TimeSeries.read(fitsarr[int(f)], format='tess.fits')
    periodogram = BoxLeastSquares.from_timeseries(ts, 'pdcsap_flux')
    results = periodogram.autopower(binSize * u.day)  
    best = np.argmax(results.power)  
    period = results.period[best]
    
    dropcols = ['timecorr', 'cadenceno', 'sap_flux', 'sap_flux_err', 'sap_bkg', 'sap_bkg_err', 'quality',
       'psf_centr1', 'psf_centr1_err', 'psf_centr2', 'psf_centr2_err',
       'mom_centr1', 'mom_centr1_err', 'mom_centr2', 'mom_centr2_err',
       'pos_corr1', 'pos_corr2']
    
    ts.remove_columns(dropcols)
    mean, median, stddev = sigma_clipped_stats(ts['pdcsap_flux'])
    ts['pdcsap_flux_norm'] = ts['pdcsap_flux'] / median
    
    lc = lk.LightCurve(time=range(len(ts)), flux=ts['pdcsap_flux_norm'])
    fold_lc = lc.fold(period=period, normalize_phase=True)
    
    return(fold_lc)
    
    ###########################
    
def MakeTimeSeries(i, L, f, binSize=0.2, timedBinSize=0.003):
    
    print("Processing File {} of {}".format(i+1,L))
    
    ts = TimeSeries.read(fitsarr[int(f)], format='tess.fits')
    periodogram = BoxLeastSquares.from_timeseries(ts, 'pdcsap_flux')
    results = periodogram.autopower(binSize * u.day)  
    best = np.argmax(results.power)  
    period = results.period[best]
    transit_time = results.transit_time[best]
    ts_folded = ts.fold(period=period, epoch_time=transit_time)
    mean, median, stddev = sigma_clipped_stats(ts_folded['pdcsap_flux'])
    ts_folded['pdcsap_flux_norm'] = ts_folded['pdcsap_flux'] / median
    
    # Delete unnecessary vars
    del ts, periodogram, period, results, best, transit_time, mean, median, stddev
    
    print("   > Converting to DataFrame")
    dropcols = ['timecorr', 'cadenceno', 'sap_flux', 'sap_flux_err', 'sap_bkg', 'sap_bkg_err', 'quality',
       'psf_centr1', 'psf_centr1_err', 'psf_centr2', 'psf_centr2_err',
       'mom_centr1', 'mom_centr1_err', 'mom_centr2', 'mom_centr2_err',
       'pos_corr1', 'pos_corr2']
    df = pd.DataFrame(ts_folded.to_pandas()[1:-1].drop(dropcols, axis=1)).reset_index()
    #df = df.reset_index()
    
    #print(f"   > Saving as Z_or_O_{f}.csv",)
    compression_opts = dict(method="zip", archive_name=f"Z_or_O_{f}.csv")  
    df.to_csv(f"./csv_files/Z_or_O_{f}.zip", index=False, compression=compression_opts)
    
    print("   > Done!")
    return
    
################################
# main
################################

def main():
    
    # Making sure we have only one cmd line parameter and it is the fluxlist that we want
    try:
        #print(sys.argv)
        fname = sys.argv[1]
        fluxarr = np.load(fname)
    except FileNotFoundError:
        print("This File Does Not Exist")
        return
        #break
    except IndexError:
        print("Please add exactly one numpy file to the command line!")
        return
        #break
    except:
        print("General Error")
        return
        #break
    print("Source File Loaded!")
    
    #####################
    
    npFileName = "ZeroOrOneExoplanetTICids.npy"
    try:
        ZeroOrOneTICList = np.load(npFileName)
    except FileNotFoundError:
        print("This File Does Not Exist; Making it now!")
        GetTICListOfOnlyZeroOrOneExoplanets(npFileName)
    print("File '{}' loaded".format(npFileName))
    
    #####################
    #     OVERRIDES     #
    #####################
    
    ZeroOrOneTICList = ZeroOrOneTICList[:4]
    
    #####################
    
    lenTIC = len(ZeroOrOneTICList)
    
    for i, ind in enumerate(ZeroOrOneTICList):
        #print(f"{i}: {ind}")
        MakeTimeSeries(i,lenTIC,ind)
    
    #foldedFluxes = [MakeTimeSeries(ind,lenTIC,x)['sap_flux_norm'] for x, ind in enumerate(ZeroOrOneTICList)]
    
    # Converting the above to a DataFrame to save it
    #print("Length of foldedFlux[0] is {}".format(len(foldedFluxes[0])))
    #print("fFlux[0]:\n{}".format(foldedFluxes[0]))
    #np.save("ZeroOrOneFoldedFluxes_TSLIST.npy",foldedFluxes)
    print("YAY")

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

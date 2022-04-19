#!/usr/bin/env python3

################################
# Scientific imports
################################
import gc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from astropy.io import fits
from astroquery.mast import Observations
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries

################################
# General imports
################################
import csv, math, io, os, os.path, sys, random, time, json
import pandas as pd
import seaborn as sb
from tqdm.notebook import tqdm, trange
import argparse

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

################################
# Initialisers
################################

# Load the Data files
fitsarr = np.load("fitslist.npy")

################################
# Functions
################################
def FoldTheFlux(fitsarr,f):
    ts = TimeSeries.read(fitsarr[int(f)], format='tess.fits')  
    return(ts)

def GetTimeseriesInfoOld(ts):
    periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')
    results = periodogram.autopower(0.2 * u.day)  
    best = np.argmax(results.power)  
    period = results.period[best]
    transit_time = results.transit_time[best]
    ts_folded = ts.fold(period=period, epoch_time=transit_time)
    foldedflux = list(ts_folded['sap_flux'].value)
    
    # UN-FOLD THE X AXIS SOMEHOW
    unfoldedtime = [0] * len(ts_folded.time.jd)

    for i in range(len(ts_folded.time.jd)):
        unfoldedtime[i] = [ts_folded.time.jd[i],i]
    
    G = []
    Q = list(ts_folded.time.jd.copy())

    for i,e in enumerate(Q):
        G.append({'index': i, 'time': e, 'flux': foldedflux[i]})
    
    newlist = sorted(G, key=lambda k: k['time'])
    
    FLATFLUX = [d['flux'] for d in newlist]

def GetTimeseriesInfo(ts,binsize=0.2):
    periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')
    results = periodogram.autopower(binsize * u.day)  
    best = np.argmax(results.power)  
    period = results.period[best]
    transit_time = results.transit_time[best]
    ts_folded = ts.fold(period=period, epoch_time=transit_time)
    foldedflux = list(ts_folded['sap_flux'].value)
    
    # UN-FOLD THE X AXIS SOMEHOW
    unfoldedtime = [0] * len(ts_folded.time.jd)

    for i in range(len(ts_folded.time.jd)):
        unfoldedtime[i] = [ts_folded.time.jd[i],i]
    
    G = []
    Q = list(ts_folded.time.jd.copy())

    for i,e in enumerate(Q):
        G.append({'index': i, 'time': e, 'flux': foldedflux[i]})
    
    newlist = sorted(G, key=lambda k: k['time'])
    
    FLATFLUX = [d['flux'] for d in newlist]
    
    # ts = timeseries
    # everything else comes from that
    
    # periodigram          (FROM ts)
    # results              (FROM periodigram)
    # best                 (FROM results)
    # period               (FROM results & best)
    # transit_time         (FROM results & best)
    # ts_folded            (FROM period & transit_time)
    # ts_folded.time.jd    (FROM ts_folded)
    # foldedflux           (FROM ts_folded)
    
    return (FLATFLUX)


################################
# main
################################

def main():
    
    # Generating FitsList  #
    print("### Generating FitsList ###")
    
    sectors = 160
    fitsLen = len(fitsarr)
    maxLen = fitsLen // sectors
    
    print("### Flattening Fluxes ###")
    
    for x in range(sectors):
        
        list_of_flattened_fluxes = [0] * maxLen
        
        tS = x * maxLen
        tE = (x+1) * maxLen
        print("From {} to {}".format(tS,tE-1))
        for y in range(tS,tE):
            #continue
            #print(f"{y:<6} ",end='')
            newfluxstuff = FoldTheFlux(fitsarr,tS)
            print(newfluxstuff[0])
            list_of_flattened_fluxes.append(newfluxstuff)
            
            
        npName = "timeseries_"+str(x)+".log"
        print(npName)
        
        #np.save(npName, list_of_flattened_fluxes)
        with open(npName, mode="w") as f:
            for item in list_of_flattened_fluxes:
                f.write(str(item))
        print("Saved")
        
        # Garbage Cleanup
        del list_of_flattened_fluxes
        gc.collect()
        print("Cleaned Up")

#    for i in range(fitsLen):
#        file = fitsarr[i]
#        if i%(fitsLen/100)==0:
#            print("{} percent complete".format(i/160))
#        #list_of_flattened_fluxes.append(GetTimeseriesInfo(FoldTheFlux(fitsarr,i)))
#        list_of_flattened_fluxes.append(FoldTheFlux(fitsarr,i))
#        print("#",end='')
#
#    np.save('newfoldedflux.npy', list_of_flattened_fluxes)

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

################################
# Scientific imports
################################
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astroquery.mast import Observations
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries

###
# General imports
###
import csv, math, io, os, os.path, sys, random, time
import pandas as pd
import seaborn as sb
from tqdm.notebook import tqdm, trange

###
# SciKitLearn Imports
###
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from IPython.display import display

################################
# MatPlotLib Settings
################################
plt.rcParams["figure.figsize"] = (20,9)
sb.set()

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
#fluxarr = np.load("fluxlist.npy")
#planetarr = np.load("planetlist.npy")
#isplanetarr = np.load("isplanetlist.npy")

################################
# Functions
################################

def FoldTheFlux(f):
    ts = TimeSeries.read(fitsarr[int(f)], format='tess.fits')  
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
    
    return (FLATFLUX)

################################
# main
################################

def main():
    
    list_of_flattened_fluxes = []

    for i in range(len(fitsarr)):
        file = fitsarr[i]
        if i%(len(fitsarr)/100)==0:
            print("{} percent complete".format(i/160))
    #    with fits.open(file, mode="readonly") as hdulist:
    #        sap_fluxes = hdulist[1].data['SAP_FLUX']
            #print(sap_fluxes)
        list_of_flattened_fluxes.append(FoldTheFlux(i))
        #print("#",end='')
        
    # REMOVE NANs
    
    
    # SAVE
    np.save('foldedflux.npy', list_of_flattened_fluxes)

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()
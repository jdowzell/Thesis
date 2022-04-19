#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

import scipy
from scipy.signal import butter,filtfilt

import sktime as skt

from sktime.datatypes._panel._convert import from_2d_array_to_nested, from_nested_to_2d_array, is_nested_dataframe

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

################################
# General imports
################################
import csv, math, io, os, os.path, sys, random, time, json, gc
from datetime import datetime
import pandas as pd
import seaborn as sb
from collections import Counter
import joblib
from joblib import Parallel, delayed, dump, load

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

################################
# Initialisers
################################
default_rc_params = (16,5)
plt.rcParams["figure.figsize"] = default_rc_params
sb.set()

# Load the Data files
fitsarr = np.load("fitslist.npy")
xNaNs = np.load("X_NAN_LIST.npy")
xTime = np.load("X_TIME_LIST.npy")

################################
# Functions
################################

def GetFlux(f):
    
    with fits.open(f, mode="readonly") as hdulist:
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

    return (pdcsap_fluxes[1:-1])

################################

def GetNumDays():
    
    nDays = xTime[-1]-xTime[0]
    
    return (nDays)

################################

def FilterMyData(y):
    
    # First, let's calculate the observational time period;
    # This is done separately so that I can change this in the future for any TESS fits file
    numdays       = GetNumDays()
    
    # Next, fix data                           
    yMedian       = np.median(y)                                                    # Get the median value of 'y' before changing it
    y             = [yMedian if n in xNaNs else item for n,item in enumerate(y)]    # Change all the missing values to the median value of the whole array
    
    # Frequency Data Stuff
    sec           = numdays*24*60*60   # Number of seconds in the overall observation period
    freq          = len(y)/sec         # Frequency, in Hz, ie number of observations per second
    cutoff        = 0.1*freq           # HYPERPARAMETER MAYBE???????? (has to be 0 < cutoff < 0.5 because of how normal cutoff works)
    
    order         = 2                  # Approximation via polynomial of the order'th degree (2=quadratic, 3=cubic, 4=quartic, etc)
    
    # Butter Lowpass Filter
    nyq           = 0.5 * freq
    normal_cutoff = cutoff / nyq
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    newY          = filtfilt(b, a, y)
    
    # Finally, return the new X and Y values
    return (newY)

################################

def NormaliseFlux(f):
    
    # Normalise the Flux (y co-ords)
    mean = np.median(f)
    std=np.std(f)
    f[:] = [(number/mean) for number in f]
    
    # Return nornalised flux
    return (f)

################################

def FilterAllFluxes(flux,n):
    
    # Get Flux
    #flux = GetFlux(f)               # Get regular Flux
    filtFlux = NormaliseFlux(flux)  # Normalise data
    filtFlux = FilterMyData(flux)   # Filter the data
    filtFlux = filtFlux[::n]        # Keep only every 10th datapoint
    
    return (filtFlux)

################################
# main
################################

def main():
    
    # Create Filtering Number
    every_nth_pt = 10
    
    # Load base fluxlist
    fluxlist = np.load("None_Or_One_Exoplanet.npy")[:,1:-1]
    
    print(len(fluxlist),len(fluxlist[0][::every_nth_pt]))
    
    # Create empty array
    list_of_filtered_fluxes = np.zeros((len(fluxlist),len(fluxlist[0][::every_nth_pt])))
    #list_of_filtered_fluxes = np.zeros((len(fitsarr),len(GetFlux(fitsarr[0])[::every_nth_pt])))
    
    print("Processing File: ",end='')
    
    # Start Loop
    for idx,dat in enumerate(fluxlist):
        list_of_filtered_fluxes[idx] = Parallel(n_jobs=10)(delayed(FilterAllFluxes)(dat,every_nth_pt) for dat in [dat])[0]
        list_of_filtered_fluxes = list_of_filtered_fluxes.astype('>f4')

    
    print(f"LOFF has a length of {len(list_of_filtered_fluxes)}")
        
    # SAVE
    np.save("filteredfluxlistONEORNONE.npy", list_of_filtered_fluxes)

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()
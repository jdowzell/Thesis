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

################################
# SciKitLearn Imports
################################
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from scipy.signal import butter,filtfilt

from IPython.display import display

################################
# MatPlotLib Settings
################################
#plt.rcParams["figure.figsize"] = (20,9)
sb.set()

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter  (action='ignore', category=UserWarning)
warnings.simplefilter  (action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="once", category=np.VisibleDeprecationWarning)

warnings.filterwarnings("ignore")

################################
# Initialisers
################################
xNaNs = np.load("X_NAN_LIST.npy")
xTime = np.load("X_TIME_LIST.npy")

################################
# Functions
################################

def GetPositiveRates(dataArr, checkArr, param_grid): #, ncomp=8):
    
    # Make a PCA Pipeline
    print("> GPR-START")
    #pca = PCA(svd_solver='randomized', n_components=ncomp, whiten=True, random_state=42) # Number of components is something to be discussed
    #svc = SVC(kernel='rbf', class_weight='balanced')
    #model = make_pipeline(pca, svc)
    
    #model = SVC(kernel='rbf', class_weight='balanced')
    model = RandomForestClassifier(n_estimators=100)
    
    # Sort data into Test and Train
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataArr, checkArr, random_state=42)
    
    # Do gridsearch for svc params
    print("> GPR-GRIDSEARCH")
    grid = GridSearchCV(model, param_grid)
    
    # Fit model
    print("> GPR-FIT")
    grid.fit(Xtrain, ytrain)
    
    # Use svc params and predict
    print("> GPR-PREDICT")
    
    #print("> > Best parameter (CV score=%0.3f):" % grid.best_score_)
    #print("> > {}".format(grid.best_params_))
    moreStats = grid.cv_results_
    
    model = grid.best_estimator_
    yfit = model.predict(Xtest)
    
    # Now that model has done, time for confusion matrix shenanigans
    print("> GPR-CONFUSION")
    mat = confusion_matrix(ytest, yfit)
    
    return (mat, moreStats)

################################

def Every_Nth_Value(y, n=10):
    return (y[::n])

################################

def GetNumDays():
    
    nDays = xTime[-1]-xTime[0]
    
    return (nDays)

################################

def FilterMyData(y,cutVAR=0.000005):
    
    # First, let's calculate the observational time period;
    # This is done separately so that I can change this in the future for any TESS fits file
    numdays       = GetNumDays()
    
    # Next, fix data                           
    yMedian       = np.median(y)                                                    # Get the median value of 'y' before changing it
    y             = [yMedian if n in xNaNs else item for n,item in enumerate(y)]    # Change all the missing values to the median value of the whole array
    
    # Frequency Data Stuff
    sec           = numdays*24*60*60   # Number of seconds in the overall observation period
    freq          = len(y)/sec         # Frequency, in Hz, ie number of observations per second
    # FREQ IS APPROX 1/120 OR ~0.008333333
    
    #cutoff        = cutVAR*freq        # HYPERPARAMETER NOW!!!!!!!! (has to be 0 < cutoff < 0.5 because of how normal cutoff works)
    cutoff        = cutVAR
   
    order         = 2                  # Approximation via polynomial of the order'th degree (2=quadratic, 3=cubic, 4=quartic, etc)
    
    # Butter Lowpass Filter
    nyq           = 0.5 * freq
    normal_cutoff = cutoff / nyq
    
    #print(f"FREQ: {freq:8f}")# \t\t NORM CUTOFF: {normal_cutoff:8f}") 
    
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
# main
################################

def main():
    
    # Initial setups; to allow for future master-file-ification later on
    targetname = 'RandomTree'
    # Later on will be taken from a command line argument, or even an option (" > get_metrics.py --fourier true --algorithm SVM --nth_record 10")
    
    print("##########\nTIME START")
    tStart = datetime.now()
    print (tStart)
    
    print("##########\nLOADING FILES")

    # Load the Data files
    fluxarrFULL     = np.load("None_Or_One_Exoplanet_FILT_NORM_SUBS.npy")
    isplanetarrFULL = np.load("one_or_none_isplanetlist.npy")
    
    print("##########\nUSING EVERY NTH RECORD")
    
    # If I want to take every "n"th record, now I just have to flip a switch instead of commenting-and-uncommenting shenanigans
    every_nth_record     = False
    every_nth_record_num = 20          # Maybe make a CMD line argument? Future work perhaps
    
    # Initialise appropriate data arrays
    if every_nth_record == True:
        print("> Using every {}th record".format(every_nth_record_num))
        fluxarr, _, isplanetarr, _ = train_test_split(fluxarrFULL, isplanetarrFULL, random_state=42, train_size=1/every_nth_record_num)
        
    else:
        print("> False")
        fluxarr = fluxarrFULL
        isplanetarr = isplanetarrFULL
    
    print("##########\nGENERATING PARAM GRID")
    
    param_grid = {'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]}
    
    print("##########\nRUNNING GETPOSITIVERATES")
    
    confMat, moreStats = GetPositiveRates(fluxarr, isplanetarr, param_grid) #, ncomp=pca_n_components)
    (TN, FN), (FP, TP) = confMat.T
    
    print("##########\nTIME FINISH")
    
    # Calculating the total time taken to process
    tFin = datetime.now()
    tDelta = tFin - tStart
    mins = (math.floor(tDelta.seconds/60))
    timetaken = "Process took {} minutes and {} seconds".format(mins, tDelta.seconds - (60*mins))
    print(tFin)
    
    print("##########\nGATHERING STATS AND WRITING FILE")
    
    # Preparing the stats text
    data = {}
    data[targetname] = []
    data[targetname].append({
        'tstart': tStart,
        'tdelta': tDelta,
        'tfinish': tFin,
        'TN' : TN,
        'FP' : FP,
        'FN' : FN,
        'TP' : TP,
        'dateran': tStart.replace(microsecond=0),
        'CV' : moreStats
    })
    
    # File saving stuff
    targetdest="./confusionmatrices/"
    fname = targetname+"_filtered.json"
    
    print("Saving {}".format(fname))
    
    # Write all the info to a file
    with open(targetdest+fname, "w") as f:
        #f.write(stats)
        json.dump(data, f, indent=4, default=str)
    
    print("##########\nFINSIHED\n##########\n")

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()
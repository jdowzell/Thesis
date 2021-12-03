#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from astropy.io import fits

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

from scipy.signal import butter,filtfilt

import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import SpectralClustering

from sklearn.preprocessing import FunctionTransformer

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
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

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
    
    print("> GPR-TRANSFORMERS")
    nrm = FunctionTransformer(NormaliseFlux)
    flt = FunctionTransformer(FilterMyData)
    svc = SVC(kernel='rbf', class_weight='balanced')
    
    #print(f"##############################\n{flt.get_params().keys()}\n##############################")
    
    print("> GPR-MAKE-PIPELINE")
    #model = make_pipeline(nrm,flt,svc)
    #model = make_pipeline(flt,svc)
    model = Pipeline(steps=[
         ('nrm', FunctionTransformer(NormaliseFlux)),
         ('flt', FunctionTransformer(FilterMyData)),
         ('svc', SVC(kernel='rbf', class_weight='balanced'))
    ])
    
    print("> GPR-TEST_TRAIN_SPLIT")
    # Sort data into Test and Train
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataArr, checkArr, random_state=42)
    
    # Do gridsearch for svc params
    print("> GPR-GRIDSEARCH")
    grid = GridSearchCV(model, param_grid, n_jobs=10)
    
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

def Every_Nth_Value(y,n):
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
    
    print(f"FREQ: {freq:8f}")# \t\t NORM CUTOFF: {normal_cutoff:8f}") 
    
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    newY          = filtfilt(b, a, y)[::10]
    
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
    targetname = 'SVM'
    # Later on will be taken from a cmd line arg / a cmd line option (" > get_metrics.py --fourier true --algorithm SVM --nth_record 10")
    
    print("##########\nTIME START")
    tStart = datetime.now()
    print (tStart)
    
    print("##########\nLOADING FILES")

    # Load the Data files
    fluxarrFULL = np.load("filteredfluxlist.npy")
    isplanetarrFULL = ["Planet" if x==True else "Not Planet" for x in np.load("isplanetlist.npy")]
    #isplanetarrFULL = ["Planet" if x==1 else "Not Planet" for x in np.load("one_or_none_isplanetlist.npy")]
    
    print("##########\nUSING EVERY NTH RECORD")
    
    # If I want to take every "n"th record, now I just have to flip a switch instead of commenting-and-uncommenting shenanigans
    every_nth_record = True
    every_nth_record_num = 20                                  # Maybe make a CMD line argument? Future work perhaps
    
    # Initialise appropriate data arrays
    if every_nth_record == True:
        print("> Using every {}th record".format(every_nth_record_num))
        fluxarr, _, isplanetarr, _ = train_test_split(fluxarrFULL, isplanetarrFULL, random_state=42, train_size=1/every_nth_record_num)
    else:
        print("> False")
        fluxarr = fluxarrFULL
        isplanetarr = isplanetarrFULL
    
    print("##########\nGENERATING PARAM GRID")
    
    param_grid = {'flt__kw_args': [{'cutVAR': [0.000005, 0.00001, 0.000015, 0.00002]}],           #list(np.linspace(0.00001,0.0001,10,True))}],
                  #'flt__cutVAR': list(np.linspace(0.05,0.5,10,True)),
                  'svc__C': [0.01, 0.1, 1, 5, 10],#, 50],
                  'svc__gamma': [0.000001, 0.00001, 0.0001, 0.0005, 0.001]}#, 0.005]}
    
    print("##########\nRUNNING GETPOSITIVERATES")
    
    confMat, moreStats = GetPositiveRates(fluxarr, isplanetarr, param_grid)
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
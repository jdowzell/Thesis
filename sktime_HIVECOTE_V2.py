#!/usr/bin/env python3
#
#
#
#
#
#
#                      ██████   ██████       ██████  ███    ██ ███████                          
#                      ██   ██ ██    ██     ██    ██ ████   ██ ██                               
#                      ██   ██ ██    ██     ██    ██ ██ ██  ██ █████                            
#                      ██   ██ ██    ██     ██    ██ ██  ██ ██ ██                               
#                      ██████   ██████       ██████  ██   ████ ███████                          
#                                                                         
#                                                                         
#                       █████  ██       ██████   ██████  ██████  ██ ████████ ██   ██ ███    ███ 
#                      ██   ██ ██      ██       ██    ██ ██   ██ ██    ██    ██   ██ ████  ████ 
#                      ███████ ██      ██   ███ ██    ██ ██████  ██    ██    ███████ ██ ████ ██ 
#                      ██   ██ ██      ██    ██ ██    ██ ██   ██ ██    ██    ██   ██ ██  ██  ██ 
#                      ██   ██ ███████  ██████   ██████  ██   ██ ██    ██    ██   ██ ██      ██ 
#                                                                         
#                                                                         
#                       █████  ████████      █████      ████████ ██ ███    ███ ███████          
#                      ██   ██    ██        ██   ██        ██    ██ ████  ████ ██               
#                      ███████    ██        ███████        ██    ██ ██ ████ ██ █████            
#                      ██   ██    ██        ██   ██        ██    ██ ██  ██  ██ ██               
#                      ██   ██    ██        ██   ██        ██    ██ ██      ██ ███████          
#
#
#
#


################################
# General Imports
################################
import csv, math, io, os, os.path, sys, random, time, json, gc, glob
from datetime import datetime
import joblib
from joblib import Parallel, delayed, dump, load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

################################
# Multiprocessing maybe?
################################
import multiprocessing
multiprocessing

################################
# Scientific Imports
################################
import scipy
from scipy.signal import butter,filtfilt

################################
# SKLearn Imports
################################
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

################################
# SKTime Imports
################################
from sktime.datatypes._panel._convert import from_2d_array_to_nested, from_nested_to_2d_array, is_nested_dataframe
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV

from sktime.classification.hybrid import HIVECOTEV2
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

################################
# Initialisers
################################
default_rc_params = (16,9)
plt.rcParams["figure.figsize"] = default_rc_params
sb.set()

xNaNs = np.load("X_NAN_LIST.npy")
xTime = np.load("X_TIME_LIST.npy")

################################################################################################################################################################

################################
# Functions
################################

def Every_Nth_Value_EACH(y,nth=40):
    return (y[::nth])

################################

def Every_Nth_Value(masterX,nth=40):
    
    biglen = len(masterX)
    oldlen = len(masterX[0])
    newlen = len(masterX[0][::nth])
    #print(f"Old = {oldlen}; new = {newlen}")
    
    tmp = np.zeros((biglen,newlen))
    
    for n,X in enumerate(masterX):
        tmp[n] = Every_Nth_Value_EACH(X)
    
    return tmp

################################

def GetNumDays(time=xTime):
    
    #xTime = np.load("X_TIME_LIST.npy")
    nDays = time[-1]-time[0]
    
    return (nDays)

################################

def FilterMyData(x,cutoff=0.00005,order=2,xNaNs=xNaNs):
    
    """
    Function to apply a Butter Filter to timeseries.
    Vars:
    
    y:        The timeseries. Must be list or np array.
    cutoff:   The cutoff frequency. Used to determine where the filter cut off is.
    order:    Approximation via polynomial of the order'th degree (2=quadratic, 3=cubic, 4=quartic, etc)
    """
    
    # First, let's calculate the observational time period;
    # This is done separately so that I can change this in the future for any TESS fits file
    numdays       = GetNumDays()
    
    # Next, fix data                           
    xMedian       = np.median(x)                                                    # Get the median value of 'x' before changing it
    x             = [xMedian if n in xNaNs else item for n,item in enumerate(x)]    # Change all the missing values to the median value of the whole array
    
    # Frequency Data Stuff
    sec           = numdays*24*60*60   # Number of seconds in the overall observation period
    freq          = len(x)/sec         # Frequency, in Hz, ie number of observations per second
    # FREQ IS APPROX 1/120 OR ~0.008333333
    
    # Butter Lowpass Filter
    #polynomOrder  = order
    nyq           = 0.5 * freq
    normal_cutoff = cutoff / nyq
    #b, a          = butter(polynomOrder, normal_cutoff, btype='low', analog=False)
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    newX          = filtfilt(b, a, x)
    
    # Finally, return the new X and Y values
    return (newX)

################################

def FilterAllMyData(masterX,cutoff=0.00005,order=2,nanList=xNaNs):
    
    # Input:  masterX
    # Output: masterX with each LC filtered
    
    for X in masterX:
        X[:] = FilterMyData(X,cutoff,order,xNaNs)
    
    return masterX
    
################################

def Normal(masterX):
    
    # Takes in 'masterX', my 9154 long array of LCs.
    # Need to return a 9154 array, where the daya has been normalised for EACH LC
    for X in masterX:
        median = np.median(X)
        X[:] = np.asarray([(number/median) for number in X])
    
    return masterX

################################

def FIXNAN(masterX, nanList=xNaNs):
    
    # Takes in 'masterX', my 9154 long array of LCs.
    # Need to return a 9154 array, where the daya has been normalised for EACH LC
    for X in masterX:
        #print(f"\t> Length of X is {len(X)}")
        XMedian = np.median(X)
        X[:]= np.asarray([XMedian if n in nanList else item for n,item in enumerate(X)])
    
    return masterX

################################

def MakeNested(masterX):

    X_nested = from_2d_array_to_nested(np.array(masterX))
    
    return X_nested

################################

def GetMetrics(X_arr, Y_arr, param_grid):
    
    # Make a PCA Pipeline
    print("> START")
    
    algorithm = HIVECOTEV2(n_jobs=-1,)
    print(f"\t> Model: {algorithm}")
    
    # Make the transformers
    print("> GENERATING TRANSFORMERS")
    fnan = FunctionTransformer(FIXNAN)
    norm = FunctionTransformer(Normal)
    filt = FunctionTransformer(FilterAllMyData)
    enth = FunctionTransformer(Every_Nth_Value)
    mnst = FunctionTransformer(MakeNested)
    
    # Construct the Pipeline
    print("> MAKE PIPELINE")
    pipe = Pipeline(steps=[['fixnan',fnan],['normalise',norm],['filter',filt]], verbose=True) #,['everynth',enth],['nb', algorithm]])
    
    print("> INITIAL TRANSFORMATION")
    pipe.transform(X_arr)
    
    print("> PERFORM SUBSAMPLING")
    X_arr = enth.transform(X_arr)
    
    print("> MAKE NESTED")
    X_nested = mnst.transform(X_arr)
    
    # Perform data manipulation
    print("> TEST-TRAIN-SPLIT")
    #nestedX = MakeNested(X_arr)
    X_train, X_test, y_train, y_test = train_test_split(X_nested, Y_arr, random_state=42)
    
    # Do gridsearch for svc params
    print("> GRIDSEARCH")
    grid = GridSearchCV(algorithm, param_grid, return_train_score=True, n_jobs=-1)
    
    # Fit model
    print("> FIT")
    grid.fit(X_train, y_train)
    
    # Use svc params and predict
    print("> MAKESTATS")
    moreStats = grid.cv_results_
    #print("> > Best parameter (CV score=%0.3f):" % grid.best_score_)
    #print("> > {}".format(grid.best_params_))
    
    # Use svc params and predict
    print("> PREDICT")
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    
    # Get Acc/Pre/Res
    print("> CALCULATING RELIABILITY METRICS")
    mAcc = accuracy_score(y_test, y_pred)
    mPre = precision_score(y_test, y_pred)
    mRec = recall_score(y_test, y_pred)
    #metrics = (mAcc, mPre, mRec)
    
    # Now that model has done, time for confusion matrix shenanigans
    print("> CONFUSION")
    mat = confusion_matrix(y_test, y_pred)
    
    #print(f"Y_TEST = {y_test}\n"
    #      f"Y_PRED = {y_pred}")
    
    return (mat, moreStats, mAcc, mPre, mRec)

################################

def WriteJSON(targetname, tStart, tFin, tDelta, TN, FP, FN, TP, acc, pre, rec, stats):
    
    data = {}
    data[targetname] = []
    data[targetname].append({
        'dateran': tStart.replace(microsecond=0),
        'tstart': tStart,
        'tfinish': tFin,
        'tdelta': tDelta,
        'TN' : TN,
        'FP' : FP,
        'FN' : FN,
        'TP' : TP,
        'Accuracy' : acc,
        'Precision' : pre,
        'Recall' : rec,
        'CV' : stats
    })

    # File saving stuff
    fname = targetname+".json"
    targetdest = "./NEW_RESULTS/"

    print("Saving {}".format(fname))

    # Write all the info to a file
    with open(targetdest+fname, "w") as f:
        #f.write(stats)
        json.dump(data, f, indent=4, default=str)
        
################################
################################
# main
################################

def main():
    
    ############################
    # Data Initialisers
    ############################
    masterX = np.load("True_NOO_fluxes.npy")
    masterY = np.load("True_NOO_isplanetlist.npy")
    
    # TESTING PURPOSES ONLY
    masterX = masterX[::30]
    masterY = masterY[::30]
    
    print(f"Length of x-arr: {len(masterX)}\nLength of y-arr: {len(masterY)}")
    
    ############################
    # Parameter Grid Setup
    ############################
    
    # param_grid = {
    #     stc_params: [
    #         "estimator": [RotationForest(n_estimators=3)],
    #         "n_shapelet_samples": [500, 1000, 1500],
    #         "max_shapelets": [50, 100, 150],
    #         "batch_size": [500, 1000, 1500],
    #     ],
    #     drcif_params: [
    #         'n_estimators': [10, 150, 500, 800],
    #         'base_estimator': ['DTC']
    #     ],
    #     arsenal_params: [
    #         'num_kernels': [1000, 2000, 3000],
    #         'n_estimators': [15, 20, 25, 30, 35]
    #     ],
    #     tde_params: [
    #         "n_parameter_samples": [100, 150, 200],
    #         "max_ensemble_size": [500, 1000, 1500],
    #         "randomly_selected_params": [50, 100, 150],
    #     ]
    # }
    
    stc_params={
        "n_shapelet_samples": 100,
        "max_shapelets": 10,
        "batch_size": 20,
    }
    drcif_params={
        "n_estimators": 2,
        "n_intervals": 2,
        "att_subsample_size": 2
    }
    arsenal_params={
        "num_kernels": 50,
        "n_estimators": 3
    }
    tde_params={
        "n_parameter_samples": 10,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    }

    param_grid = {
        'stc_params':     list(stc_params.items()),
        'drcif_params':   list(drcif_params.items()),
        'arsenal_params': list(arsenal_params.items()),
        'tde_params':     list(tde_params.items())
    }

    param_grid = np.array(list(param_grid.items()), dtype=object)
    
    
    ############################
    # Loop Start
    ############################
    
    tStart = datetime.now()
    print(f"Staring Loops!\n####################\nStart Time: {tStart}\n####################\n")
    
    #Parallel(n_jobs=-1)(delayed(DoTheStuff)(classifier, param_grid, masterX, masterY) for classifier in list_of_classifiers)
    confMat, moreStats, acc, pre, rec = GetMetrics(masterX, masterY, param_grid)
    
    ((TN, FN), (FP, TP)) = confMat.T
    
    # End Timer and get time stats
    tFin = datetime.now()
    
    print(f"\n####################\nFinish Time: {tFin}\n####################\n")
    
    tDelta = tFin - tStart
    mins = (math.floor(tDelta.seconds/60))
    
    WriteJSON("sktime-HIVECOTEV1", tStart, tFin, tDelta, TN, FP, FN, TP, acc, pre, rec, moreStats)

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()
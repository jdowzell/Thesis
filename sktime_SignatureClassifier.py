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

from sktime.classification.kernel_based import Arsenal
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.interval_based import DrCIF
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.dictionary_based import IndividualBOSS
from sktime.classification.dictionary_based import IndividualTDE
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
#from sktime.classification.feature_based import MatrixProfileClassifier
from sktime.classification.dictionary_based import MUSE
from sktime.classification.interval_based import RandomIntervalSpectralForest
#from sktime.classification.distance_based import ShapeDTW
from sktime.classification.feature_based import SignatureClassifier
from sktime.classification.interval_based import SupervisedTimeSeriesForest
#from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.dictionary_based import WEASEL

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

def GetMetrics(classifierType, X_arr, Y_arr, param_grid):
    
    # Make a PCA Pipeline
    print("> START")
    
    algorithm = classifierType()
    cname     = classifierType.__name__
    pipecname = cname.lower()
    print(f"\t> Model: {cname}")
    
    # Make the transformers
    print("> GENERATING TRANSFORMERS")
    fnan = FunctionTransformer(FIXNAN)
    norm = FunctionTransformer(Normal)
    filt = FunctionTransformer(FilterAllMyData)
    enth = FunctionTransformer(Every_Nth_Value)
    mnst = FunctionTransformer(MakeNested)
    
    # Construct the Pipeline
    print("> MAKE PIPELINE")
    #model = make_pipeline(flt,nth,algorithm)
    pipe = Pipeline(steps=[['fixnan',fnan],['normalise',norm],['filter',filt],['everynth',enth],['makenested', mnst],[pipecname, algorithm]])
    #print(pipe)
    
    # Perform data manipulation
    print("> TEST-TRAIN-SPLIT")
    #nestedX = MakeNested(X_arr)
    X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, random_state=42)
    
    # Do gridsearch for svc params
    print("> GRIDSEARCH")
    grid = GridSearchCV(pipe, param_grid, return_train_score=True)
    
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
    
    print(f"Y_TEST = {y_test}\n"
          f"Y_PRED = {y_pred}")
    
    return (mat, moreStats, mAcc, mPre, mRec)

################################

def WriteJSON(algorithm, tStart, tFin, tDelta, TN, FP, FN, TP, acc, pre, rec, stats):
    
    targetname = algorithm.__name__
    
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
    targetdest = "./sktime_results/"

    print("Saving {}".format(fname))

    # Write all the info to a file
    with open(targetdest+fname, "w") as f:
        #f.write(stats)
        json.dump(data, f, indent=4, default=str)
        
################################

def DoTheStuff(classifier, full_param_grid, masterX, masterY):

    cname = classifier.__name__
    print(f"Model: {cname}")
    
    # Parameter Grid
    param_grid = full_param_grid[cname]
    
    # Start Timer
    tStart = datetime.now()
    
    # Confusion Matric Stuff
    confMat, moreStats, acc, pre, rec = GetMetrics(classifier, masterX, masterY, param_grid)
    
    ((TN, FN), (FP, TP)) = confMat.T
    
    # End Timer and get time stats
    tFin = datetime.now()
    tDelta = tFin - tStart
    mins = (math.floor(tDelta.seconds/60))
    
    WriteJSON(classifier, tStart, tFin, tDelta, TN, FP, FN, TP, acc, pre, rec, moreStats)

################################
################################
################################
# main
################################

def main():
    
    ############################
    # Data Initialisers
    ############################
    masterX = [x[1:-1] for x in np.load("None_Or_One_Exoplanet.npy")] # <--- x[1:-1] because this trims off the leading/trailing 0 present on every LC
    masterY = np.load("None_Or_One_isplanetlist.npy")
    
    print(f"Length of x-arr: {len(masterX)}\nLength of y-arr: {len(masterY)}")

    ############################
    # Classifier List Setup
    ############################
    
    list_of_classifiers = [
        CanonicalIntervalForest,
        ContractableBOSS,
        DrCIF,
        SignatureClassifier,
        SupervisedTimeSeriesForest
    ]
    
    ############################
    # Parameter Grid Setup
    ############################
    
    full_param_grid = {
        
        # Algorithm: {
        #   'algorithm__key1': [vals],
        #   'algorithm__key2': [vals],
        # },
        
    'CanonicalIntervalForest':
            {'canonicalintervalforest__n_estimators': [50, 200, 250, 300, 800], 'canonicalintervalforest__base_estimator': ['CIT', 'DTC']},
    'ContractableBOSS':
            {'contractableboss__n_parameter_samples': [50, 200, 250, 300, 800], 'contractableboss__max_ensemble_size': [10, 45, 50, 55, 100] },
    'DrCIF':
            {'drcif__n_estimators': [10, 150, 200, 250, 800], 'drcif__base_estimator': ['CIT', 'DTC']},
    'WEASEL':
            {'weasel__anova': [True, False], 'weasel__window_inc': [2, 3, 4, 5, 6]}
    }
    

    ############################
    # Loop Start
    ############################
    
    print("Staring Loops!\n####################\n")
    
    Parallel(n_jobs=16)(delayed(DoTheStuff)(classifier, full_param_grid, masterX, masterY) for classifier in list_of_classifiers)

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

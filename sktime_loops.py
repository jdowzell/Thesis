#!/usr/bin/env python3

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
from sktime.classification.feature_based import MatrixProfileClassifier
from sktime.classification.dictionary_based import MUSE
from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.classification.distance_based import ShapeDTW
from sktime.classification.feature_based import SignatureClassifier
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.dictionary_based import WEASEL

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
default_rc_params = (16,9)
plt.rcParams["figure.figsize"] = default_rc_params
sb.set()

xNaNs = np.load("X_NAN_LIST.npy")
xTime = np.load("X_TIME_LIST.npy")

################################################################################################################################################################

################################
# Functions
################################

def Every_Nth_Value(y,nth=40):
    return (y[::nth])

################################

def Every_Nth_ValueXY(x,y,n=40):
    return (Every_Nth_Value(x,nth=n), Every_Nth_Value(y,nth=n))

################################

def GetNumDays(time=xTime):
    
    #xTime = np.load("X_TIME_LIST.npy")
    nDays = time[-1]-time[0]
    
    return (nDays)

################################

def FilterMyData(x,cutoff=0.00005,order=2):
    
    """
    Function to apply a Butter Filter to timeseries.
    Vars:
    
    y:        The timeseries. Must be list or np array.
    cutoff:   The cutoff frequency. Used to determine where the filter cut off is.
    order:    Approximation via polynomial of the order'th degree (2=quadratic, 3=cubic, 4=quartic, etc)
    """
    
    # DATA VALIDATION
    
    # Flag
#    isNested = False
    
    # Check to see if x is a nested dataframe or not
##    if type(x) == pd.core.frame.DataFrame:
##        isNested = True
##        #print("NESTED DATAFRAME FOUND! UNPACKING FOR CALCULATIONS, THE REPACKING...")
##        x = from_nested_to_2d_array(x)
    
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
    
#    if isNested == True:
#        newX = from_2d_array_to_nested(newX)
    
    # Finally, return the new X and Y values
    return (newX)

################################

def Normal(X,fixnan=True):
    # First of all, decide if wan to Fix all the 0s / NaNs
    if fixnan:
        X = FIXNAN(X)

    median = np.median(X)

    #print(f"OldNormal median = {median}")

    X[:] = [(number/median) for number in X]
    return X

################################

def FIXNAN(y, nanList=xNaNs):
    yMedian = np.median(y)
    y = [yMedian if n in nanList else item for n,item in enumerate(y)]
    return y

################################

def GetMetrics(classifierType, Xtrain, Xtest, ytrain, ytest, param_grid):
    
    # Make a PCA Pipeline
    print("> GM: START")
    
    algorithm = classifierType()
    cname     = classifierType.__name__
    print(f"\t> Model: {cname}")
    
    print("> GM: GENERATING TRANSFORMERS")
    flt = FunctionTransformer(FilterMyData)
    nth = FunctionTransformer(Every_Nth_Value)
    
    print("> GM: MAKE PIPELINE")
    #model = make_pipeline(flt,nth,algorithm)
    pipe = Pipeline(steps=[['filter',flt],['everynth',nth],['algorithm',algorithm]])
    
    # Do gridsearch for svc params
    print("> GM: GRIDSEARCH")
    grid = GridSearchCV(pipe, param_grid)
    
    # Fit model
    print("> GM: FIT")
    grid.fit(Xtrain, ytrain)
    
    # Use svc params and predict
    print("> GM: MAKESTATS")
    moreStats = grid.cv_results_
    #print("> > Best parameter (CV score=%0.3f):" % grid.best_score_)
    #print("> > {}".format(grid.best_params_))
    
    # Use svc params and predict
    print("> GM: PREDICT")
    model = grid.best_estimator_
    yfit = model.predict(Xtest)
    
    # Get Acc/Pre/Res
    print("> GM: CALCULATING RELIABILITY METRICS")
    mAcc = accuracy_score(ytest, yfit)
    mPre = precision_score(ytest, yfit)
    mRec = recall_score(ytest, yfit)
    metrics = (mAcc, mPre, mRec)
    
    # Now that model has done, time for confusion matrix shenanigans
    print("> GM: CONFUSION")
    mat = confusion_matrix(ytest, yfit)
    
    return (mat, moreStats, metrics)

################################

def WriteJSON(targetname, tStart, tFin, tDelta, TN, FP, FN, TP, acc, pre, rec, stats):
    # Preparing the stats text
#    data = {}
#    data[targetname] = []
#    data[targetname].append({
#        'Accuracy' : acc,
#        'Precision' : pre,
#        'Recall' : rec,
#        'CV Stats': stats
#    })
    
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

def MergeDicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res

################################################################################################################################################################

################################
# main
################################

def main():
    
    ############################
    # Data Initialisers
    ############################
    masterX = [x[1:-1] for x in np.load("None_Or_One_Exoplanet.npy")]
    masterY = np.load("None_Or_One_isplanetlist.npy")
    
    masterX = masterX[::5]
    masterY = masterY[::5]
    
    print(f"Length of x-arr: {len(masterX)}\nLength of y-arr: {len(masterY)}")

    X_nested = from_2d_array_to_nested(np.array(masterX))
    Xtrain, Xtest, ytrain, ytest  = train_test_split(X_nested, masterY, random_state=42)
    #Xtrain_,Xtest_,ytrain_,ytest_ = train_test_split (masterX, masterY, random_state=42)
    
    ############################
    # Classifier List Setup
    ############################
    
    #list_of_classifiers = [x.split('_')[2] for x in glob.glob('./sktime_results/*.json')]
    #list_of_classifiers.sort()
    
    list_of_classifiers = [
        Arsenal,
        CanonicalIntervalForest,
        ContractableBOSS,
        DrCIF,
        IndividualBOSS,
        IndividualTDE,
        MUSE,
        MatrixProfileClassifier,
        RandomIntervalSpectralForest,
        ShapeDTW,
        SignatureClassifier,
        SupervisedTimeSeriesForest,
        TSFreshClassifier,
        WEASEL
    ]
    
    ############################
    # Parameter Grid Setup
    ############################
    
    custom_transformers_param_grid = dict(
                  filter__kw_args = 
                  [
                      #{'cutoff': list(np.linspace(0.00001,0.0018755128487341842))},
                      {'order': [1,2,3]},
                  ],
                  #everynth__kw_args = 
                  #[
                  #    {'nth': [10, 20, 30, 40, 50]}
                  #]
                  #'drcif__base_estimator': ['DTC', 'CIT'],
                  #'drcif__n_estimators':   np.linspace(100,1000,19)
                    Muse = {[]}
             )
    
                    #   n_features_per_kernel=4
    
    param_grid = {'algorithm__num_kernels':  [1000,1500,2000,2500,3000],
                  'algorithm__n_estimators': [15, 20, 25, 30, 35]}#, 0.005]}
    
    # WILL NEED TO MAKE A PARAM GRID FOR EACH ALGORITHM SEPARATELY, AND ADD IT HERE; MAYBE APPENDING IT? LOAD FROM JSON FILE?
    # EITHER WAY, IF I ADD ARSENAL KWARGS FOR, SAY, MUSE, IT DOESN'T WORK.
    # THEREFORE SHOULD MAYBE APPEND EACH ALGORITHM'S HYPERPARAMETERS AS PART OF THE LOOP BELOW?
        # > NEED TO WORK OUT HOW TO APPEND TO DICTS
        # I GOT A MERGEDICTS FUNCTION READY!!!
        
# EXAMPLE:
#
#    dict2 = dict(drcif__base_estimator = ['DTC', 'CIT'],
#             drcif__n_estimators   =   np.linspace(100,1000,19))
#
#    dict3 = Merge(custom_transformers_param_grid, dict2)
#    
    
    ############################
    # Loop Start
    ############################
    
    print("Staring Loops!\n####################\n")
    
    for classifier in list_of_classifiers:
        print(f"Model: {classifier}")
        
        # Start Timer
        tStart = datetime.now()
        
        # Make custom param_grid
        #param_grid = custom_transformers_param_grid ############## APPEND ALGO SPECIFIC STUFF HERE
        
        # Confusion Matric Stuff
        #Parallel(n_jobs=10)(delayed(GetMetrics)(c, Xtrain, Xtest, ytrain, ytest, param_grid) for c in [classifier])
        confMat, moreStats, acc, pre, res = GetMetrics(classifier, Xtrain, Xtest, ytrain, ytest, param_grid)
        (TN, FN), (FP, TP) = confMat.T
        
        # End Timer and get time stats
        tFin = datetime.now()
        tDelta = tFin - tStart
        mins = (math.floor(tDelta.seconds/60))
        
        WriteJSON(classifier, tStart, tFin, tDelta, TN, FP, FN, TP, acc, pre, rec, stats)


################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

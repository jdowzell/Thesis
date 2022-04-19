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
from sklearn.cluster import KMeans

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
    
    print("> GPR-TRANSFORMERS")
    #nrm = FunctionTransformer(NormaliseFlux)
    #flt = FunctionTransformer(FilterMyData)
    #nth = FunctionTransformer(Every_Nth_Value)
    #svc = SVC(kernel='rbf', class_weight='balanced')
    
    #print(f"##############################\n{flt.get_params().keys()}\n##############################")
    
    print("> GPR-MAKE-PIPELINE")
    #model = make_pipeline(nrm,flt,svc)
    #model = make_pipeline(flt,svc)
#    model = Pipeline(steps=[
#         ('nrm', FunctionTransformer(NormaliseFlux)),
#         ('flt', FunctionTransformer(FilterMyData)),
#         ('nth', FunctionTransformer(Every_Nth_Value)),
#         ('svc', SVC(kernel='rbf', class_weight='balanced'))
#    ])
    model = KMeans(n_clusters=2, random_state=42, init='k-means++', max_iter=5000, n_init=50)
    
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
# main
################################

def main():
    
    # Initial setups; to allow for future master-file-ification later on
    targetname = 'KMeans'
    # Later on will be taken from a cmd line arg / a cmd line option (" > get_metrics.py --fourier true --algorithm SVM --nth_record 10")
    
    print("##########\nTIME START")
    tStart = datetime.now()
    print (tStart)
    
    print("##########\nLOADING FILES")

    # Load the Data files
    #fluxarrFULL     = np.load("fluxlist.npy")
    fluxarrFULL     = np.load("filteredfluxlistONEORNONE.npy")
    #isplanetarrFULL = ["Planet" if x==True else "Not Planet" for x in np.load("isplanetlist.npy")]
    #isplanetarrFULL = ["Planet" if x==1 else "Not Planet" for x in np.load("one_or_none_isplanetlist.npy")]
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
    
    param_grid = {'algorithm': ['auto', 'full', 'elkan']}
    
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
    
    # Reliability Metrics
    mAcc = accuracy_score(y_test, y_pred)
    mPre = precision_score(y_test, y_pred)
    mRec = recall_score(y_test, y_pred)
    
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
        'Accuracy': mAcc,
        'Precision': mPre,
        'Recall': mRec,
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
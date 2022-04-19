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

from IPython.display import display

################################
# MatPlotLib Settings
################################
#plt.rcParams["figure.figsize"] = (20,9)
sb.set()

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
    
    #I use this to run self-tests on the difference between Base TS and Fourier Transformed Series
    # Later on will be taken from a command line argument, or even an option (" > get_metrics.py --fourier true --algorithm SVM --nth_record 10")
    tryFourier = False
    tryFolded  = False
    tryNormal  = False
    
    if   len(sys.argv) > 1 and sys.argv[1].lower() in ['true', 't', 'tru', 'fourier']:
        tryFourier = True
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ['fold', 'folded', 'fld']:
        tryFolded = True
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ['normalised', 'normalized', 'normal', 'norm', 'nrm']:
        tryNormal = True
        
    print("FOURIER? {}".format(tryFourier))
    print("FOLDED? {}".format(tryFolded))
    print("NORMAL? {}".format(tryNormal))

    # Load the Data files
    if tryFourier == True:
        fluxarrFULL = np.load("fourierfluxlist.npy")
        targetname = targetname+"_FFT"
    elif tryFolded == True:
        #fluxarrFULL = np.load("foldedfluxNONAN.npy")                                    ### <<< Full Fluxlist, with no Nans
        #fluxarrFULL = np.load("ZeroOrOneFoldedFluxes.npy", allow_pickle=True)           ### <<< I think Obsolete now? Keeping for insurance
        #fluxarrFULL = np.load("ZeroOrOneFoldedFluxes_TEST.npy",allow_pickle=True)       ### <<< Zero or One exoplanet LCs, binned, and NaNs removed
        fluxarrFULL = np.load("ZeroOrOneFoldedFluxes_4717bins.npy")
        targetname = targetname+"_FOLDED"
    elif tryNormal == True:
        fluxarrFULL = np.load("positive_None_Or_One_Exoplanet_NORMALISED.npy")
        targetname = targetname+"_NORM"
    else:
        fluxarrFULL = np.load("fluxlist.npy")
    
    
    #isplanetarrFULL = ["Planet" if x==True else "Not Planet" for x in np.load("isplanetlist.npy")]
    isplanetarrFULL = ["Planet" if x==1 else "Not Planet" for x in np.load("one_or_none_isplanetlist.npy")]
    
    
    #isplanetarrFULL = [1 if x==True else 0 for x in np.load("isplanetlist.npy")]
    
    print("##########\nUSING EVERY NTH RECORD")
    
    # If I want to take every "n"th record, now I just have to flip a switch instead of commenting-and-uncommenting shenanigans
    every_nth_record = False
    
    # I can also change the number if I want to here
    every_nth_record_num = 10
    # Maybe make a CMD line argument?
    # Future work perhaps
    
    
    # Initialise appropriate data arrays
    if every_nth_record == True:
        print("> Using every {}th record".format(every_nth_record_num))
        fluxarr, _, isplanetarr, _ = train_test_split(fluxarrFULL, isplanetarrFULL, random_state=42, train_size=1/every_nth_record_num)
    else:
        print("> False")
        fluxarr = fluxarrFULL
        isplanetarr = isplanetarrFULL
    
    print("##########\nGENERATING PARAM GRID")
    
    #param_grid = {'C': [1, 5, 10, 50],
    #              'gamma': [0.0001, 0.0005, 0.001, 0.005]}
    
    #param_grid = {'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]}
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
    fname = targetname+".json"
    
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
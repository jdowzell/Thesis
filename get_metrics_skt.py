#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np

from astroquery.mast import Observations
from astroquery.mast import Catalogs

from astropy.table import Table
from astropy.table import QTable
from astropy.io import fits
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries
from astropy.stats import sigma_clipped_stats
from astropy.timeseries import aggregate_downsample

import sktime as skt

from sktime.datatypes._panel._convert import (
    from_2d_array_to_nested,
    from_nested_to_2d_array,
    is_nested_dataframe,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

################################
# Initialisers
################################
skt_model_folder = "./sktime_models/"
skt_results_folder = "./sktime_results/"

################################
# FUNCTIONS
################################

def MakingAList(prnt=False):
    jl = []
    sktfolder = skt_model_folder
    dirs = [x[0] for x in os.walk(sktfolder, topdown=True)]

    for sktfolder, dirs, files in os.walk(sktfolder):
        for file in files:
            fullpath = os.path.join(sktfolder, file)
            if (os.path.splitext(fullpath.lower())[1]).endswith('.joblib'):
                jl.append(fullpath)
    if prnt==True:
        print("Number of JOBLIB files: {}".format(len(jl)))
    #print(len(fl))
    return jl

################################

def WriteJSON(targetname, acc, pre, rec, stats):
    # Preparing the stats text
    data = {}
    data[targetname] = []
    data[targetname].append({
        'Accuracy' : acc,
        'Precision' : pre,
        'Recall' : rec,
        'Stats': stats
    })

    # File saving stuff
    targetdest = skt_model_folder + "sktime_"
    fname = targetname+".json"

    print("Saving {}".format(fname))

    # Write all the info to a file
    with open(targetdest+fname, "w") as f:
        #f.write(stats)
        json.dump(data, f, indent=4, default=str)

################################

def GetSKTScores(model, X_train, y_train):
    
    classifier = joblib.load(model)
    
    classifier.fit(X_train, y_train)
    
    # MORE STATS
    moreStats = classifier.cv_results_
    
    model = classifier.best_estimator_
    yfit = model.predict(Xtest)
    
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"Accuracy: {acc}\nPrecision: {pre}\nRecall: {rec}")
    
    return (acc,pre,rec,moreStats)

################################

sktmodels = MakingAList()
skt_modelnames = [x.split(skt_model_folder)[1].split('.joblib')[0] for x in sktmodels]

X = np.load("None_Or_One_Exoplanet_FILT_NORM_SUBS.npy")
y = np.load("one_or_none_isplanetlist.npy")

X_nested = from_2d_array_to_nested(X).copy()
X_train, X_test, y_train, y_test = train_test_split(X_nested, y)

################################

for i in range(len(sktmodels)):
    
    model = skt_modelnames[i]
    print(f"FITTING MODEL {skt_modelnames[i]}")
    
    acc,pre,rec,stats = GetSKTScores(sktmodels[i], X_train, y_train)
    WriteJSON(model,acc,pre,rec,stats)
    
    print("DONE!")

#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np

import sktime

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

def WriteJSON(targetname, acc, pre, rec):
    # Preparing the stats text
    data = {}
    data[targetname] = []
    data[targetname].append({
        'Accuracy' : acc,
        'Precision' : pre,
        'Recall' : rec
    })

    # File saving stuff
    targetdest = skt_results_folder + "sktime_"
    fname = targetname+".json"

    print("Saving {}".format(fname))

    # Write all the info to a file
    with open(targetdest+fname, "w") as f:
        #f.write(stats)
        json.dump(data, f, indent=4, default=str)

################################

def GetSKTScores(model, modelname, X_train, y_train):
    
    classifier = joblib.load(model)
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"Model: {modelname}\n\tAccuracy: {acc}\n\tPrecision: {pre}\n\tRecall: {rec}")
    
    WriteJSON(modelname,acc,pre,rec)
    
    return (acc,pre,rec)

################################

sktmodels = MakingAList()
skt_modelnames = [x.split(skt_model_folder)[1].split('.joblib')[0] for x in sktmodels]

skt_list = list(zip(sktmodels,skt_modelnames))

X = np.load("None_Or_One_Exoplanet_FILT_NORM_SUBS.npy")
y = np.load("one_or_none_isplanetlist.npy")

X_nested = from_2d_array_to_nested(X).copy()
X_train, X_test, y_train, y_test = train_test_split(X_nested, y)

del X, y, X_nested

print("READY TO PARALLELISE")

################################

print(skt_list)

Parallel(n_jobs=10)(delayed(GetSKTScores)(c[0],c[1],X_test,y_test) for c in skt_list)

print("DONE")

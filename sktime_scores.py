#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np

from sktime.datatypes._panel._convert import (
    from_2d_array_to_nested,
    from_nested_to_2d_array,
    is_nested_dataframe,
)

#from sktime.classification.compose import ColumnEnsembleClassifier                 <---- MULTIVARIATE SO NOT NEEDED
#from sktime.classification.dictionary_based import BOSSEnsemble                    <---- MULTIVARIATE SO NOT NEEDED
from sktime.classification.dictionary_based import IndividualBOSS
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import MUSE
from sktime.classification.dictionary_based import IndividualTDE
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.distance_based import ProximityForest
from sktime.classification.distance_based import ProximityStump
from sktime.classification.distance_based import ProximityTree
from sktime.classification.distance_based import ShapeDTW
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
#from sktime.classification.feature_based import Catch22Classifier                  <---- MULTIVARIATE SO NOT NEEDED
from sktime.classification.feature_based import MatrixProfileClassifier
from sktime.classification.feature_based import SignatureClassifier
from sktime.classification.feature_based import TSFreshClassifier
#from sktime.classification.hybrid import Catch22ForestClassifier                   <---- For some reason this one doesn't work? Or, rather, gives me an error
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.interval_based import DrCIF
from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import SupervisedTimeSeriesForest
#from sktime.classification.interval_based import TimeSeriesForestClassifier        <---- MULTIVARIATE SO NOT NEEDED
from sktime.classification.kernel_based import Arsenal
from sktime.classification.kernel_based import ROCKETClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
#from sktime.classification.shapelet_based import MrSEQLClassifier                  <---- MULTIVARIATE SO NOT NEEDED

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

################################
# General imports
################################
import csv, math, io, os, os.path, sys, random, time, json, gc
import pandas as pd
import argparse
from datetime import datetime

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

################################
# Initialisers
################################

# Load the Data files
#fitsarr = np.load("fitslist.npy")

################################
# Functions
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
    targetdest="./confusionmatrices/sktime_"
    fname = targetname+".json"

    print("Saving {}".format(fname))

    # Write all the info to a file
    with open(targetdest+fname, "w") as f:
        #f.write(stats)
        json.dump(data, f, indent=4, default=str)
    
################################
# main
################################

def main():
    
    print("Loading X and Y...")
    #X = np.array([item[1:-1] for item in np.load("None_Or_One_Exoplanet.npy")])
    X = np.load("None_Or_One_Exoplanet_FILT_NORM_SUBS.npy")
    y = np.load("one_or_none_isplanetlist.npy")
    
    print("Splitting into Test and Train sets...")
    X_nested = from_2d_array_to_nested(X)
    X_train, X_test, y_train, y_test = train_test_split(X_nested, y)
    
    # List of all classifiers
    #
    #
    # SOME OF THESE MIGHT NOT BE NEEDED!
    # See: (https://www.sktime.org/en/stable/examples/03_classification_multivariate.html) for MULTIVARIATE ones - I have UNIVARIATE
    #
    print("Load Classifier list...")
    list_of_classifiers = [
        #ColumnEnsembleClassifier(estimators=20),   # < this is the only one where I believe you *need* a param; the others seem fine without it
        #BOSSEnsemble(),
        IndividualBOSS(),
#        ContractableBOSS(),
        MUSE()#,
#        IndividualTDE(),
#        TemporalDictionaryEnsemble(),
#        WEASEL(),
#        ElasticEnsemble(),
#        ProximityForest(),
#        ProximityStump(),
#        ProximityTree(),
#        ShapeDTW(),
#        KNeighborsTimeSeriesClassifier(),
        #Catch22Classifier(),
#        MatrixProfileClassifier(),
#        SignatureClassifier(),
#        TSFreshClassifier(),
        #Catch22ForestClassifier(),
#        HIVECOTEV1(),
#        HIVECOTEV2(),
#        CanonicalIntervalForest(),
#        DrCIF(),
#        RandomIntervalSpectralForest(),
#        SupervisedTimeSeriesForest(),
        #TimeSeriesForestClassifier(),
#        Arsenal(),
#        ROCKETClassifier(),
#        ShapeletTransformClassifier(),
        #MrSEQLClassifier()
    ]
    
    
    print("Staring Loops!\n####################\n")
    
    # DO A LOOP FOR ALL CLASSIFIERS HERE\
    for i in list_of_classifiers:
    
        tStart = datetime.now()

        classifier = i
        cname = type(i).__name__
        
        # Tell me what classifier is running currently
        print(f"Classifier: {cname}")
        
        print("\nFitting...")
        classifier.fit(X_train, y_train)
        
        print("Predicting...")
        y_pred = classifier.predict(X_test)
        
        # Calculate Metrics
        print("Calculating Metrics...")
        mAcc = accuracy_score(y_test, y_pred)
        mPre = precision_score(y_test, y_pred)
        mRec = recall_score(y_test, y_pred)

        # SAVE TO JSON INSTEAD OF PRINT
        print(f" > Accuracy: {mAcc}\n > Precision: {mPre}\n > Recall: {mRec}")
        WriteJSON(cname, mAcc, mPre, mRec)
        
        tFin = datetime.now()
        tDelta = tFin - tStart
        mins = (math.floor(tDelta.seconds/60))
        print(f"Process took {mins} minutes and {tDelta.seconds - (60*mins)} seconds")
        
        # Garbage Collection to save memory
        del classifier, y_pred, mAcc, mPre, mRec
        gc.collect()
        print("Garbage Collected!\n")

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

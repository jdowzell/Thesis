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
from sktime.classification.feature_based import MatrixProfileClassifier
from sktime.classification.feature_based import SignatureClassifier
from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.interval_based import DrCIF
from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.classification.kernel_based import Arsenal
from sktime.classification.kernel_based import ROCKETClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

################################
# General imports
################################
from joblib import Parallel, delayed, dump, load

################################
# Suppress Warnings
################################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

################################
# Initialisers
################################

#os.system("taskset -p 0xfff %d" % os.getpid())

# Load the Data files
#fitsarr = np.load("fitslist.npy")

################################
# Functions
################################

def MakeModels(classifierType, rows=-1):
    
    X = np.array([item[1:-1] for item in np.load("None_Or_One_Exoplanet_NORMALISED.npy")])
    y = np.load("one_or_none_isplanetlist.npy")
    
    if rows > 0:
        X = X[:rows, :]
        y = y[:rows]
    
    print("Splitting into Test and Train sets...")
    X_nested = from_2d_array_to_nested(X)
    X_train, X_test, y_train, y_test = train_test_split(X_nested, y)
    
    # Selecting Classifier Type
    c = classifierType()
    cname = classifierType.__name__
    
    # Fitting Classifier
    print(f"Fitting the {cname} model...")
    c.fit(X_train, y_train)
    
    # Saving Model
    print(f"Saving {cname} model...")
    if rows > 0:
        dump(c, f'./sktime_models/_{cname}_fitted_{rows}.joblib')
    else:
        dump(c, f'./sktime_models/_{cname}_fitted_FULL.joblib')
        
    print("Done!")

def MakeModelsNEW(classifierType, rows=None):
    
    print("Loading the Test and Train Datasets...")
    
    y = np.load("one_or_none_isplanetlist.npy")[:rows]
    X_nested = from_2d_array_to_nested(np.array([item[1:-1] for item in np.load("None_Or_One_Exoplanet_NORMALISED.npy")])[:rows, :])
    X_train, X_test, y_train, y_test = train_test_split(X_nested, y)
    
    # Selecting Classifier Type
    c = classifierType()
    cname = classifierType.__name__
    
    # Fitting Classifier
    print(f"Fitting the {cname} model...")
    c.fit(X_train, y_train)
    
    # Saving Model
    print(f"Saving {cname} model...")
    
    if rows==None:
        rows = "FULL"
    
    dump(c, f'./sktime_models/_{cname}_fitted_{rows}.joblib')
        
    print("Done!")
    
################################
# main
################################

def main():
    
    list_of_classifiers = [
        IndividualBOSS,                 # 0
        ContractableBOSS,               # 1
        MUSE,                           # 2
        IndividualTDE,                  # 3
        TemporalDictionaryEnsemble,     # 4
        WEASEL,                         # 5
        ElasticEnsemble,                # 6
        ProximityForest,                # 7
        ProximityStump,                 # 8
        ProximityTree,                  # 9
        ShapeDTW,                       # 10
        KNeighborsTimeSeriesClassifier, # 11
        MatrixProfileClassifier,        # 12
        SignatureClassifier,            # 13
        TSFreshClassifier,              # 14
        HIVECOTEV1,                     # 15
        HIVECOTEV2,                     # 16
        CanonicalIntervalForest,        # 17
        DrCIF,                          # 18
        RandomIntervalSpectralForest,   # 19
        SupervisedTimeSeriesForest,     # 20
        Arsenal,                        # 21
        ROCKETClassifier,               # 22
        ShapeletTransformClassifier     # 23
    ]
    
    classifier = list_of_classifiers[12]
    
    #X = np.array([item[1:-1] for item in np.load("None_Or_One_Exoplanet_NORMALISED.npy")])
    #y = np.load("one_or_none_isplanetlist.npy")
    
    print("Staring Loops!\n####################\n")
    
    #Parallel(n_jobs=None)(delayed(MakeModelsNEW)(c) for c in [classifier])
    MakeModels(classifier)
    
#    for i in list_of_classifiers:
        
#        print(f"Now running model: {i.__name__}")
        
#        # Run Parallel on the models, ONE AT A TIME
    
#        Parallel(n_jobs=2, prefer="threads")(delayed(MakeModelsNEW)(c) for c in [i])
#        #MakeModels(i)

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

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
import argparse

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
from sklearn.cluster import SpectralClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from IPython.display import display

################################
# MatPlotLib Settings
################################
#plt.rcParams["figure.figsize"] = (20,9)
sb.set()

################################
# Functions
################################


################################
# main
################################

def main():
    ArgTest()

def ArgTest():
    
    # Set up command line arguments via argparse
    welcome = "###\nRuns Machine Learning algorithms to read through timeseries datasets and record results\n###"
    
    parser = argparse.ArgumentParser(description=welcome)
    
    parser.add_argument('--algorithm', '-a', required=True,
                        help="Which Machine Learning algorithm you want to use. Options are: NaiveBayes (NB); RandomTree (RT); SupportVectorMachine (SVM); KMeans (KM)")
    parser.add_argument('--every_nth_record', '-n', type=int, default=0,
                        help="If used, will only use every 'n'th record instead of the full list. If not used, will default to using all records.")
    parser.add_argument('--dataset', '-d', default="default",
                        help="Determines whether to use the normal dataset, the Fourier transformed dataset (-d fourier) of the folded dataset (-d fold)")
    parser.parse_args()

    args = parser.parse_args()
    
    # Arguments
    algo = args.algorithm.upper()  if args.algorithm.lower() in ['nb', 'naivebayes', 'rt', 'randomtree', 'svm', 'supportvectormachine', 'kmeans', 'km'] else "ERROR"
    nth  = args.every_nth_record   if args.every_nth_record > 0 else "ALL"
    dataset = args.dataset.upper() if args.dataset.lower()   in ['default', 'fold', 'folded', 'fourier', 'ftt'] else "ERROR"
    
    # Argument Checking / Error Test
    if algo == "ERROR" or dataset == "ERROR":
        if algo == "ERROR":
            print("Please select a valid algorithm.\nOptions are: NaiveBayes (NB); RandomTree (RT); Support Vector Machine (SVM); KMeans (KM)")
        if dataset=="ERROR":
            print("Please select a valid dataset.\nOptions are: [Fold OR Folded]; [Fourier OR FFT]")
        sys.exit()
    
    # Print out for checking sake
    print(f"{'Algorithm: '.ljust(20) + algo}\n{'Every Nth Record: '.ljust(20) + str(nth)}\n{'Dataset: '.ljust(20) + dataset}")
    
    print(f"./get_metrics_"+algo+".py "+dataset)
    
    #                    #
    # CANNOT USE DATASET #
    # AND FOUR / FOLD AT #
    #   THE SAME TIMES   #
    #                    #
    
    # Return all vals
    return(args)
    
def TEST():
    
    # Initial setups; to allow for future master-file-ification later on
    targetname = 'NaiveBayes'
    # Later on will be taken from a command line argument, or even an option (" > get_metrics.py --fourier true --algorithm SVM --nth_record 10")
    
    print("##########\nTIME START")
    tStart = datetime.now()
    print (tStart)
    
    print("##########\nLOADING FILES")
    
    #I use this to run self-tests on the difference between Base TS and Fourier Transformed Series
    # Later on will be taken from a command line argument, or even an option (" > get_metrics.py --fourier true --algorithm SVM --nth_record 10")
    tryFourier = False
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['true', 't', 'tru', 'fourier']:
        tryFourier = True
        
    print("FOURIER? {}".format(tryFourier))

    # Load the Data files
    if tryFourier == True:
        fluxarrFULL = np.load("fourierfluxlist.npy")
        newtargetname = targetname+"_FFT"
        targetname = newtargetname
    else:
        fluxarrFULL = np.load("fluxlist.npy")
    
    #fitsarr = np.load("fitslist.npy")
    #fluxarrFULL = np.load("abridgedfluxlist.npy")
    #fluxarrFULL = np.load("abridgedfourierfluxlist.npy")
    #planetarrFULL = np.load("planetlist.npy")
    isplanetarrFULL = [1 if x==True else 0 for x in np.load("isplanetlist.npy")]
    
    # If I want to take every "n"th record, now I just have to flip a switch instead of commenting-and-uncommenting shenanigans
    #
    #
    #
    every_nth_record = False
    #
    #
    #
    
    # I can also change the number if I want to here
    every_nth_record_num = 10
    # Maybe make a CMD line argument?
    # Future work perhaps
    
    
    # Initialise appropriate data arrays
    if every_nth_record == True:
        print("##########\nUSING EVERY NTH RECORD")
        print("> Using every {}th record".format(every_nth_record_num))
        fluxarr, _, isplanetarr, _ = train_test_split(fluxarrFULL, isplanetarrFULL, random_state=42, train_size=1/every_nth_record_num)
    else:
        fluxarr = fluxarrFULL
        isplanetarr = isplanetarrFULL
    
    print("##########\nGENERATING PARAM GRID")
    
    #param_grid = {'C': [1, 5, 10, 50],
    #              'gamma': [0.0001, 0.0005, 0.001, 0.005]}
    
    alphaList=list(range(1,100))
    alphaList[:] = [x / 100 for x in list(range(1,100))]
    
    param_grid = {'alpha': alphaList}
    
    print("##########\nRUNNING GETPOSITIVERATES")
    
    confMat = GetPositiveRates(fluxarr, isplanetarr, param_grid) #, ncomp=pca_n_components)
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
        'dateran': tStart.replace(microsecond=0)
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
    
    
    
    
    
    
def GetLC(rndFile=-1, fullOutput=False):
    
    # rndFile is random, unless specified
    rndFile = random.randint(0,len(fitsarr)) if rndFile==-1 else rndFile
    #print("Curve = {}".format(rndFile))
    
    # Get LC data from the requisite fits file
    fitsFile = fitsarr[rndFile]

    # The following line of code gives us the header values
    fitsHeaders = fits.getheader(fitsFile)

    with fits.open(fitsFile, mode="readonly") as hdulist:

        # Extract stellar parameters from the primary header.
        obj       = hdulist[0].header['OBJECT']
        #sector    = hdulist[0].header['SECTOR']

        # Extract some of the columns of interest for the first TCE signal.  These are stored in the binary FITS table
        tess_bjds     = hdulist[1].data['TIME']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

    # X- and Y- labels
    ylab = "PDCSAP Flux (e-/s)"
    xlab = "Time (TBJD)"

    if (fullOutput == True):
        return (tess_bjds, pdcsap_fluxes, str(rndFile), str(obj), ylab, xlab)
    else:
        return (tess_bjds, pdcsap_fluxes)

    return

    ts = TimeSeries.read(fitsarr[int(f)], format='tess.fits')  
    periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')
    results = periodogram.autopower(0.2 * u.day)  
    best = np.argmax(results.power)  
    period = results.period[best]
    transit_time = results.transit_time[best]
    ts_folded = ts.fold(period=period, epoch_time=transit_time)
    foldedflux = list(ts_folded['sap_flux'].value)
    
    # UN-FOLD THE X AXIS SOMEHOW
    unfoldedtime = [0] * len(ts_folded.time.jd)

    for i in range(len(ts_folded.time.jd)):
        unfoldedtime[i] = [ts_folded.time.jd[i],i]
    
    G = []
    Q = list(ts_folded.time.jd.copy())

    for i,e in enumerate(Q):
        G.append({'index': i, 'time': e, 'flux': foldedflux[i]})
    
    newlist = sorted(G, key=lambda k: k['time'])
    
    FLATFLUX = [d['flux'] for d in newlist]
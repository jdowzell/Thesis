################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np
import csv, math, io, os, os.path, sys, random, time
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

################################
# General Settings
################################
%matplotlib inline
from IPython.display import display

################################
# MatPlotLib Settings
################################
plt.rcParams["figure.figsize"] = (20,9)
sb.set()

################################
# Load the data from the previously saved NP files
################################

itsarr = np.load("fitslist.npy")
fluxarr = np.load("fluxlist.npy")
planetarr = np.load("planetlist.npy")
isplanetarr = np.load("isplanetlist.npy")

################################
# Split Data into Test and Train
################################
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(fluxarr, isplanetarr, random_state=42)
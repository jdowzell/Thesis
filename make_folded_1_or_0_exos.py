#!/usr/bin/env python3

################################
# Scientific imports
################################
import matplotlib.pyplot as plt
import numpy as np

from astroquery.mast import Observations
from astroquery.mast import Catalogs

from astropy.io import fits
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import TimeSeries
from astropy.stats import sigma_clipped_stats
from astropy.timeseries import aggregate_downsample

import lightkurve

################################
# General imports
################################
import csv, math, io, os, os.path, sys, random, time, json, gc
from datetime import datetime
import pandas as pd
import seaborn as sb
from collections import Counter

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
fitsarr = np.load("fitslist.npy")
C = 0

################################
# Functions
################################

def LoadList():
    
    planet_list ="tsop301_planet_data.txt"
    star_list   ="tsop301_star_data.txt"
    eb_list     ="tsop301_eb_data.txt"
    beb_list    ="tsop301_backeb_data.txt"
    
    starlist = LoadListGeneral(star_list)
    planetlist = LoadListGeneral(planet_list)
    eblist = LoadListGeneral(eb_list)
    beblist = LoadListGeneral(beb_list)
    
    alllists = {"s": starlist, "p": planetlist, "eb": eblist, "beb": beblist}
    
    return alllists

def LoadListGeneral(f):
    
    lst=[]
    try:
        # Assuming everything CAN go well, do this
        with open('./SIM_DATA/unpacked/{}'.format(f)) as df:
            csvdf = csv.reader(df)
            for lineholder in csvdf:
                line = lineholder[0]                # 'lineholder' is a list, 1 element long, containing only a single string
                if line[0]!="#":                    # Ignore commented lines (lines w/ FIRST STRING ELEMENT is a # character)
                    lst.append(line.split()[0])     # Add line to list
                # endif
            # endfor
        # endwith
    except FileNotFoundError:
        print("FNF")
        return
    # end try
    
    return lst


# Python3 implementation to find elements
# that appeared only once
# Function to find the elements that
# appeared only once in the array
def OccurredOnce(in_arr, out_arr, count=1):
    
    n = len(in_arr)
    #counting frequency of every element using Counter
    mp=Counter(in_arr)
    # Traverse the map and print all
    # the elements with occurrence 1
    for it in mp:
        if mp[it] == count:
            out_arr.append(int(it))
            #print(it, end = " ")
    return

def GetTICListOfOnlyZeroOrOneExoplanets(fname):
    
    master_list = LoadList()
    starList, planetList, ebList, bebList = master_list['s'], master_list['p'], master_list['eb'], master_list['beb']
    starList   = [int(x) for x in starList]
    planetList = [int(x) for x in planetList]
    ebList     = [int(x) for x in ebList]
    bebList    = [int(x) for x in bebList]
    
    TICList=list(np.load("TICList.npy"))
    print("TICList is {} entries long".format(len(TICList)))
    
    #####################
    
    one_planet_lcs = []
    lst_a = one_planet_lcs
    
    OccurredOnce(planetList, one_planet_lcs)
    
    #####################
    
    fLen = len(fluxarr)
#    print("The file '{}' contains a list of {} LCs!".format(fname,fLen))
    
    zero_or_one_exoplanets = list(set(starList + one_planet_lcs))
    zero_or_one_exoplanets.sort()
    
    flist = [TICList.index(zero_or_one_exoplanets[x]) for x in range(len(zero_or_one_exoplanets))]
    print("FList has {} entries".format(len(flist)))
    
    # 'flist' now contains a list of file indices of the FITS files that contain Zero or One exoplanet.
    
    fname = fname+"_TEST"
    
    np.save(fname,flist)


def MakeTimeSeries(L, f, binSize=0.2, timedBinSize=0.003, numbins=None):
    
    global C
    C = C+1
    print("Processing File {} of {}...".format(C,L))
    
    ts = TimeSeries.read(fitsarr[int(f)], format='tess.fits')
    periodogram = BoxLeastSquares.from_timeseries(ts, 'sap_flux')
    results = periodogram.autopower(binSize * u.day)  
    best = np.argmax(results.power)  
    period = results.period[best]
    transit_time = results.transit_time[best]
    ts_folded = ts.fold(period=period, epoch_time=transit_time)
    mean, median, stddev = sigma_clipped_stats(ts_folded['sap_flux'])
    ts_folded['sap_flux_norm'] = ts_folded['sap_flux'] / median
    ts_binned = aggregate_downsample(ts_folded, time_bin_size=timedBinSize * u.day, n_bins=numbins)
    
    # Delete unnecessary vars
    del ts, periodogram, period, results, best, transit_time, mean, median, stddev, ts_folded
    
    #return(ts_folded['sap_flux_norm'].value)
    return(ts_binned['sap_flux_norm'].value)
    
################################
# main
################################

def main():
    
    # Making sure we have only one cmd line parameter and it is the fluxlist that we want
    try:
        #print(sys.argv)
        fname = sys.argv[1]
        fluxarr = np.load(fname)
    except FileNotFoundError:
        print("This File Does Not Exist")
        return
        #break
    except IndexError:
        print("Please add exactly one numpy file to the command line!")
        return
        #break
    except:
        print("General Error")
        return
        #break
    print("Source File Loaded!")
    
    #####################
    
    npFileName = "ZeroOrOneExoplanetTICids.npy"
    try:
        ZeroOrOneTICList = np.load(npFileName)
    except FileNotFoundError:
        print("This File Does Not Exist; Making it now!")
        GetTICListOfOnlyZeroOrOneExoplanets(npFileName)
    print("File '{}' loaded".format(npFileName))
    
    #####################
    #     OVERRIDES     #
    #####################
    
    ZeroOrOneTICList = ZeroOrOneTICList[:10]
    
    # The max length on bins is 4717
    # Lets set them all to that - significantly less than the 20081 than before,
    # and this ensures all data is the same length
    nbins = 4717
    
    if nbins == None:
        suffix = ""
    else:
        suffix = f"_{nbins}bins"
    
    #####################
    
    lenTIC = len(ZeroOrOneTICList)
    
    foldedFluxes = [MakeTimeSeries(lenTIC,x, numbins=nbins) for x in ZeroOrOneTICList]
    
    # Converting the above to a DataFrame to save it
    #print("Length of foldedFlux[0] is {}".format(len(foldedFluxes[0])))
    #print("fFlux[0]:\n{}".format(foldedFluxes[0]))
    np.save(f"ZeroOrOneFoldedFluxes{suffix}.npy",foldedFluxes)
    print("YAY")

################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()

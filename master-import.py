################################
# Scientific imports
###
%matplotlib inline
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import Observations
from astroquery.mast import Catalogs

###
# General imports
###
import csv, math, io, os, os.path, sys, random, time, bisect
import pandas as pd
import seaborn as sb
from tqdm.notebook import tqdm, trange
import sklearn
from sklearn import metrics
from IPython.display import display

###
# MatPlotLib Settings
###

plt.rcParams["figure.figsize"] = (20,9)
sb.set()

###
# Global Variables
###
# Keep track of current LC and it's TIC identifier
lastRandom={
    "number": 0,
    "id": 0
}

################################
# Functions
###

def MakingAList(prnt=False):
    # Function for Reading which LC datafiles we have into a list
    fl = []
    fitsroot = "SIM_DATA/"
    fits_directories = [x[0] for x in os.walk('./SIM_DATA/.', topdown=True)]

    for fitsroot, fits_dirs, fits_files in os.walk(fitsroot):
        for fits_file in fits_files:
            fullpath = os.path.join(fitsroot, fits_file)
            if (os.path.splitext(fullpath.lower())[1]).endswith('.fits'):
                fl.append(fullpath)
    if prnt==True:
        print("Number of FITS files: {}".format(len(fl)))
    return fl

# Chooses a random number
def GetRandomLC(randict, n = None):
    global lastRandom
    #print("1: {}".format(n))
    if isinstance(n, int):
        if 0 <= n < len(fitsList):
            n = n
        else:
            n = random.randint(0,len(fitsList))
    else:
        n = random.randint(0,len(fitsList))
    
    randict["number"] = n
    randict["id"] = str(fitsList[n].split("-")[2].lstrip("0"))
    return n

def DrawACurve(randict, n = None):
    rndFile = GetRandomLC(randict) if n == None else GetRandomLC(n)
    fitsFile = fitsList[rndFile]
    
    # The following line of code gives us the header values
    fitsHeaders = fits.getheader(fitsFile)

    with fits.open(fitsFile, mode="readonly") as hdulist:

        # Extract stellar parameters from the primary header.  We'll get the effective temperature, surface gravity,
        # and TESS magnitude.
        star_teff = hdulist[0].header['TEFF']
        star_logg = hdulist[0].header['LOGG']
        star_tmag = hdulist[0].header['TESSMAG']
        obj = hdulist[0].header['OBJECT']
        sector = hdulist[0].header['SECTOR']

        # Extract some of the fit parameters for the first TCE.  These are stored in the FITS header of the first
        # extension.
        duration = (hdulist[1].header['LIVETIME'])

        # Extract some of the columns of interest for the first TCE signal.  These are stored in the binary FITS table
        # in the first extension.  We'll extract the timestamps in TBJD, phase, initial fluxes, and corresponding
        # model fluxes.
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']

    # Start figure and axis.
    fig, ax = plt.subplots()

    # Plot the timeseries in black circles.
    ## Using the [1:-1] identifier to cut off the leading and trailing zeroes
    ax.plot(tess_bjds[1:-1], pdcsap_fluxes[1:-1], 'k.', markersize=1)

    # Let's label the axes and define a title for the figure.
    fig.suptitle(CurrentLC())
    ax.set_ylabel("PDCSAP Flux (e-/s)")
    ax.set_xlabel("Time (TBJD)")

    # Adjust the left margin so the y-axis label shows up.
    plt.subplots_adjust(left=0.15)
    
    plt.show()



def LoadCSV(csvfile):
    return(pd.read_csv(csvfile,comment='#',header=None,skipinitialspace=True,sep = "\s+|\t+|\s+\t+|\t+\s+", engine='python')[[0]])

def ListNames():
    mainlist={
    'planet':"./SIM_DATA/unpacked/tsop301_planet_data.txt",
    'star':"./SIM_DATA/unpacked/tsop301_star_data.txt",
    'eb':"./SIM_DATA/unpacked/tsop301_eb_data.txt",
    'beb':"./SIM_DATA/unpacked/tsop301_backeb_data.txt"
    }
    return(mainlist)

def LoadList():
    L=ListNames()
    p=LoadCSV(L['planet'])
    s=LoadCSV(L['star'])
    eb=LoadCSV(L['eb'])
    beb=LoadCSV(L['beb'])
    return(p,s,eb,beb)

def PlanetLookup(lst,x):
    idx = bisect.bisect_left(lst,x)
    return (idx<len(lst) and lst[idx] == x)

def IsThisA(lst,x):
    L=ListNames()
    #lst = 'planet', 'star', 'eb', or 'beb'
    # x is the objectID to search for
    idx = bisect.bisect_left(len(L[lst]),x)
    return (idx<len(L[lst]) and L[lst][idx] == x)
        
def IsThisAStar(n):
    #return n in alllists["s"]
    IsThisA('star',n)
    
def IsThisAPlanet(n):
    #return n in alllists["p"]
    IsThisA('planet',n)

def IsThisAEB(n):
    #return n in alllists["eb"]
    IsThisA('eb',n)

def IsThisABEB(n):
    #return n in alllists["beb"]
    IsThisA('beb',n)
    
def DFToList(*args):
    print("Converting DataFrames to Lists")
    lists=()
    if len(args) > 0:
        for i in args:
            tmp = [x[0] for x in i.values.tolist()]
            tmp.sort()
            lists+=(tmp,)
        return(lists)
    return 0


# Function to tell you what an item is
def WhatIsMyLC(n):
    lbl = []
    lbl.append("Star") if IsThisAStar(n) else lbl
    lbl.append("Planet") if IsThisAPlanet(n) else lbl
    lbl.append("EB") if IsThisAEB(n) else lbl
    lbl.append("BRB") if IsThisABEB(n) else lbl
    
    return "UNKNOWN" if lbl==[] else lbl

# Purely for convenience
def CurrentLC():
    return ("File № {} - {}".format(lastRandom["number"], lastRandom["id"]))


def MakeDataFrame(fitsList):
    """
    Reads a list of FITS files to examine
    
    Firstly, it reads in a list of FITS files to open and examine (param=fitsList)
    Next, it generates three lists (id-,dat-,p-) and makes them all equal in length to the length of the fitsList.
    It then reads the object ID (stored in the filename) and the flux timeseries, and assigns it to the two lists (params=idlist,datlist)
    Finally, it runs the "IsThisAPlanet" function to determind if the objID is a planet, and then outputs that into the last list (param=plist)
    
    RETURNS:
    A thruple of all three lists
    """
    # Make empty lists and array
    rng=int(len(fitsList))
    ilist=[None]*rng
    plist=[None]*rng
    dataArr = np.zeros((rng,20340)) #optional = datatype

    # Loop thru every FITS file
    print("Opening Files",end='')
    for n, file in enumerate(tqdm(fitsList[0:rng])):
        # Print the file number (NOT ID, but the number of the file opened)
        if(n%1000==0):
            print("{},".format(n),end='')
        
        objid = np.uint32(str(fitsList[n].split("-")[2].lstrip("0")))
        ilist[n] = objid
        plist[n] = PlanetLookup(planetList,objid)
        
        # Open the file
        with fits.open(file) as hdu:
            # Get the PDSCAP flux data
            flux = hdu[1].data['PDCSAP_FLUX']
            dataArr[n] = flux
            
    print("\n")
    return(ilist,dataArr,plist)

################################
# RUN ALL INITIALISERS
###
def Initialise():
    # Set up the list of FITS files
    print("Populating fitsList...")
    fitsList = MakingAList()
    #WriteToFile("FITSLIST",fitsList)
    
    # Make the list of star/planet/eclipsingbinary/backeclipsingbinary IDs
    print("Loading the s/p/eb/beb Lists")
    p, s, eb, beb = LoadList()
    return(fitsList,p,s,eb,beb)

def MakeData(flist):
    # Make the lists of ID, Flux, IsPlanet
    print("Populating the DataFrame")
    idl, fl, pl = MakeDataFrame(flist)
    return (idl,fl,pl)
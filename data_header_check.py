def dataAndHeaderCheck(f, log=False):
    """
    A function to make sure that all headers required for plotting a Light Curve
    are present, and alert the user if any are missing (and which ones).
    
    Input parameter (f) should be a path to a FITS file (local, cloud, etc)
    Input parameter (log) displays the printouts IFF set to True
    
    Returns:
    1 - ALL headers are present; and
    0 - otherwise.
    """
    
    # Flags
    allStellar = True
    allFitPara = True
    allFitData = True
    
    # List of needed headers
    stellar_params = ['TEFF', 'LOGG', 'TESSMAG']
    
    fit_params = ['TPERIOD', 'TDUR', 'TEPOCH', 'TDEPTH']
    
    fit_data = ['TIME', 'PHASE', 'LC_INIT', 'MODEL_INIT']
    
    # FITS Headers
    fh0 = fits.getheader(f, ext=0)
    fh1 = fits.getheader(f, ext=1)
    
    # FITS Columns
    fc = fits.getdata(f).columns
    
    # Loop through all headers and see if they are present using a Try/Except block
    if(log):
        print("Testing to see if all relevant information is present...")
    
    # First, the Stellar Parameters block
    for i in range (len(stellar_params)):
        try:
            fh0[stellar_params[i]]
        except:
            if(log):
                print("\tHeader {} not present!".format(stellar_params[i]))
            allStellar = False
    if(allStellar & log):
        print("\tAll Stellar Parameters present")
    
    # Next, the Fit Parameters block
    for i in range (len(fit_params)):
        try:
            fh1[fit_params[i]]
        except:
            if(log):
                print("\tFit Parameter {} not present!".format(fit_params[i]))
            allFitPara = False
    if(allFitPara & log):
        print("\tAll Fit Parameters present")
            
    # Lastly, the Fit Data block
    for i in range (len(fit_data)):
        try:
            fc[fit_data[i]]
        except:
            if(log):
                print("\tFit Data {} not present!".format(fit_data[i]))
            allFitData = False
    if(allFitData & log):
        print("\tAll Fit Data present")
        
    #allgood = (allStellar & allFitPara & allFitData)
    return (allStellar & allFitPara & allFitData)

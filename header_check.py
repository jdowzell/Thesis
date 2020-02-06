def areEssentialHeadersPresent(f, log=False):
    """
    A function to make sure that all headers required for plotting a Light Curve
    are present, and alert the user if any are missing (and which ones).
    
    Input parameter (f) should be of the form: f=fits.getheader("file.fits")
    """
    
    # Generic Flag
    allgood = True
    
    # List of needed headers
    neededHeaders = ['TIME', 'LC_INIT']

    # FITS Columns
    fc = fits.getdata(f).columns
    
    # Loop through all headers and see if they are present using a Try/Except block
    for i in range (len(neededHeaders)):
        try:
            fc[neededHeaders[i]]
        except:
            if(log):
                print("Header {} not present!".format(neededHeaders[i]))
            allgood = False
        #else:
        #    print("{}: {}".format(neededHeaders[i], f[neededHeaders[i]]))
    return allgood

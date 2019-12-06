def areAllHeadersPresent(heads):
    """
    A function to make sure that all headers required for plotting a Light Curve
    are present, and alert the user if any are missing (and which ones).
    
    Input parameter (heads) should be of the form: heads=fits.getheader("file.fits")
    """
    
    # Generic Flag
    allgood = True
    
    # List of needed headers
    neededHeaders = ['TEFF', 'LOGG', 'TESSMAG',
                     'TPERIOD', 'TDUR', 'TEPOCH', 'TDEPTH',
                     'TIME', 'PHASE', 'LC_INIT', 'MODEL_INIT']
    
    # Loop through all headers and see if they are present using a Try/Except block
    for i in range (len(neededHeaders)):
        try:
            heads[neededHeaders[i]]
        except:
            #print("Header {} not present!".format(neededHeaders[i]))
            allgood = False
        #else:
        #    print("{}: {}".format(neededHeaders[i], heads[neededHeaders[i]]))
    return allgood

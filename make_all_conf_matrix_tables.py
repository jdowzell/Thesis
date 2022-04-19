#!/usr/bin/env python3

################################
# General imports
################################
import csv, math, io, os, os.path, sys, random, time, json, re
import pandas as pd

################################
# Functions
################################

def GetJSONFile(f):
    with open(f) as jf:
        jsonfile = json.load(jf)
    return(jsonfile)

################################

def GetReliabilityMetrics(data,cols):
    df = pd.DataFrame(data, columns=cols)
    df = df.drop(['CV','dateran', 'tstart', 'tfinish', 'tdelta'], axis=1, errors='ignore')

    df['Accuracy']  = ((df['TP']+df['TN']) / (df['TP']+df['TN']+df['FP']+df['FN']))
    df['Precision'] = ((df['TP'])          / (df['TP']+df['FP']))
    df['Recall']    = ((df['TP'])          / (df['TP']+df['FN']))
    TN, FP, FN, TP = df['TN'][0], df['FP'][0], df['FN'][0], df['TP'][0]
    
    a = (TP+TN)/(TP+TN+FP+FN)
    p = (TP)/(TP+FP)
    r = (TP)/(TP+FN)
    
    return(TN,FP,FN,TP,a,p,r)

################################

def OutputText(file, fname, dest):

    # Load and Parse the JSON file       
    jsondata = GetJSONFile(file)

    # Get the Keys
    keylist = []
    for keys in jsondata:
        keylist.append(jsondata[keys])

    columnList = list(keylist[0][0].keys())

    jsondatalist = []
    for i, x in enumerate(keylist):
        jsondatalist.append(list(list(x)[0].values()))

    #print(jsondatalist[0])

    # Convert TP,TF, etc into ints, not strings
    for row in jsondatalist:
        for i in range(3, len(row)-1):
            #print(row[i])
            try:
                row[i] = int(row[i])
            except ValueError:
                continue
                print("Row is a time not an int")

    # Get The Metrics
    (TN,FP,FN,TP,a,p,r) = GetReliabilityMetrics(jsondatalist, columnList)

    defaultText = r"""
    % Set these to make table look nicer
    \renewcommand{\arraystretch}{2}
    \renewcommand{\tabcolsep}{20.25pt}
    % Begin Table
    \begin{table}[ht]
    \begin{tabular}{cccc}
     & \multicolumn{3}{c}{Predicted Values} \\ \cline{3-4}
     & \multicolumn{1}{c|}{} & \multicolumn{1}{c|}{\textbf{Not Exoplanet}} & \multicolumn{1}{c|}{\textbf{Exoplanet}} \\ \cline{2-4}
    \multicolumn{1}{c|}{\multirow{2}{2.0cm}{Actual Values}} & \multicolumn{1}{c|}{\textbf{Not Exoplanet}} & \multicolumn{1}{c|}{True Negative} & \multicolumn{1}{c|}{False Positive} \\ \cline{2-4}
    \multicolumn{1}{c|}{} & \multicolumn{1}{c|}{\textbf{Exoplanet}} & \multicolumn{1}{c|}{False Negative} & \multicolumn{1}{c|}{True Positive} \\ \cline{2-4}
    \end{tabular}
    \caption{A Confusion Matrix, showing the comparison between the Actual and Predicted values, and what conclusion they represent. A \emph{True} result implies that the given model correctly predicted the presence, or absence, of a feature from the testing and training data. A \emph{False} result implies that the model incorrectly predicted the presence, or absence, of the feature.}
    \label{tab:XXXconfusionmatrix}
    \end{table}

    \label{eq:precisionXXX}
    \begin{align*}
        Accuracy &= &\frac{TP + TN}{TP + FP + TN + FN} &= &\frac{True Positive + True Negative}{True Positive + False Positive + True Negative + False Negative} &= & ACCURACY
        Precision &= &\frac{TP}{TP + FP} &= &\frac{True Positive}{True Positive + False Positive} &= & PRECISION
        Recall &= &\frac{TP}{TP + FN} &= &\frac{True Positive}{True Positive + False Negative} &= & RECALL
    \end{align*}

    % End Table
    % Set commands back to normal
    \renewcommand{\arraystretch}{1}
    \renewcommand{\tabcolsep}{5.25pt}
    """

    defaultText = re.sub("True Negative",  str(TN), defaultText)
    defaultText = re.sub("True Positive",  str(TP), defaultText)
    defaultText = re.sub("False Negative", str(FN), defaultText)
    defaultText = re.sub("False Positive", str(FP), defaultText)

    defaultText = re.sub("ACCURACY" , "{:.4%} \\\\\\\\".format(a), defaultText)
    defaultText = re.sub("PRECISION", "{:.4%} \\\\\\\\".format(p), defaultText)
    defaultText = re.sub("RECALL"   , "{:.4%} \\\\\\\\".format(r), defaultText)

    defaultText = re.sub("XXX", str(fname)[:-5], defaultText)
    defaultText = re.sub("\% \\\\", "\\\% \\\\", defaultText)

    #print(defaultText)
    
    saveDest = "./latex/"
    with open(dest+saveDest+fname[:-5]+".tex", "w") as myfile:
        myfile.write(defaultText)
    #print("Written file: {}.json".format(fname[:-5]))

################################
# main
################################

def main():
    
    # Check to see if file has been passed
    
    #path_to_json = "./confusionmatrices/"
    
    path_to_json = "./NEW_RESULTS/"
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    
    n = len(json_files)
    dest = "./NEW_RESULTS/"
    #print(sys.argv)
    #print(n)
    #print("List Empty: {}".format(not(sys.argv[1:])))
    
    for fname in json_files:
        try:
            #fname = sys.argv[1]
            file = dest+fname
            f = open(file, 'r')
            #print("f = {}".format(f))
        except FileNotFoundError:
            print("\nThe file, {}, cannot be found or does not exist\n".format(file))
            return
        except:
            print("You must specify exactly one file!\nYou specified: {} file(s)\n".format(n))
            return
        else:
            #print(file, 'has', len(f.readlines()), 'lines')
            #print("\nThe file, {}, has {} lines\n".format(file,len(f.readlines())))
            f.close()
            OutputText(file, fname, dest)


################################
# EXECUTE ORDER 66
################################

if __name__ == "__main__":
    main()
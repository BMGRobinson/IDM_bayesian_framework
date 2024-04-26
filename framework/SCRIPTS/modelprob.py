import os
import numpy as np

if __name__=='__main__':
    #For each model found, compute the model probabilites
    logev = []
    prob = []
    names = []

    for file in os.listdir("."):
        i = file.find("logevidence")
        if ( file.endswith(".dat") and (i != -1) ):
            f = open(file, "r")
            names.append(file.rstrip("-logevidence.dat"))
            logev.append(float(f.read()))

    evidenceTotal = 0.0
    for ev in logev:
        evidenceTotal += np.exp(ev)

    evidenceTotal = np.log(evidenceTotal)
    for i in range(0,len(logev)):
        number = np.exp(logev[i]-evidenceTotal)*100.0
        print (names[i] + " -------> " + "{:10.2f}".format(number) +"%")

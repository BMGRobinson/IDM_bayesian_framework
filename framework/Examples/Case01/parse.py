import numpy as np
from scipy import stats
import xml.etree.ElementTree as ET


def parseXML(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    #State size
    n = len(root.find('states').find('state').find('mean').text.split())

    #Time discretization
    numStates = len(root.find('states').findall('state'))

    #Number of particles (enkf, pf otherwise it's 0)
    npar = len(root.find('states').find('state').findall('p'))

    #nlik
    ##nlog = len(root.getElementsByTagName('loglikelihood'))
    #print(len(root.find('states').find('state').find('mean').text.split()))
    #print('Numstates = ', numStates)
    #print('Npar = ', npar)

    particles = np.zeros((npar,n,numStates))
    mean      = np.zeros((n,numStates))
    var       = np.zeros((n*n,numStates))
    weights   = np.zeros((n,numStates))
    time      = np.zeros(numStates)
    for x in root.findall('states'):
        for j in x.findall('state'):
            index = int(j.find('index').text)
            vec = j.find('mean').text.split()
            for s in range(0,n):
                mean[s,index] = float(vec[s])
            vec = j.find('covariance').text.split()
            for s in range(0,n*n):
                var[s,index] = float(vec[s])

            time[index] = float(j.find('time').text)

    return [time,mean,var]


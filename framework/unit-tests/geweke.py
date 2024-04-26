#Python script to save a jpg of p-p plot
import numpy as np
import matplotlib
#So that it can run on the cluster/bigdell2
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy import stats



#CDF plot of samples is defined by the proportion of x values less than or equal to x
#construct the CDF of each unidimensional samples

def cdf(X):
	n = len(X)
	#Sort from min to max
	cdfs = np.zeros([2,n])
	cdfs[1,:] = np.sort(X)
	
	for i in range(0,n):
		cdfs[0,i] = (i+1.0)/n

	return cdfs

def getPerc(value, cdfs):
    i = 0
    #Find the location of value on the cdf
    i = np.searchsorted(cdfs[1,:], value)    
    return cdfs[0,i]

xs = np.loadtxt("geweke_chain1.txt");
ys = np.loadtxt("geweke_chain2.txt");

xs = xs[0,:]
ys = ys[:,1]

X = cdf(xs)
Y = cdf(ys)

xl = min(min(xs), min(ys))
xr = min(max(xs), max(ys))

## for all the values in either sample (doesn't matter which one), get the cdf for the other
n = 1000
s = np.linspace(xl,xr,n)
pp = np.zeros([2,n])

for i in range(0,n):
    pp[0,i] = getPerc(s[i], X)
    pp[1,i] = getPerc(s[i], Y)


plt.plot(pp[0,:], pp[1,:],'.')
plt.savefig('geweke.png', bbox_inches='tight')


xs = np.loadtxt("geweke_chainB1.txt");
ys = np.loadtxt("geweke_chainB2.txt");

xs = xs[0,:]
ys = ys[:,1]

X = cdf(xs)
Y = cdf(ys)

xl = min(min(xs), min(ys))
xr = min(max(xs), max(ys))

## for all the values in either sample (doesn't matter which one), get the cdf for the other
n = 1000
s = np.linspace(xl,xr,n)
pp = np.zeros([2,n])

for i in range(0,n):
    pp[0,i] = getPerc(s[i], X)
    pp[1,i] = getPerc(s[i], Y)


plt.plot(pp[0,:], pp[1,:],'.')
plt.savefig('gewekeB.png', bbox_inches='tight')
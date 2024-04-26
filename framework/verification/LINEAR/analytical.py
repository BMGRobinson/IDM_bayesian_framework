## State estimation values (static problem)
## Forecast xkp1 = xk. Pkp1 = 0.5 + Pk
## Time = 0											Update with d = 0.5
## u = 0.0, P = 1.0							K =  1.0/(1.0+1.0) = 0.5  u = 0.0 + 0.5 * 0.5 = 0.25  P = 0.5
##
## Time = 0.5										Update with d = 1.0
## u = 0.25, P = 1.0							K =  1.0/(1.0+1.0) = 0.5  u = 0.25 + 0.5 * 0.75 = 0.625  P = 0.5*1.0 = 0.5
##
## Time = 1.0										Update with d = -0.2
## u = 0.625, P = 1.0							u =
##
##
import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt
sys.path.append('../../visualization')
from visualisation import *


data = np.loadtxt('data.dat')
#data = [1.51276083624124e-02]
var = np.loadtxt('variance.dat')
a = 0.00001
b = 10.0
#dt is 0.1. Variance of increase
#double ev1 = log(1.0/std::sqrt(2.0 * datum::pi * (1.0 + 1.0))) + (-0.5 * 0.5 * 0.5 / 2.0); //0.24000778968602721


def logfunc( sigma ):
    logpost = 0.0
    u = 0.0
    P = 1.0
    for d in data:
        ##Calculate log evidence
        y = d - u
        ev = np.log(1.0/np.sqrt(2.0*np.pi*(P+var))) -0.5 * y * y / (var + P)
        logpost = logpost + ev
        #print('Log evidence for data point ', d, ' is ', ev)
        ##Update
        S = var + P
        K = P/S
        u = u + K * y
        P = (1.0-K)*P

        ##Forecast until next data point
        P = P + 0.5*sigma**2

    return logpost - np.log(b)



##Trapezoidal rule
N = 10000
h = (b-a)/float(N)
sig = a
r = 0.0
fmax = 0.0
pdf = np.zeros(N)
sigs = np.zeros(N)
for i in range(1,N-1):
    sig = sig + h
    sigs[i] = sig
    f = np.exp(logfunc(sig))
    r = r + 2.0*f
    if f > fmax:
        fmax = f
        sigmamap = sig
    pdf[i] = f

r = r + np.exp(logfunc(a)) + np.exp(logfunc(b))
r = r * h * 0.5
pdf[0] = np.exp(logfunc(a))
pdf[N-1] = np.exp(logfunc(b))
sigs[0] = a
sigs[N-1] = b
x = 0.0

## Get analytical PDF p(sigma | D) = 1/P(D) * p(D|sigma)p(sigma)
chain = np.loadtxt('White-000.dat')
X,Y = get_marginal_pdf(chain[:,1], 500, 0.05)
plt.plot(sigs, pdf / r)
plt.plot(X,Y)
plt.legend(['Analytical', 'MCMC'])
plt.xlim([0.0,b])
plt.savefig("posterior_pdf.eps", format='eps', dpi=1000)

###Chib-Jeliazkov. Get posterior ordinate p(sigma* | D)
sigmastar = sigmamap
propcov = 12.0
num = 0.0
den = 0.0

print('Prop cov chol is ', np.sqrt(propcov) )

for i in range(0,N):
    ##Numerator
    num = num + min(1.0, np.exp(np.log(fmax) - chain[i,-1]))*1.0/np.sqrt(2.0*np.pi*propcov)*np.exp(-0.5*(chain[i,1]-sigmastar)**2/propcov)

    ##Denominator
    s = np.random.randn()*np.sqrt(propcov) + sigmastar
    if (s >= 0.0 and s <= b):
        f = logfunc(s)
        den = den + min(1.0, np.exp(f - np.log(fmax)))

num = num/float(N)
den = den/float(N)

logPos = np.log(num)-np.log(den)



print('********************************************')
print('********************************************')
print('sigma*: ', sigmamap)
print('Python: log p(D|sigma*)p(sigma*) = ', np.log(fmax) )
print('C++: log p(D|sigma*)p(sigma*) = ', -3.168998811007368)
print('')
print('Evidence Chib-Jeliazkov - Python')
print('Chib numerator is', num)
print('Chib den is', den)
print('log Ev (CJ):', np.log(fmax) - logPos)
print('')
print('Evidence - Numerical')
print('log Evidence:', np.log(r))
print('')
print('Posterior ordinate (CJ - Python):', np.exp(logPos), 'at sigma = ', sigmastar)
print('Posterior ordinate (analytical):', np.exp(logfunc(sigmastar)-np.log(r)), 'at sigma = ', sigmastar)
print('log Posterior ordinate (CJ - Python):', logPos, 'at sigma = ', sigmastar)
print('log Posterior ordinate (analytical):', logfunc(sigmastar)-np.log(r), 'at sigma = ', sigmastar)

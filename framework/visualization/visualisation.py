import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import getopt
import re
import xml.etree.ElementTree as ET
#from sklearn.neighbors import KernelDensity
#from pyqt_fit import kde_methods
#from pyqt_fit import kde

#Plugins to get
#git-+
#minimap
#minimap-git-diff
#atom-beautify

#adapted from http://bikulov.org/blog/2013/10/03/creation-of-paper-ready-plots-with-matlotlib/
# and http://bkanuka.com/articles/native-latex-plots/
def init_plotting( fontsize, labelsize, latex_textwidth_inpt ):
	fig_width_pt = latex_textwidth_inpt			# Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27					# Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0		# Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt		# width in inches
	fig_height = fig_width*golden_mean			# height in inches

	plt.rcParams['figure.figsize'] = (fig_width, fig_height)
	plt.rcParams['font.size'] = fontsize
	plt.rcParams['font.family'] = 'Nimbus Roman'
	plt.rcParams['axes.labelsize'] = labelsize
	plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
	plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
	plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
	plt.rcParams['savefig.dpi'] = 600#2*plt.rcParams['savefig.dpi']
	plt.rcParams['xtick.major.size'] = 3
	plt.rcParams['xtick.minor.size'] = 3
	plt.rcParams['xtick.major.width'] = 1
	plt.rcParams['xtick.minor.width'] = 1
	plt.rcParams['ytick.major.size'] = 3
	plt.rcParams['ytick.minor.size'] = 3
	plt.rcParams['lines.linewidth'] = 2
	plt.rcParams['ytick.major.width'] = 1
	plt.rcParams['ytick.minor.width'] = 1
	plt.rcParams['legend.frameon'] = False
	plt.rcParams['text.usetex'] = True
	plt.rcParams['legend.loc'] = 'center left'
	plt.rcParams['axes.linewidth'] = 1
	plt.gca().spines['right'].set_color('none')
	plt.gca().spines['top'].set_color('none')
	plt.gca().xaxis.set_ticks_position('bottom')
	plt.gca().yaxis.set_ticks_position('left')


#
#	Read XML State estimation file
#
#
#   Commented out lines relating to weights
#
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
            parid = 0
            for p in j.findall('p'):
                id = int(p.find('id').text)
                vec = p.find('val').text.split()
#                w = float(p.find('w').text)
                particles[parid,0,index] = float(vec[0])
                particles[parid,1,index] = float(vec[1])
#                weights[parid,index] = w
                parid = parid + 1

    return [time,mean,var,particles,weights]
#


#
#	Plot the autocorrelation function of a chain up to maxlag
#
#
def get_autocorr( chain, maxlag ):
	corr = np.zeros(maxlag)
	x_mean = np.mean(chain, dtype=np.float64);
	N = len(chain)

	den = 0
	lags = range(1, maxlag+1)
	for i in range(0,N):
		den += (chain[i] - x_mean)**2

	for lag in range(1, maxlag+1):
		num = 0
		for i in range(0,N - lag):
			num += (chain[i] - x_mean)*(chain[i+lag] - x_mean)
		corr[lag-1] = num / den

	return lags, corr

#
#	Load MCMC chain from text file
#
#
def load_mcmc(filename):
	return np.loadtxt(filename)


#
#	Plot MCMC samples in 1D
#
#
def plot_samples1D(samples, ytitle, step = 1):
	X = samples[::step]

	font = {'family' : 'serif',
	    'color'  : 'black',
	    'weight' : 'normal',
	    'size'   : 16,
	    }

	plt.plot(X,'bo')
	plt.xlabel('Sample', fontdict=font)
	plt.ylabel(ytitle, fontdict=font)
	return fig

#
#	Plot MCMC samples in 2D
#
#
def plot_samples2D(samplesX, samplesY, step = 1):
	X = samplesX[::step]
	Y = samplesY[::step]

	font = {'family' : 'serif',
	    'color'  : 'black',
	    'weight' : 'normal',
	    'size'   : 16,
	    }

	plt.plot(X,Y,'bo')
	plt.xlabel('Sample', fontdict=font)
	plt.ylabel('Sample', fontdict=font)
	plt.show()



def plot_hist(samples,bins,xlabel = '', ylabel = ''):

	# the histogram of the data
	n, bins, patches = plt.hist(samples, bins, normed=1)

	font = {'family' : 'serif',
	    'color'  : 'black',
	    'weight' : 'normal',
	    'size'   : 16,
	    }

	plt.xlabel(xlabel, fontdict=font)
	plt.ylabel(ylabel, fontdict=font)
	plt.grid(True)
	plt.show()

def plot_marginal_pdf_with_true(samples,xl,xr, numpoints, factor, xlabel, true_value):
	init_plotting()
	density = stats.kde.gaussian_kde(np.transpose(samples))
	density.covariance_factor = lambda : factor
	density._compute_covariance()
	xs = np.linspace(xl,xr,numpoints)
	font = {'family' : 'serif',
	    'color'  : 'black',
	    'weight' : 'normal',
	    'size'   : 16,
	    }
	fig = plt.figure()
	plt.plot(xs, density(xs))
	plt.xlabel(xlabel, fontdict=font)
	plt.ylabel('pdf', fontdict=font)
	plt.vlines(true_value,0,10.0*max(density(xs)), linestyles=[(0,(9,3,4,4))], colors='k')
	#return fig


def kde_scipy(x, x_grid, bw, selected_kernel):
	# score_samples() returns the log-likelihood of the samples
	kde = KernelDensity(kernel='tophat', bandwidth=bw).fit(x)
	log_pdf = kde.score_samples(x_grid)
	return np.exp(log_pdf)

def get_1D_kde(samples, numpoints, minx=None , maxx=None, bandwidth = 0.2, linear_approx = 0):
	if minx is None:
		minx = min(samples)

	if maxx is None:
		maxx = max(samples)

	xs = np.linspace(minx,maxx,numpoints)
	if (linear_approx == 0):
		est = kde.KDE1D(samples)
	else:
		est = kde.KDE1D(samples, lower=minx, method=kde_methods.linear_combination)
	return (xs, est(xs) )

def get_marginal_pdf(samples, numpoints, bw, xl=None, xr=None):
	if xl is None:
		xl = min(samples)
		#if xl > 0.0 :
		#	xl = xl * 0.90
		#else:
		#	xl = xl * 1.01

	if xr is None:
		xr = max(samples)
		#if xr > 0.0 :
		#	xr = xr * 1.01
		#else:
		#	xr = xr * 0.99

	density = stats.kde.gaussian_kde(np.transpose(samples))
	density.covariance_factor = lambda : float(bw)
	density._compute_covariance()
	xs = np.linspace(xl,xr,numpoints)
	return (xs,density(xs))

def plot_marginal_pdf(samples, numpoints, xlabel, xl=None, xr=None, factor = 0.2):
	density = stats.kde.gaussian_kde(np.transpose(samples))
	density.covariance_factor = lambda : factor
	density._compute_covariance()
	xs = np.linspace(xl,xr,numpoints)

	plot_graph(xs, density(xs), xlabel, 'pdf')

def plot_graph(X, Y, xlabel, ylabel ):

	font = {'family' : 'serif',
	    'color'  : 'black',
	    'weight' : 'normal',
	    'size'   : 16,
	   }

	fig = plt.figure()
	plt.plot(X, Y)
	plt.xlabel(xlabel, fontdict=font)
	plt.ylabel(ylabel, fontdict=font)
	return fig
	#plt.show()

#Helper function to plot the data attached to an axis
def plot_data(ax, X, Y, xlims = [], ylims = [], xlabel = "", ylabel = "", legendLabel = [], optionalArgs = []):
    # now all plot function should be applied to ax
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')

    # offset the spines
    #for spine in ax.spines.values():
    #        spine.set_position(('outward', 5))
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    # put the grid behind
    ax.set_axisbelow(True)


    # change xlim to set_xlim
    if (len(xlims) > 0):
        ax.set_xlim(xlims[0], xlims[1])
        #change xticks to set_xticks
        #ax.set_xticks(np.arange(xlims[0], xlims[1], 100))
    if (len(ylims) > 0):
        ax.set_ylim(ylims[0], ylims[1])

    if (optionalArgs != []):
        ax.plot(X, Y , optionalArgs, linewidth=2)
    else:
        ax.plot(X, Y , linewidth=2)

    if (ylabel != ""):
        #ax.set_yticklabels([])
        ax.set_ylabel(ylabel)

    if (xlabel != ""):
        #ax.set_yticklabels([])
        ax.set_xlabel(xlabel)

    if len(legendLabel) > 0:
        legend = ax.legend(legendLabel, loc=4);
        frame = legend.get_frame()
        frame.set_facecolor('1.0')
        frame.set_edgecolor('1.0')

def save_multiple_graph(X, Y, xlabel, ylabel , name, legend, args ):
	init_plotting()
	(nrows, ncols) = X.shape
	for i in range(0, nrows):
		plt.plot(X[i,:],Y[i,:], args[i])

	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.legend(legend)
	plt.savefig(name, format='eps', dpi=1000)
	plt.clf()

def save_single_graph(X, Y, xlabel, ylabel , name, arg = None, trueValue = None ):
	init_plotting()
	if (arg == None) :
		plt.plot(X,Y)
	else:
		plt.plot(X,Y, arg)

	if (trueValue != None):
		plt.vlines(trueValue ,0,1.2*max(Y), linestyles=[(0,(9,3,4,4))], colors='k')

	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.tight_layout()
	plt.savefig(name, format='eps', dpi=1000)
	plt.clf()

def save_graph(Xs, Ys, xlabels, ylabels, n , m, sharex, sharey, name ):
	init_plotting()
	#params = {
    #'axes.labelsize': 8,
    #'text.fontsize': 8,
    #'legend.fontsize': 10,
    #'xtick.labelsize': 10,
    #'ytick.labelsize': 10,
    #'text.usetex': True,
    #'figure.figsize': [4.5, 4.5]
	#}
	#mpl.rcParams.update(params)

	fig, axs = plt.subplots(n,m, sharex=False, sharey=False)
	#For each row
	for i in range(0,n):
		#For each column
		for j in range(0,m):
			axs[i,j].plot(Xs[i],Ys[i])

	#plt.plot(X, Y)
	#plt.xlabel(xlabel, fontdict=font)
	#plt.ylabel(ylabel, fontdict=font)
	fig.savefig(name, format='eps', dpi=1000)

def save_state_estimation(time, data, x, xlabel, ylabel, name):
	init_plotting()

	indexD = []
	segments = []
	updates = []
	#Find the index at which there is a measurement
	pI = 0
	for i in range(0,time):
		if (data[i] != -999.0):
			indexD.append(i)
			segments.append([x[pI:i],time[pI:i]])
			updates.append([[x[i-1], x[i]],[time[i-1], time[i]]])
			pI = i

	plt.plot(segments, 'k-')
	plt.plot(updates, 'r--')
	plt.show()
	return

def main(argv):
	inputfile = ''
	j = 0
	names = []
	bw = []
	doData = False
	doParam = True
	try:
		opts, args = getopt.getopt(argv,"hi:ts:n:b:d",['help'])
	except getopt.GetoptError as err:
		print(err)
		print('visualisation.py -i <inputfile>')
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h','--help'):
			print("Usage:")
			print('		test.py -[ts] -i <inputfile>')
			print('		t: Print the chains together')
			print('		s: Step')
			print('		n: names')
			print('     b: bandwidth')
			print('     d: data')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-t"):
			together = True
		elif opt in ("-s"):
			step = arg
		elif opt in ("-n"):
			names.append(arg)
		elif opt in ("-b"):
			bw.append(arg)
		elif opt in ("-d"):
			doData = True
			doParam = False

	init_plotting()

	if (doData):
		data = np.loadtxt("data.dat")
		true = np.loadtxt("true.dat")
		plt.plot(data[1,:], data[0,:], 'ro')
		plt.xlabel("Time [s]")
		plt.ylabel(names[0])
		plt.plot(true[-1,:], true[0,:], 'k')
		plt.legend(['Measurements', 'True signal'], 'upper right')
		plt.tight_layout()
		plt.savefig('data.eps', format='eps', dpi=1000)

	#Load the chain
	if(doParam):
		chain = np.loadtxt(inputfile)
		inputfile = re.sub('\.dat$', '', inputfile)

		(nrows, ncols) = chain.shape
		nParam = ncols - 2

		for i in range(0, nParam):
			save_single_graph(chain[::100,0], chain[::100,i+1] , "Samples", "$\\" + names[i] + "$", inputfile+'-samples-' + str(i) + '.eps', 'bo')

			#save_single_graph(chain[:,0], chain[:,i+1] , "Samples", names[i], inputfile+'-hist-' + str(i) + '.eps')
			X,Y = get_marginal_pdf(chain[:,i+1], 500, bw[i])
			save_single_graph(X, Y , "$\\" + names[i] + "$", "pdf" , inputfile+'-kde-' + str(i) + '.eps')

			X,Y = get_autocorr(chain[:,i+1], 50)
			save_single_graph(X, Y , "Lag", "Acorr" , inputfile+'-acorr-' + str(i) + '.eps')

	return

if __name__=='__main__':
	main(sys.argv[1:])

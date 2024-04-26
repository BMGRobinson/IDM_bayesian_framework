
#include "mcmc.hpp"
using namespace arma;

/*
* Helper methods
*/

void removeOffDiagonal(mat & A) {
	unsigned int i,j;
	for (i = 0; i < A.n_rows; i++) {
		for (j = 0; j < A.n_cols; j++) {
			if (i!=j) { A(i,j) = 0.0; }
		}
	}
}



//See https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
bool isAPowerOf2( const int x) {
		return (x != 0) && ((x & (x - 1)) == 0);
}
//return 2^x
int twoExp(const int x) {
	assert(x >= 0);
	if (x == 0) {
		return 1;
	}
	return 2*twoExp(x - 1);

}

//Only avaiable for proc in power of 2: 2,4,8,16,32, etc...
//see http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
mat getParallelCovariance(running_stat_vec<colvec> & lSamples, const MPI_Comm & com ) {
	const int N = lSamples.count();
	colvec T = double(N)*lSamples.mean();
	colvec Tp = lSamples.mean();
	mat Sp = lSamples.cov();
	colvec diff;
	std::cout.precision(16);
	mat S = double(N-1)*lSamples.cov();

	int dim = T.size();
	colvec Tother = zeros<colvec>(dim);
	mat Sother = zeros<mat>(dim,dim);
	int size, id;
	MPI_Comm_size(com, &size);
	MPI_Comm_rank(com, &id);
	assert(isAPowerOf2(size));
	const int gN = N * size;
	const int level = log(double(size))/log(2.0);
	mat Stemp;
	for (int l = 0; l < level; l ++) {
		//This processor will be active
		if ((id % twoExp(l)) == 0) {
				//Process that receives
				if ((id % twoExp(l+1)) == 0) {
					MPI_Recv(Tother.memptr(), dim, MPI_DOUBLE, id + twoExp(l), 0, com, MPI_STATUS_IGNORE);
					MPI_Recv(Sother.memptr(), dim*dim, MPI_DOUBLE, id + twoExp(l), 1, com, MPI_STATUS_IGNORE);
					diff = T - Tother;
					Stemp = diff * diff.t() / double(twoExp(l+1)*N);
					S += Sother + diff * diff.t() / double(twoExp(l+1)*N);
					T += Tother;
					//Process that sends
				} else {
					MPI_Send(T.memptr(), dim, MPI_DOUBLE, id - twoExp(l) , 0, com);
					MPI_Send(S.memptr(), dim*dim, MPI_DOUBLE, id - twoExp(l), 1, com);
				}
		}
	}
	//Broadcast S
	MPI_Bcast(S.memptr(), dim*dim, MPI_DOUBLE, 0, com);
	return S/(double(gN-1.0));
}


/*****************************************************************
MCMC CLASS
*****************************************************************/

mcmc::mcmc() {
	//MPI related variables
	id = 0;
	i = 0;
	parallelStateEstimation = false;
	chainSamples = running_stat_vec<colvec>(true);
	windowSamples = running_stat_vec<colvec>(true);

	checkRejectionRatioConvergence = false;
	checkKLDistanceConvergence = false;
	checkMinIterationsConvergence = false;
	checkMAPUnchangedConvergence = false;

	nruns = 1;
	ADAPTATION_WINDOW_LENGTH = 1;
	mapIterations = 0;
	minIterations = 0;
	Rtrace = 0.0;
	Rdet = 0.0;

	//Output related variables
	displayInfo = true;
	diagInfo = false;
	saveMap = true;
	saveProposal = true;
	printAtrow= 1 ;

	doDRAM = false;
	doAP = false;

	path = "";
	burninpath = "";
	chainpath = "";
	evidencepath = "";
	functionName = "";
	totalSamples = 0;
} //default constructor

mcmc::mcmc(const unsigned run, const unsigned window, const unsigned bwindow, const mat & _proposal) : mcmc() {
	proposalDistributionCov = _proposal;
	length = run*window;
	dim = _proposal.n_rows;
	chain = Samples(dim, length);
	samples_info = zeros<mat>(length,2);
	//currentSample.first = zeros<colvec>(dim);
	//currentSample.second = 0.0;
	random_sample = zeros<colvec>(dim);
	sd = 2.38*2.38/double(dim);
	ssd = (2.38/std::sqrt(double(dim)));
	//Get the cholesky of the proposal distribution covariance
	CholOfproposalDistributionCov = trans( chol(proposalDistributionCov) );
	ADAPTATION_WINDOW_LENGTH = bwindow;
	WINDOW_LENGTH = window;
	nruns = run;

	//allocate memory for map sample
	mapSample.first = zeros<colvec>(dim);
	mcmcId = -1; //default id
}

mcmc::mcmc(const unsigned run, const unsigned window, const unsigned bwindow, const mat & _proposal, bayesianPosterior _func) : mcmc(run, window, bwindow, _proposal) {
	func = _func;
}
/*
mcmc::mcmc(const std::string path, const std::string functionName, bayesianPosterior _func) : mcmc() {
	func = _func;

	//Load proposal
	proposalDistributionCov.load( path + "/proposal-"+functionName+".dat" ,raw_ascii);

	//Load MAP
	mapSample.first.load(path + "/map-"+functionName + ".dat", raw_ascii);
	mapSample.second = func.evaluate(mapSample.first); //bayesian posterior

	//Load chain
	//TODO:: fix parallel implementation here
	mat result;
	result.load(path + "/" + functionName + ".dat",auto_detect);
	length = result.n_rows;
	dim = result.n_cols - 2;

	//chain = Samples(dim, length);
	samples_info = zeros<mat>(length,2);
	mat _samples = zeros<mat>(dim, length);


	samples_info.unsafe_col(0) = result.unsafe_col(0);
	samples_info.unsafe_col(1) = result.unsafe_col(dim+1);

	//Copy the samples
	for (int j = 1; j < dim+1; j++) {
		_samples.row(j-1) = result.unsafe_col(j).t();
	}

	chain = Samples( _samples );

	//currentSample.first = zeros<colvec>(dim);
	//currentSample.second = 0.0;
	random_sample = zeros<colvec>(dim);
	sd = (2.38/std::sqrt(double(dim)));
	//Get the cholesky of the proposal distribution covariance
	CholOfproposalDistributionCov = trans( chol(proposalDistributionCov) );
}
*/

void mcmc::load() {
	//Load proposal
	proposalDistributionCov.load( path + "/proposal-"+functionName+".dat" ,raw_ascii);

	//Load MAP
	mapSample.first.load(path + "/map-"+functionName + ".dat", raw_ascii);
	mapSample.second = func.evaluate(mapSample.first); //bayesian posterior

	//Each proc load its chain
	int oldmcmcId = mcmcId;
	if (parallelStateEstimation) {
		MPI_Bcast(&mcmcId, 1, MPI_INT, 0, statecom); //Broadcast the mcmcId of that chain
	}

	mat result;
	std::string mcmcIds = std::to_string(mcmcId);

	//Return to the original value
	if (parallelStateEstimation) {
		mcmcId = oldmcmcId;
	}

	//Append "0" to the left
	if (mcmcIds.length() == 1) {
		mcmcIds = "00" + mcmcIds;
	} else if (mcmcIds.length() == 2) {
		mcmcIds = "0" + mcmcIds;
	}

	result.load(path + "/" + chainpath+ "/" + functionName + "-" + mcmcIds + ".dat",auto_detect);
	length = result.n_rows;
	dim = result.n_cols - 2;

	//chain = Samples(dim, length);
	samples_info = zeros<mat>(length,2);
	mat _samples = zeros<mat>(dim, length);

	samples_info.unsafe_col(0) = result.unsafe_col(0);
	samples_info.unsafe_col(1) = result.unsafe_col(dim+1);

	//Copy the samples
	for (int j = 1; j < dim+1; j++) {
		_samples.row(j-1) = result.unsafe_col(j).t();
	}

	chain = Samples( _samples );

	//currentSample.first = zeros<colvec>(dim);
	//currentSample.second = 0.0;
	random_sample = zeros<colvec>(dim);
	sd = (2.38/std::sqrt(double(dim)));
	ssd = (2.38/std::sqrt(double(dim)));

	//Get the cholesky of the proposal distribution covariance
	CholOfproposalDistributionCov = trans( chol(proposalDistributionCov) );
}

mcmc::~mcmc() {
}

void mcmc::setid( int _id ) {
	id = _id;
}

void mcmc::checkKLDistance(double _kldistance) {
	checkKLDistanceConvergence = true;
	kldistance = _kldistance;
}
void mcmc::checkRejectionRatio(double min, double max) {
	minrej = min;
	maxrej = max;
	checkRejectionRatioConvergence = true;
}
void mcmc::checkMinIterations(int miniterations) {
	minit = miniterations;
	checkMinIterationsConvergence = true;
}

void mcmc::checkMinMAPUnchangedIterations(int min) {
	mapChanged = false;
	minMAPNotChanged = min;
	checkMAPUnchangedConvergence = true;
}

Samples mcmc::getChain() {
	return chain;
}

void mcmc::setSD(double _sd) {
	sd = _sd;
}

void mcmc::setPath( const std::string _path, const std::string _burninpath, const std::string _chainpath, const std::string _evidencepath, const std::string _functionName) {
	path = _path;
	burninpath = _burninpath;
	chainpath = _chainpath;
	evidencepath = _evidencepath;
	functionName = _functionName;
	if (id == 0) {
		std::ofstream burnInlogFile(path + "/" + burninpath + "/burnInInfo-" + functionName + ".dat", std::ios_base::out | std::ios_base::trunc );
		burnInlogFile << std::setprecision(6) << std::setw(12) << "Iteration "<< std::setprecision(6) << std::setw(12) << "Sample" << std::setprecision(6) << std::setw(15) << "lnMap";
		burnInlogFile << std::setprecision(3) << std::setw(12) << "Min" << std::setprecision(3) << std::setw(12) << "Max";
		burnInlogFile << std::setprecision(3) << std::setw(12) << "Rdet" << std::setprecision(3) << std::setw(12) << "Rtrace" << std::endl;
		burnInlogFile.close();

		std::ofstream runlogFile(path + "/" + chainpath + "/Info-" + functionName + ".dat", std::ios_base::out | std::ios_base::trunc );
		runlogFile << std::setprecision(6) << std::setw(12) << "Sample" << std::setprecision(6) << std::setw(15) << "lnMap";
		runlogFile << std::setprecision(3) << std::setw(12) << "Min" << std::setprecision(3) << std::setw(12) << "Max" << std::endl;
		runlogFile.close();
	}

}

void mcmc::propose_new_sample(const mcmcSample& mean, mcmcSample& proposedSample, const mat & CholOfCov) {
	if (id == 0) { //Root propose a new sample
		random_sample.randn(); //Each element of the vector is drawn from a Normal(0,1)
		proposedSample.first = mean.first + CholOfCov * random_sample;
	}
	if (parallelStateEstimation) {
		statecom.Bcast(proposedSample.first.memptr(), dim , MPI::DOUBLE, 0);
	}

	//Evaluate log likelihood of current_sample
	proposedSample.second = func.evaluate(proposedSample.first); //bayesian posterior
}

/*
*	returns min(1, p(y)/p(x)), only valid for symmetric proposal distribution, p(x), p(y) are known
*/
long double mcmc::getAcceptanceProbability(mcmcSample& x, mcmcSample& y) {
	long double ratio = getDensityRatio(x, y);
	double alpha = 0.0;
	if (id == 0) {
		alpha = std::min(1.0L, ratio);
	}
	return alpha;
}
/*
*		returns p(y)/p(x). p(x) & p(y) are known should already be calculated
*/
long double mcmc::getDensityRatio(mcmcSample& x, mcmcSample& y) {
	double ratio;

	if (id == 0) {
		if (std::isnan(y.second)) {
			ratio = 0.0;
		} else {
			//std::cout << "Proposed log density = " << proposedLogDensity << " Previous log density = " << previousLogDensity << std::endl;
			//ratio = std::min(1.0L, std::exp(proposedLogDensity - previousLogDensity));
			ratio = std::exp(y.second - x.second);
		}
	} else {  //For other process
		ratio = 0.0;
	}
	return ratio;
}

//Accept the proposed sample
void mcmc::accept_sample( mcmcSample& proposedSample ) {
	if (diagInfo && mcmcId == 0) {
		std::cout << "Sample " << i << std::endl;
		proposedSample.first.print("Accepted sample:");
	}

	/* Have we found a new map? */
	if ( proposedSample.second > mapSample.second ) {
		if (diagInfo && mcmcId == 0) {
			std::cout << "New map at " << proposedSample.second << " VS " << mapSample.second << std::endl;
			proposedSample.first.print("New map is ");
		}
		mapSample = proposedSample;
		mapChanged = true;
	}
	//currentSample = proposedSample;
}
//Assign a point to each subchain
bool mcmc::setStartingPoint(const colvec & point) {

	bool validSample = true;
	if (id == 0) {
		mapSample.first = point;
	}

	if (parallelStateEstimation) {
		statecom.Bcast(mapSample.first.memptr(), dim , MPI::DOUBLE, 0);
	}


	mapSample.second = func.evaluate(mapSample.first);
	validSample = !(std::isnan(mapSample.second));

	if (parallelStateEstimation) {
		statecom.Bcast(&validSample, 1, MPI::BOOL, 0);
	}

	if (id == 0 && !validSample) {
		//std::cout << "WARNING: log density at starting point is -infinity. Exiting." << std::endl;
	}

	currentSample = mapSample;
	return validSample;
}

void mcmc::setPrintAtrow( int _row ) {
	printAtrow = _row;
}

void mcmc::enableDiagInfo() {
	diagInfo = true;
}

void mcmc::displayStatus() {
	//Print progress bar
	if (id == 0) {
		std::cout << "\033[" << printAtrow+3 << ";1H" <<std::flush; //need row

		double progress =  double(i+1)/double(length);
		int w = 25;
		int c =  progress * w; //there are 25 bars

		std::cout << "Chain: " << std::setw(3) << int(progress*100.0) << "% [";
		for (int x=0; x<c; x++) std::cout << "=";
		for (int x=c; x<w; x++) std::cout << " ";
		std::cout << "]" << " Rejection ratio : " << std::setw(3) << int(rejratio*100.0) << "%\r" << std::flush;
		return;
	}
}


// Print each row correspond to a sample
/*
* Sample #     Sample   logDensity
*/
void mcmc::save_proposal(bool flag) {
	saveProposal = flag;
}
void mcmc::save_map( bool flag) {
	saveMap = flag;
}

void mcmc::savePropAndMapBurnin() {
	if (mcmcId == 0 && saveProposal) {
		proposalDistributionCov.save(path + "/burnin-proposal-"+functionName+".dat", raw_ascii);
	}
	if (saveMap && mcmcId == 0) {
		mapSample.first.save(path + "/burnin-map-"+functionName + ".dat", raw_ascii);
	}
}

bool mcmc::print() {
	//Save the proposal and map
	bool res = false;
	std::string mcmcIds = std::to_string(mcmcId);
	//Append "0" to the left
	if (mcmcIds.length() == 1) {
		mcmcIds = "00" + mcmcIds;
	} else if (mcmcIds.length() == 2) {
		mcmcIds = "0" + mcmcIds;
	}

	if (mcmcId == 0 && saveProposal) {
		proposalDistributionCov.save(path + "/proposal-"+functionName + ".dat", raw_ascii);
	}
	if (saveMap && mcmcId == 0) {
		mapSample.first.save(path + "/map-"+ functionName + ".dat", raw_ascii);
	}
	if (id == 0) {
		mat result = zeros<mat>(length, dim + 2);
		result.unsafe_col(0) = samples_info.unsafe_col(0); //sample number
		result.unsafe_col(dim+1) = samples_info.unsafe_col(1); //logDensity
		for (int j = 1; j < dim+1; j++) {
			result.unsafe_col(j) = chain.getSamples().row(j-1).t();
		}
		//res = result.save(path + "/" + functionName + ".dat",arma_binary);
		res = result.save(path + "/" + chainpath + "/" + functionName + "-" + mcmcIds + ".dat",raw_ascii);
	}
	if (parallelStateEstimation) {
		statecom.Bcast(&res, 1, MPI::BOOL, 0);
	}

	//Print Summary
	double * rejratios;
	if (id == 0) {
		rejratios = new double[numChains];
		MPI_Gather(&rejratio, 1 ,MPI_DOUBLE  , rejratios , 1, MPI_DOUBLE, 0, headnodescom);
	}

	if (mcmcId == 0) {
		std::ofstream log_file("summary" + functionName + ".txt", std::ios_base::out | std::ios_base::trunc );
		log_file << "Summary for " << functionName << std::endl;
		log_file << std::setw(15) << "Chain Id" << std::setw(15) << "Samples"  << std::setw(15) << "Reject Ratio" << std::endl;
		for (int s = 0; s < numChains; s ++) {
			log_file << std::setw(15) << s << std::setw(15) << length << std::setw(15) << rejratios[s] << std::endl;
		}
		log_file.close();
	}

	return  res;
}

void mcmc::enableScreenInfo() {
	displayInfo = true;
}
void mcmc::disableScreenInfo(){
	displayInfo = false;
}

void mcmc::displayBurnInInfo() {
	std::cout  << "\033[" << printAtrow+1 << ";1H" <<std::flush;
	std::cout << "\33[2K" << std::flush;  //Erase the content of the current line
	//Following line erase the content of the console
	std::cout  << "[Burn-in " << functionName << "] Sample " << totalSamples  << " ["  << std::setw(3) << mapIterations << " / " << std::setw(3) <<  minMAPNotChanged <<  "]" << " Iteration "  << std::setw(3) <<  minIterations << " / " << std::setw(3) << minit << std::flush;
	std::cout << " Rej: " <<  std::setprecision(3) << std::setw(3) << gminrej * 100.0 << "% - " << std::setprecision(3) << std::setw(3) << gmaxrej * 100.0  << "%" << std::flush;
	std::cout << " MAP: " << std::setw(3) << std::setprecision(3) << mapSample.second << std::flush;
	std::cout << " Rdet: " <<  std::setprecision(3) << std::setw(3) << Rdet << " Rtrace: " <<  std::setprecision(3) << std::setw(3) << Rtrace << "\r" << std::flush;
}

void mcmc::storeBurnInInfo() {
	std::ofstream burnInlogFile(path + "/" + burninpath + "/burnInInfo-" + functionName + ".dat", std::ios_base::out | std::ios_base::app );
	burnInlogFile << std::setprecision(6) << std::setw(12) << minIterations;
	burnInlogFile << std::setprecision(6) << std::setw(12) << totalSamples << std::setprecision(6) << std::setw(15) << mapSample.second;
	burnInlogFile << std::setprecision(3) << std::setw(12) << gminrej << std::setprecision(3) << std::setw(12) << gmaxrej;
	burnInlogFile << std::setprecision(3) << std::setw(12) << Rdet << std::setprecision(3) << std::setw(12) << Rtrace << std::endl;
	burnInlogFile.close();
}


void mcmc::displayChainInfo() {
	std::cout  << "\033[" << printAtrow+1 << ";1H" <<std::flush;
	std::cout << "\33[2K" << std::flush; //Erase the content of the current line
	std::cout  << "[Chain " << functionName << "] " << "Sample " << totalSamples  <<  std::flush;
	std::cout << " Rej: " <<  std::setprecision(3) << std::setw(3) << gminrej * 100.0 << "% - " << std::setprecision(3) << std::setw(3) << gmaxrej * 100.0  << "%" << std::flush;
	std::cout << " MAP: " << std::setw(3) << std::setprecision(3) << mapSample.second << std::flush;
}

void mcmc::storeChainInfo() {
	std::ofstream runlogFile(path + "/Info-" + functionName + ".dat", std::ios_base::out | std::ios_base::app );
	runlogFile << std::setprecision(6) << std::setw(12) << totalSamples << std::setprecision(6) << std::setw(15) << mapSample.second;
	runlogFile << std::setprecision(3) << std::setw(12) << gminrej << std::setprecision(3) << std::setw(12) << gmaxrej << std::endl;
	runlogFile.close();
}

colvec mcmc::getMAP() {
	return mapSample.first;
}
//TODO wrong goodness of fit
void mcmc::evidenceSummary(const long double & logEv, const long double & logGoodnessOfFit, const long double & EIG) {
	//Print Summary
	long double ev[numChains];
	long double gf[numChains];
	long double eig[numChains];
	long double avg = 0.0;
	long double avggf = 0.0;
	long double avgeig = 0.0;
	if (id == 0) {
		MPI_Gather(&logEv,						1,	MPI_LONG_DOUBLE,	ev,		1,	MPI_LONG_DOUBLE,	0,	headnodescom);
		MPI_Gather(&logGoodnessOfFit,	1,	MPI_LONG_DOUBLE,	gf,		1,	MPI_LONG_DOUBLE,	0,	headnodescom);
		MPI_Gather(&EIG,							1,	MPI_LONG_DOUBLE,	eig,	1,	MPI_LONG_DOUBLE,	0,	headnodescom);
	}

	if (mcmcId == 0) {
		//std::ofstream log_file("Evidence-" + functionName + ".txt", std::ios_base::out | std::ios_base::trunc );
		std::ofstream log_file(path + "/" + evidencepath + "/evidence-" + functionName + ".dat", std::ios_base::out | std::ios_base::trunc );
		log_file << "Evidence Summary for " << functionName << std::endl;
		log_file << std::setw(15) << "Chain Id" << std::setw(15) << "lnEvidence"  << std::setw(15) << "lnGF" << std::setw(15) << "lnEIG" <<  std::endl;
		for (int s = 0; s < numChains; s ++) {
			log_file << std::setprecision(9) << std::setw(15) << s << std::setprecision(9) << std::setw(15) << ev[s] << std::setprecision(9) << std::setw(15) << gf[s] << std::setprecision(9) << std::setw(15) << eig[s] << std::endl;
			avg += ev[s];
			avggf += gf[s];
			avgeig += eig[s];
		}
		log_file << "Average " << std::setprecision(9) << std::setw(15) << avg/double(numChains) << std::setprecision(9) << std::setw(15) << avggf/double(numChains) << std::setprecision(9) << std::setw(15) << avgeig/double(numChains) << std::endl;
		log_file.close();
	}
}

void mcmc::logEvidenceAtMAP(int J, long double & logEv, long double & logGoodnessOfFit, long double & EIG, int trim) {
	logEv = logEvidence( mapSample.first , J, trim);
	long double logPrior = func.evaluatePrior(mapSample.first);
	long double thetaStarDensity = mapSample.second; //= log p(d|parameters) + log p(parameters)
	logGoodnessOfFit = getAvglogGoodnessOfFit(trim);
	EIG = logGoodnessOfFit - logEv;

	evidenceSummary(logEv, logGoodnessOfFit, EIG);

}

long double mcmc::logEvidence(const colvec& thetaStar, int J, int trim) {
	long double thetaStarDensity = func.evaluate(thetaStar); //bayesian posterior
	long double logposterDensityAtThetaStar = chibJeliazkov(thetaStar,thetaStarDensity, proposalDistributionCov, J, trim);
	//std::cout << "ln(p(sigma*|D)) = " << logposterDensityAtThetaStar << std::endl;
	return thetaStarDensity-logposterDensityAtThetaStar;
}

//Returns Eq. 9 in Marginal Likelihood from the M-H Output.
// I did not fixe thetaStar to the map because we can then look
// at the error in terms of thetaStar and in terms of J (draws from the proposal distribution)
long double mcmc::chibJeliazkov(const colvec& thetaStar, const long double thetaStarDensity, const mat& propCovariance,  int J, int trim) {
	long double num,den;
	num = chibJeliazkov_numerator(thetaStar, thetaStarDensity,  propCovariance, trim);
	den = chibJeliazkov_denominator(thetaStar, thetaStarDensity, propCovariance, J);
	//std::cout << "Map is " << std::endl << thetaStar << " density at map is " << thetaStarDensity << " Cov is " << std::endl << propCovariance << std::endl << " and J is " << J << std::endl;
	//std::cout << "NUM = " << std::exp(log(num)-log(double(length/trim))) << ", DEN = " << std::exp(log(den)-log(double(J))) << std::endl;
	//std::cout << "MCMC Length = " << length << std::endl;
	//std::cout << "MCMC Trim = " << trim << std::endl;
	//std::cout << "MCMC Length/Trim = " << length/trim << std::endl;
	//std::cout << "J = " << J << std::endl;
	return log(num) - log(den) + log(double(J)) - log(double(length/trim));
}
//Return the average goodness of fit
long double mcmc::getAvglogGoodnessOfFit(int trim) {
	long double lgf, currentLogDensity;
	lgf = 0.0;
	if (id == 0) {
		for (int i = 0; i < length; i = i + trim) {
			currentLogDensity = samples_info(i,1);
			//Remove the prior
			lgf += currentLogDensity - func.evaluatePrior(chain.getSampleAt(i));
		}}
	lgf = lgf / double(length/trim);
	if (parallelStateEstimation) {
		statecom.Bcast(&lgf, 1 , MPI::LONG_DOUBLE, 0);
	}
	return lgf;
}

//Numerator of Eq. 9 in MarginadisplayBurnInInfol Likelihood from the M-H Output.
//Uses the likelihood stored in the matrix from the parameter estimation step
//Only valid for symmetric proposal distribution.
//not normalized by length
long double mcmc::chibJeliazkov_numerator(const colvec& thetaStar, const long double thetaStarDensity, const mat& propCovariance, int trim) {
	long double currentLogDensity;

	//Build the fixed proposal distribution
	Gaussian q = Gaussian( thetaStar, propCovariance );
	long double logtemp;
	double _q;
	long double numerator = 0.0L;
	if (id == 0) {
		for (int i = 0; i < length; i = i + trim) {
			currentLogDensity = samples_info(i,1); //LogDensity of sample i
			// the next line corresponds to log(min(1, thetaStarDensity/currentLogDensity) * q(sample) )
			logtemp = std::min(0.0L, thetaStarDensity - currentLogDensity) + q.getLogDensity( chain.getSampleAt(i) );

			numerator += std::exp(logtemp);
			//first method may reduce computational errors
			//std::min(1.0, std::exp(thetaStarDensity - currentLogDensity)) * std::exp( q.getLogDensity( chain.getSampleAt(i) ));
		}}
	if (parallelStateEstimation) {
		statecom.Bcast(&numerator, 1 , MPI::LONG_DOUBLE, 0);
	}
	return numerator;
}

//Denominator of Eq. 9 in Marginal Likelihood from the M-H Output.
//Only valid for symmetric proposal distribution.
//Not normalized by J
long double mcmc::chibJeliazkov_denominator(const colvec& thetaStar,const long double thetaStarDensity, const mat& propCovariance, int J) {
	//Generate N samples drawn from the normal proposal distrubtion with mean thetaStar and covariance propCovariance
	//accumulate the loglikelihood
	colvec thetaJ = zeros<colvec>(dim);
	mat propCovarianceChol = trans( chol(propCovariance) );
	long double alpha = 0.0L;
	long double ratio,currentLogDensity;
	int w,c;
	double progress;
	//Running for each sub-chain

	for (int i = 0; i < J; i++) {
		//propose sample
		if (id == 0) {
			random_sample.randn(); //Each element of the vector is drawn from a Normal(0,1)
			thetaJ = thetaStar + propCovarianceChol * random_sample;
			//thetaJ = thetaStar + CholOfproposalDistributionCov * random_sample;
		}

		//Broadcast new thetaJ
		if (parallelStateEstimation) {
			statecom.Bcast(thetaJ.memptr(), dim , MPI::DOUBLE, 0);
		}
		//Evaluate alpha
		currentLogDensity = func.evaluate(thetaJ); //bayesian posterior
		if (std::isnan(currentLogDensity)) {
			ratio = 0.0;
		} else {
			ratio = std::min(1.0L, std::exp(currentLogDensity - thetaStarDensity));
		}

		alpha += ratio;

		if (mcmcId == 0 && i>0 && ((i+1) % DISPLAY_WINDOW_LENGTH) == 0) {
			if (displayInfo) {
				std::cout << "\033[" << printAtrow+4 << ";1H" <<std::flush;
				progress =  double(i+1)/double(J);
				w = 25;
				c =  progress * w;
				std::cout << "[Chib-Jeliazkov]   " << int(progress*100.0) << "% [";
				for (int x=0; x<c; x++) std::cout << "=";
				for (int x=c; x<w; x++) std::cout << " ";
				std::cout << "]" << "\r" << std::flush;
			}
		}
	}
	return alpha;
}
/*
* TODO need to use AP so instead use a enumaration instead of bool
*/
void mcmc::ap_preruns(const int _runs) {
	//std::cout << "IN AP_PRERUNS" << totalSamples << std::endl;
	for (int r = 0; r < _runs; r ++) {
		generate(ADAPTATION_WINDOW_LENGTH, CholOfproposalDistributionCov, rejratio, windowSamples, chainSamples, false );
		share_map();
		share_rejectionRatio();

		if (mcmcId == 0) {
			displayBurnInInfo();
			storeBurnInInfo();
		}

		if (id == 0) {
			adaptChain();
		}

		windowSamples.reset();
		chainSamples.reset();

		//Start the new starting point based on the MAP
		propose_new_sample(mapSample, currentSample, CholOfproposalDistributionCov);

	}
	if (mcmcId == 0) {
		std::ofstream burnInlogFile(path + "/" + burninpath + "/burnInInfo-" + functionName + ".dat", std::ios_base::out | std::ios_base::app );
		burnInlogFile << "Completed Adaptive Proposal runs" << std::endl;
		burnInlogFile.close();
	}


}

void mcmc::burnin() {
	bool adapt = true;
	//Each group works on his own chain
	minIterations = 0; //Need to fix that
	//Divide the chain in windows
	totalSamples = 0;

	while (adapt) {
		windowSamples.reset();
		generate(ADAPTATION_WINDOW_LENGTH, CholOfproposalDistributionCov, rejratio, windowSamples, chainSamples, false );
		share_map();
		share_rejectionRatio();

		if (id == 0) {
			adapt = !checkConvergence();
			if (adapt) { adaptChain(); }
		}

		//Output the info
		if (mcmcId == 0) {
			displayBurnInInfo();
			storeBurnInInfo();
		}

		//Share convergence across all process
		if (parallelStateEstimation) {
			MPI_Bcast(&adapt, 1,  MPI_BYTE, 0, statecom);
		}
	}
	if (mcmcId == 0) {
		savePropAndMapBurnin();
	}
}
//Following is true only for symmetric proposals
void mcmc::generate(const int _l, const mat & _cholCov, double & _rejratio, running_stat_vec<colvec> & window_stats, running_stat_vec<colvec> & chain_stats, const bool store  ) {
	double r, alpha,  alphadr; //r - U(0,1), alpha is the move probability and percRej is the rejection ratio percentage (a number between 0 and 100)
	mcmcSample proposedSample;
	int rej = 0;
	int j;
	bool _DRAM = false;
	proposedSample.first = zeros<colvec>(dim);
	proposedSample.second = 0.0;
	window_stats.reset();
	for (j = 0; j <  _l; j ++) {
		totalSamples ++;
		//Statistics of the chain
		if (id == 0) {
			window_stats(currentSample.first);
			chain_stats(currentSample.first);
		}

		//Storing the samples
		if (store && id == 0) {
			chain.setSampleAt(i, currentSample.first);
			samples_info.at(i,0) = i;
			samples_info.at(i,1) = currentSample.second;
			i++;
		}
		propose_new_sample(currentSample, proposedSample, _cholCov);
		alpha = getAcceptanceProbability(currentSample, proposedSample); //getDensityRatio(); //All process can call this function
		if (id == 0) {
			r = randu();
			if (r < alpha) {
				accept_sample(proposedSample);
				currentSample = proposedSample;
			} else {
				//Sample rejected
				//if DRAM and u[0,1] < probability of doing dram for this sample
				if (doDRAM && randu() < dramProb) {
					_DRAM = true;
				} else {
					_DRAM = false;
				}
				rej ++;
			}
		}
		if (doDRAM && parallelStateEstimation) {
			MPI_Bcast(&_DRAM, 1,  MPI_BYTE, 0, statecom);
		}
		/*
		*	Delayed rejection - proposing another sample
		*/
		if (doDRAM && _DRAM) {
			propose_new_sample(currentSample, proposedSample, drScale * _cholCov);
			alphadr = std::min(1.0L,  (getDensityRatio(currentSample, proposedSample) - alpha)/(1.0 - alpha));
			if (id == 0) {
				r = randu();
				if (r < alphadr) {
					accept_sample(proposedSample);
					currentSample = proposedSample;
				}
			}
		}

	}
	_rejratio = double(rej)/double(_l);
}

void mcmc::run() {
	//Reset samples
	totalSamples = 0;
	for (int s = 0; s < nruns; s ++) {
		generate(WINDOW_LENGTH, CholOfproposalDistributionCov, rejratio, windowSamples, chainSamples, true );
		share_map();
		share_rejectionRatio();

		if (mcmcId == 0) {
			displayChainInfo();
			storeChainInfo();
		}

		if (id == 0) {  //Only one proc per chain
			adaptChain();
		}
	}
}




void mcmc::share_rejectionRatio() {
	struct key{
		double val;
		int rank;
	} mykey, minkey, maxkey;

	mykey.val = rejratio;
	mykey.rank = mcmcId;

	//Find which process has the maximum and minimum rejection ratio
	if (id == 0) {
		MPI_Allreduce(&rejratio, &gminrej , 1, MPI_DOUBLE, MPI_MIN, headnodescom);
		MPI_Allreduce(&rejratio, &gmaxrej , 1, MPI_DOUBLE, MPI_MAX, headnodescom);
	}
}
//Share the map and the covariance of the chains
void mcmc::share_map() {
	struct key{
		double val;
		int rank;
	} mykey, maxkey;

	mykey.val = mapSample.second;
	mykey.rank = mcmcId;

	//Find which process has the MAP
	if (id == 0) {
		MPI_Allreduce(&mykey, &maxkey , 1, MPI_DOUBLE_INT, MPI_MAXLOC, headnodescom);
		mapSample.second = maxkey.val;
		MPI_Bcast(mapSample.first.memptr(), dim,  MPI_DOUBLE, maxkey.rank, headnodescom);
	}
	//Share map with the state estimation group
	if (parallelStateEstimation) {
		statecom.Bcast(mapSample.first.memptr(), dim , MPI::DOUBLE, 0);
		statecom.Bcast(&mapSample.second, 1 , MPI::DOUBLE, 0);
	}
}

//Chain adpatation
//AP, AM, DRAM
void mcmc::adaptChain() {
	//Compute the global covariance matrix and cholesky decomposition
	mat globalCov;

	if (doAP) {
		globalCov	 = getParallelCovariance(windowSamples, headnodescom);
		proposalDistributionCov = sd*globalCov;
		CholOfproposalDistributionCov = trans(chol(proposalDistributionCov));
	}
	//Use whole history
	if (doAM || doDRAM) {
		globalCov	 = getParallelCovariance(chainSamples, headnodescom);
		proposalDistributionCov = sd*globalCov;
		CholOfproposalDistributionCov = trans(chol(proposalDistributionCov));
	}
}

void mcmc::setAM() {
	doAP = false;
	doAM = true;
	doDRAM = false;
}
void mcmc::setDRAM(const double DRProb, const double DRScale) {
	doAP = false;
	doAM = false;
	doDRAM = true;
	dramProb = DRProb;
	drScale = DRScale;
}
void mcmc::setAP() {
	doAP = true;
	doAM = false;
	doDRAM = false;
}

/*
* Convergence methods. Returns true when adaptation is over (chain has converged)
*/
bool mcmc::checkConvergence() {
	bool global;
	bool flagRej = true, flagKL = true, flagMin = true, flagMap = true, flagBRG = true;

	if (checkRejectionRatioConvergence) {
		flagRej = checkRejectionRatio();
	}

	//if (checkKLDistanceConvergence) {
	//	flagKL = checkKLDistance(previousCov, cov);
	//}

	if (checkMAPUnchangedConvergence) {
		if (mapChanged) {
			mapIterations = 0;
			mapChanged = false; //Reset the flag
		} else {
			mapIterations++;
		}
		flagMap = (mapIterations > minMAPNotChanged);
	}

	if (checkMinIterationsConvergence) {
		minIterations ++;
		flagMin = (minIterations > minit);
	}
	flagBRG = checkBGR();

	global = flagRej && flagKL && flagMin && flagMap && flagBRG;
	//MPI_Bcast(&global, 1,  MPI_INT, 0, headnodescom);
	MPI_Bcast(&global, 1,  MPI_BYTE, 0, headnodescom);
	return global;
}
/*
*	If rejection ratio falls between [min, max] return true
*/
bool mcmc::checkRejectionRatio() {
	return (gmaxrej < maxrej && gminrej > minrej);
}

/*
* See Gelman, Andrew, et al. Bayesian data analysis, p. 303-304
* Returns if max(R) < Rmax, normally 1.1
*/

void mcmc::setBGR( double _maxrdet, double _maxrtrace ) {
	maxRtrace = _maxrtrace;
	maxRdet = _maxrdet;
}


//W is the average covariance matrix of each chain
//B/n is the variance of the chains average
bool mcmc::checkBGR() {
	//m chains (j) of n samples (i)

	mat means = zeros<mat>(dim, numChains); //Each col contains the mean value of a chain
	mat W = zeros<mat>(dim,dim);
	mat BoverN = zeros<mat>(dim,dim);
	mat diff, V;
	double m = double(numChains);
	double n = double(ADAPTATION_WINDOW_LENGTH);
	bool converged = false;
	/* MPI */
	//Gather mean and variance
	MPI_Gather(windowSamples.mean().memptr()  ,dim,MPI_DOUBLE,means.memptr(), dim, MPI_DOUBLE, 0, headnodescom);
	colvec mu = mean(means,1);

	MPI_Reduce(windowSamples.cov().memptr(), W.memptr(), dim*dim, MPI_DOUBLE, MPI_SUM, 0, headnodescom);
	W = W / m;
	diff = means - repmat(mu,1,numChains);
	BoverN = 1.0/(m - 1.0)*diff*diff.t();

	if (mcmcId == 0) {
		V = (n - 1.0)/n * W + (1.0+1.0/m)*BoverN;
		Rtrace = trace(V)/trace(W);
		Rdet = det(V)/det(W);

		if (Rtrace <= maxRtrace && Rdet <= maxRdet) {
			converged = true;
		}
	}
	//MPI_Bcast(&converged, 1,  MPI_INT, 0, headnodescom);
	MPI_Bcast(&converged, 1,  MPI_BYTE, 0, headnodescom);

	return converged; //
	}


/*
bool mcmc::checkKLDistance(const mat& previousCov, const mat& cov) {
	double currentkldistance;
	mat covInv;
	if (!inv_sympd(covInv,cov) ) {
		std::cout << "ERROR: can't inverse the covariance" << std::endl;
		cov.print("Covariance:");
	}
	currentkldistance = 0.5 * trace(covInv*previousCov) - double(dim) + log(det(cov)/det(previousCov));
	if (std::abs(currentkldistance) < kldistance) { return true; } else { return false; }
}
*/
void mcmc::setStateEstimatorCom( const MPI::Intracomm& _com) {
	statecom = _com;
	id = statecom.Get_rank();
	parallelStateEstimation = true;
}
//Multiple ids
void mcmc::setParallelChainsCom(const MPI::Intracomm& _com ) {
	headnodescom	 = _com;
	MPI_Comm_rank(headnodescom, &mcmcId);
	MPI_Comm_size(headnodescom, &numChains);

	/*
	int procPerchain = 1;
	//How manu chains?
	//How many process per chain (parallelStateEstimation)
	if (parallelStateEstimation) {
		procPerchain = MPI_SIZE
	}
	*/
}


/*
*		Not used, to calculate the mean and covariance. Replaced with running_stat_vector of Armadillo c++
*

			//Recursively keep track of the mean
			if (i == 0 ) {
				mean = current_sample;
				X_prime.unsafe_col(0) = current_sample;
			} else { //i = 1 or over
				mean = (previousMean * double(i) + current_sample) / ( double(i) + 1.0);
				mean.print("My mean");
				std::cout << "Their mean " << burninSamples.mean() << std::endl;
			}

			//Recursively keep track of the covariance. Need at least two samples for calculate covariance
			if (i == 1) {
				X_prime.unsafe_col(1) = current_sample;
				X_prime = X_prime - repmat( mean, 1, 2);
				previousCov = X_prime * trans(X_prime);
			} else if (i > 1) {
				//Eq.3 Haario 2001 paper
				cov = (double(i) - 1.0)/double(i) * previousCov + 1.0/double(i)*( double(i) * previousMean*trans(previousMean) - (double(i) + 1.0) * mean * trans(mean)  + current_sample*trans(current_sample) );
				previousCov = cov;
				cov.print("My covariance:");
				std::cout << "Their covariance " << burninSamples.cov() << std::endl;
			}
			//Record the current mean for the next iteration
			previousMean = mean;




*/

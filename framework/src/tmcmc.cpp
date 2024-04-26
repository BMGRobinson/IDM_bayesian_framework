#include "tmcmc.hpp"
using namespace arma;

void tmcmc::propose_new_sample(const tmcmcSample& mean, tmcmcSample& proposedSample, const mat & CholOfCov) {
	if (id == 0) { //Root propose a new sample
		random_sample.randn(); //Each element of the vector is drawn from a Normal(0,1)
		proposedSample.value = mean.value + CholOfCov * random_sample;
	}
  /*
	if (parallelStateEstimation) {
		statecom.Bcast(proposedSample.value.memptr(), dim , MPI::DOUBLE, 0);
	}
  */
	//Evaluate log likelihood of current_sample
	proposedSample.loglik = func.evaluateLogLikelihood(proposedSample.value); //bayesian posterior
  proposedSample.logprior = func.evaluatePrior(proposedSample.value); //bayesian posterior

	//std::cout << "proposed sample loglik/logprior is " << proposedSample.loglik << " , " << proposedSample.logprior << std::endl;
	//std::cout << "map sample loglik/logprior is " << mapSample.loglik << " , " << mapSample.logprior << std::endl;

  if ((proposedSample.loglik + proposedSample.logprior) > (mapSample.loglik + mapSample.logprior)) {
		mapSample = proposedSample;
		//std::cout << "Found a new map sample!" << std::endl;
	}

}

/*
*	returns min(1, p(y)/p(x)), only valid for symmetric proposal distribution, p(x), p(y) are known
*/
long double tmcmc::getAcceptanceProbability(tmcmcSample& x, tmcmcSample& y) {
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
long double tmcmc::getDensityRatio(tmcmcSample& x, tmcmcSample& y) {
	double ratio;

	if (id == 0) {
		if (std::isnan(y.loglik + y.logprior)) {
			ratio = 0.0;
		} else {
			ratio = std::exp((y.loglik + y.logprior) - (x.loglik + x.logprior));
		}
	} else {  //For other process
		ratio = 0.0;
	}
	return ratio;
}


void tmcmc::setPath( const std::string _path, const std::string _chainpath, const std::string _evidencepath, const std::string _functionName) {
	path = _path;
	chainpath = _chainpath;
	evidencepath = _evidencepath;
	functionName = _functionName;
}


tmcmc::tmcmc() {}

tmcmc::tmcmc(int _N, int _dim, double _COV_threshold, double _COV_threshold_convergence, bayesianPosterior _func, const MPI_Comm& _headnodescom ) {
	Nglobal = _N;
	dim = _dim;

  COV_threshold = _COV_threshold;
  COV_threshold_convergence = _COV_threshold_convergence;
	random_sample = zeros<colvec>(dim);
	//betaSqrd = 2.38*2.38/double(dim);
	betaSqrd = 0.04;
	func = _func;
  //logLikelihoods = zeros<colvec>(Nlocal); //Each proc has it's on array
	headnodescom = _headnodescom;
	id = 0;
	if (headnodescom != MPI_COMM_NULL) {
		MPI_Comm_rank(headnodescom, &mcmcId);
		MPI_Comm_size(headnodescom, &numProcs);
		logLikelihoods = SamplesMPI(1, Nglobal, headnodescom);
		LikelihoodsTrial = SamplesMPI(1, Nglobal, headnodescom);

		samples = WeightedSamplesMPI(dim, Nglobal, headnodescom);
		first = samples.getFirst(); //need to check for parallelStateEstimation
		last = samples.getLast(); //inclusive
		Nlocal = samples.getSize();
  	proposedSample = tmcmcSample(zeros<colvec>(dim), 0.0, 0.0);
  	currentSample = tmcmcSample(zeros<colvec>(dim), 0.0, 0.0);
		mapSample = tmcmcSample(zeros<colvec>(dim), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
	} else {
		mcmcId = -1;
	}

}

void tmcmc::run() {

double p_up, p_low, p_trial, p, p_previous;
double COV; //Coeffcient of variation
std::vector<unsigned int> localIndex, globalIndex;
bool p_convergence;
int i,j;
double lsj = 0.0;
mat globalChain, proposal_cov, globalLogLikelihood, tempGlobalLogLikelihood;
int iteration = 0;
vec::fixed<1> logLik;
double w;
double max = 0.0;
double sumWeight = 0.0;
globalIndex.reserve(Nglobal);
mat proposal_cov_chol = zeros<mat>(dim,dim);
//Sample from the prior distribution and record loglikelihood.
//Currently only works with ind priors
std::vector<pdf1d*> priors = func.getPriors();
for (i = first; i <= last; i++ ) {
  for (j = 0; j < dim; j++) {
    random_sample[j] = priors[j]->sample();
  }
  //std::cout << random_sample << std::endl;
  samples.setSampleAt(i, random_sample);
	//Not optimal
	//std::cout << "Calculating point likelihood: " << std::endl;
	logLik[0] = func.evaluateLogLikelihood(random_sample);
    if (logLik[0]!=logLik[0]) {    
        logLik[0] = -1.0e+10;        
        //std::cout << "log likelihood is nan" << std::endl;
    }
	//std::cout << logLik[0] << std::endl;
	logLikelihoods.setSampleAt(i, logLik);
}

globalChain = samples.getSamplesAtRoot();
if (mcmcId == 0) {
	globalChain.save(path + "/" + chainpath + "/" + functionName + "-" + "prior" + ".dat",raw_ascii);
}

//Run until the posterior is sampled (last run will be with p = 1.0)
p_previous = 0.0;
p = 0.0;
while (p < 1.0) {
  //Finding pj+1 so that the Coeffcient of Variation (COV) is equal to a prescribed threshold
  //Using the Bisection method
  p_convergence = false;
  p_low = p_previous;
  p_up = 1.0;

	//Retrieve the maximum value so we can normalize by it
	max = log(0);

	for (i = first; i<= last; i++) {
		if (max < as_scalar(logLikelihoods.getSampleAt(i))) {
			max = as_scalar(logLikelihoods.getSampleAt(i));
		}
	}
	//Get the max value
	MPI_Allreduce( &max, &max, 1, MPI_DOUBLE, MPI_MAX, headnodescom);

	/* Following block find the next p */
  while (!p_convergence) {
    p_trial = (p_up + p_low)/2.0;

		for (i = first; i <= last; i++ ) {
			//std::cout << "Log value is " << logLikelihoods.getSampleAt(i) << std::endl;
			logLik[0] = exp( as_scalar((logLikelihoods.getSampleAt(i)-max) * (p_trial-p_previous)));
			//std::cout << "Value is " << logLik[0] << std::endl;
			LikelihoodsTrial.setSampleAt(i, logLik);
		}
		//std::cout << "STD: " << sqrt(LikelihoodsTrial.getCovarianceAtRoot()) << " and mean is " << LikelihoodsTrial.getMeanAtRoot() << std::endl;
    COV = as_scalar(sqrt(LikelihoodsTrial.getCovarianceAtRoot()))/as_scalar(LikelihoodsTrial.getMeanAtRoot()); //Since COV is a double, all procs have the COV
		MPI_Bcast(&COV, 1, MPI_DOUBLE, 0, headnodescom);
		//std::cout << "COV IS " << COV << std::endl;
		//COV = stddev(temp)/mean(temp);
    if ((std::abs(COV - COV_threshold) < COV_threshold_convergence) || p_trial > 0.998 ) {
    	p = p_trial;
    	p_convergence = true;
    } else {
    	//Change the bounds of bisection method
    	//Root is at the right of p_trial
    	if ((COV - COV_threshold) > 0.0) {
      	p_up = p_trial;
    	} else {
      	p_low = p_trial;
    	}
    }
	 	//So that the last p will always be 1.0
  	if (p_trial > 0.998) {
    	p = 1.0;
  	}
	}
	if (mcmcId == 0) {
		std::cout << "P used: " << p << std::endl;
	}
	/* Evluate the weights of each sample */
  sumWeight = 0.0;
  for (i = first; i <= last; i++ ) {
    w = std::exp(as_scalar((p-p_previous)*(logLikelihoods.getSampleAt(i)-max)));
    samples.setWeightAt(i, w);
    sumWeight += w;
  }
	MPI_Allreduce(MPI_IN_PLACE, &sumWeight, 1, MPI_DOUBLE, MPI_SUM, headnodescom );

  //Normalize the weights
  for (i = first; i <= last; i++ ) {
    samples.normalizeWeightAt(i, sumWeight);
  }

	//At this point we resample based on the weights
  //Each procs has part of the localIndex
	//globalChain = samples.getSamplesAtRoot();
	//if (mcmcId == 0) {
  //  globalChain.save("Chain_br_" + std::to_string(iteration) + ".dat", raw_ascii);
  //}

  //localIndex = samples.systematic_resampling_index();


  //Evidence estimation
  //lsj = lsj + log(mean(w)) + (p_cur-p_prv)*max(ll);
  //lsj += log(mean(w));

  //For each sample do a 1 step MH hastings step
  //With proposal covariance defined in Eq. 17
  //proposal_cov.zeros(); //Reset proposal covariance
  //for (int i = first, i < last; i ++) {
  //proposal_cov += w[i]*(Samples.col(i)-)
  //}
	//for (i = first; i <= last; i ++) {
	//	std::cout << localIndex[i] << " " << std::endl;
	//}
  proposal_cov = betaSqrd*samples.getCovarianceAtRoot();
  cout << proposal_cov;

	if (mcmcId == 0) {
		proposal_cov_chol = trans( chol(proposal_cov) );
	}


	//The root broadcast to all process - now will give an error for parallel state estimation
	MPI_Bcast(proposal_cov_chol.memptr(), dim*dim, MPI_DOUBLE, 0, headnodescom);

  proposedSample.value = zeros<colvec>(dim);
	proposedSample.loglik = 0.0;
  proposedSample.logprior = 0.0;
  j = 0;
  rej = 0;

	//Due to resampling, need
	previousSamples = samples;
	previouslogLikelihoods = logLikelihoods;
	localIndex = previousSamples.systematic_resampling();
	previouslogLikelihoods.redistribute( localIndex );
	/*
	for (i = 0; i < Nlocal; i ++) {
		std::cout << localIndex[i] << " ";
	}
	std::cout << std::endl;
	*/
	//Rearange the logLikelihoods based on the localIndex
	//First step is to get the globalIndex vector

	//MPI_Gather(&localIndex.front(), Nlocal, MPI_UNSIGNED, &globalIndex.front(), Nlocal, MPI_UNSIGNED,0, headnodescom );
	//globalLogLikelihood = logLikelihoods.getSamplesAtRoot();

	//Roots reorganize
	//if (mcmcId == 0) {
	//	tempGlobalLogLikelihood = globalLogLikelihood;
	//	for (i = 0; i < Nglobal; i ++) {
	//		tempGlobalLogLikelihood.unsafe_col(i) = globalLogLikelihood.unsafe_col( globalIndex[i] );
			//tempGlobalLogLikelihood.unsafe_col(i) = globalLogLikelihood.unsafe_col( localIndex[i] );
			//previouslogLikelihoods.setSampleAt(i, globalLogLikelihood.unsafe_col( localIndex[i] ));
			//previouslogLikelihoods.setSampleAt(i, logLikelihoods.getSampleAt( localIndex[i] ));
	//	}
	//}

	//tempGlobalLogLikelihood.print("Scatter this");
	//previouslogLikelihoods.getSamples().print("From this");
	//MPI_Scatter(tempGlobalLogLikelihood.memptr(), Nlocal, MPI_DOUBLE, previouslogLikelihoods.getSamples().memptr(), Nlocal, MPI_DOUBLE, 0, headnodescom);
	//previouslogLikelihoods = logLikelihoods.redistribute( localIndex );
	//previouslogLikelihoods.getSamples().print("Into this");


  for (i = first; i <= last; i++) {
    //currentSample.value = previousSamples.getSampleAt( localIndex[j] );
		//currentSample.loglik = as_scalar(previouslogLikelihoods.getSampleAt( localIndex[j] ));
		currentSample.value = previousSamples.getSampleAt( i );
		currentSample.loglik = as_scalar(previouslogLikelihoods.getSampleAt( i ));
    currentSample.logprior = func.evaluatePrior( currentSample.value );
		//std::cout << "[1/2] CurrentSample value is " << currentSample.value << " from j = " << j << " localIndex of " << localIndex[j] << std::endl;

    propose_new_sample(currentSample, proposedSample, proposal_cov_chol);
		alpha = getAcceptanceProbability(currentSample, proposedSample); //getDensityRatio(); //All process can call this function
		if (id == 0) {
			r = randu();
			if (r < alpha) {
				currentSample = proposedSample;
			} else {
				rej ++;
			}
		}
		//std::cout << "[2/2] Accepted sample value is " << currentSample.value << std::endl;
    samples.setSampleAt(i, currentSample.value);
    //logLikelihoods[i] = currentSample.loglik;
		logLik[0] = currentSample.loglik;
		logLikelihoods.setSampleAt(i, logLik);
  }

	//Share the rejection ratio
	if (mcmcId == 0) {
		MPI_Reduce(MPI_IN_PLACE, &rej, 1, MPI_INT, MPI_SUM, 0, headnodescom);
	} else {
		MPI_Reduce(&rej, &rej , 1, MPI_INT, MPI_SUM, 0, headnodescom);
	}
	if (mcmcId == 0) {
  	std::cout << "Rejection ratio " << double(rej)/double(Nglobal) << std::endl;
	}
  //MH steps completed
	//globalChain = samples.getSamplesAtRoot();
	if (mcmcId == 0) {
  	//globalChain.save(path + "/" + chainpath + "/" + functionName + "-" + std::to_string(iteration) + ".dat", raw_ascii);
  }
  iteration++;
  p_previous = p;
}

globalChain = samples.getSamplesAtRoot();
if (mcmcId == 0) {
	//globalChain.save("Chain_posterior.dat", raw_ascii);
	globalChain.save(path + "/" + chainpath + "/" + functionName + "-" + "posterior" + ".dat",raw_ascii);
	mapSample.value.save(path + "/" + chainpath + "/" + functionName + "-" + "map" + ".dat",raw_ascii);
}



}

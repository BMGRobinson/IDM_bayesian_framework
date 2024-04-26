#ifndef TMCMC_HPP_
#define TMCMC_HPP_

//Following is used to disable warning from outside libraries
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <iomanip> //std::setw
#include <mpi.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>
#include <functional>
#include "bayesianPosterior.hpp"
#pragma GCC diagnostic pop

#include "samples.hpp"

class tmcmcSample {
public:
	colvec value;
	double loglik;
	double logprior;
	tmcmcSample() {};
	tmcmcSample(colvec value, double loglik, double logprior):value(value), loglik(loglik), logprior(logprior) {};

};

using namespace arma;


/**
TMCMC class.
*/
class tmcmc {
public:
	/*
	*  Constructors
	*/
	tmcmc(); //default constructor required for gtest
	tmcmc(int _N, int _dim, double _COV_threshold, double _COV_threshold_convergence, bayesianPosterior _func,const MPI_Comm& _headnodescom );
	//~tmcmc();
	void run();

	//Set up parallel state estimation or parallel mcmc
	void setStateEstimatorCom( const MPI::Intracomm& _com);

	void setPath( const std::string _path, const std::string _chainpath, const std::string _evidencepath, const std::string _functionName);

private:
	tmcmcSample currentSample, proposedSample, mapSample;
	WeightedSamplesMPI samples, previousSamples;
	SamplesMPI logLikelihoods, LikelihoodsTrial, previouslogLikelihoods;
	int dim;
	int first;
	int last;
	int Nglobal;
	int Nlocal;
	std::string path, chainpath, evidencepath, functionName;
	bayesianPosterior func;
	colvec random_sample;
	double COV_threshold;
  double COV_threshold_convergence;
	double betaSqrd;
	double r, alpha;
	int rej;


	/*
	* Parallel TMCMC
	*/
	int numProcs;
	MPI_Comm headnodescom;
	int chainId;
	int mcmcId;

	//Communicator used when state estimation is parallel
	bool parallelStateEstimation;
	MPI::Intracomm statecom;
	int id;


	//Helper methods
	void propose_new_sample(const tmcmcSample& mean, tmcmcSample& proposedSample, const mat & CholOfCov);
	long double getAcceptanceProbability(tmcmcSample& x, tmcmcSample& y);
	long double getDensityRatio(tmcmcSample& x, tmcmcSample& y);
};
#endif

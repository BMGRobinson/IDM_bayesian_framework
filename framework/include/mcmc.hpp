#ifndef MCMC_HPP_
#define MCMC_HPP_

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

//logLikelihoodFunc
//typedef std::function<double(const colvec &)> logPosteriorFunc;
typedef std::pair<colvec, double> mcmcSample;

using namespace arma;

/* Helper functions */
void removeOffDiagonal(mat & A);
bool isAPowerOf2( const int x);
int twoExp(const int x);
mat getParallelCovariance(running_stat_vec<colvec> & lSamples, const MPI_Comm & com );

enum class ADAPTATION_METHOD { AP, AM, DRAM };

/**
MCMC class.
*/
class mcmc {
public:
	/*
	*  Constructors
	*/
	mcmc(); //default constructor required for gtest
	mcmc(const unsigned run, const unsigned window, const unsigned bwindow, const mat & _proposal); //Underlying constructor, should not be called directly
	mcmc(const unsigned run, const unsigned window, const unsigned bwindow, const mat & _proposal, bayesianPosterior _func );

	//Constructor used to load MCMC chain from textfile
	mcmc(const std::string path, const std::string functionName, bayesianPosterior _func);
	~mcmc();

	/*
	*	Path
	*/
	std::string path;
	std::string burninpath;
	std::string chainpath;
	std::string evidencepath;
	std::string functionName;

	void setPath( const std::string _path, const std::string _burninpath, const std::string _chainpath, const std::string _evidencepath, const std::string _functionName);

	/*
	DRAM
	*/
	double dramProb;
	bool doDRAM, doAP, doAM;
	double drScale;

	/*
	* Loading the chain
	*/
 	void load();

	/*
	* Burn-in methods
	*/
	bool adaptiveProposal(int window);
	bool adaptiveMetropolisburnin(int startIndex, int ADAPTATION_WINDOW_LENGTH, int maxSamplesUntilConvergence);
	bool DRAMburnin(int startIndex, int ADAPTATION_WINDOW_LENGTH, int maxSamplesUntilConvergence, double drProb, double drScale);
	bool adaptiveProposal(int window, int nruns);
  void generate(const int _l, const mat & _cholCov, double & _rejratio, running_stat_vec<colvec> & window_stats, running_stat_vec<colvec> & chain_stats, const bool store  );

	/*
	* Sampler
	*/
	bool adaptiveMetropolis(int ADAPTATION_WINDOW_LENGTH);
	bool DRAM(int ADAPTATION_WINDOW_LENGTH, double drProb, double drScale);
	void metropolisHastings();


	void setAM();
  void setDRAM(const double DRProb, const double DRScale);
	void setAP();
	void ap_preruns(const int _runs);
	/*
	* Get & Set methods
	*/
	void setid( int _id );

	void setSD(double _sd);
	colvec getMAP();
	void setPrintAtrow( int _row );
	void setStateEstimatorCom( const MPI::Intracomm& _com);
	void setParallelChainsCom(const MPI::Intracomm& _com);

	Samples getChain();
	bool setStartingPoint( const colvec & point );


	/*
	*	Display and flags
	*/
	void enableScreenInfo();
	void disableScreenInfo();
	void enableDiagInfo();
	bool print();
	void savePropAndMapBurnin();
	void save_proposal(bool flag);
	void save_map( bool flag);
	void displayStatus();


	/*
	*	Convergence flag
	*/
	void checkKLDistance(double kldistance);
	void checkRejectionRatio(double min, double max);
	void setBGR( double maxrdet, double maxrtrace );
	//TODO: fix that
	bool checkRejectionRatio();
	bool checkBGR();


	void checkMinIterations(int min);
	void checkMinMAPUnchangedIterations(int min);

	/*
	*	Evidence calculation
	*/
	void logEvidenceAtMAP(int J, long double & logEv, long double & logGoodnessOfFit, long double & EIG, int trim);
	long double logEvidence(const colvec& thetaStar, int J, int trim);
	void evidenceSummary(const long double & logEv, const long double & logGoodnessOfFit, const long double & EIG);
	void burnin();
	void share_map();
	void share_rejectionRatio();
	void run();
	long double getAvglogGoodnessOfFit(int trim);
private:

	mcmcSample mapSample; //also the staring point
	mcmcSample currentSample; //Each chain track the current position of the chain

	/*
	* Parallel MCMC
	*/
	int numChains;
	MPI::Intracomm headnodescom;
	int chainId;
	int mcmcId;

	//Communicator used when state estimation is parallel
	bool parallelStateEstimation;
	MPI::Intracomm statecom;
	int id;
	int totalSamples;
	int i;
	int nruns;
	/*
	*	MCMC variables
	*/
	running_stat_vec<colvec> chainSamples;
	running_stat_vec<colvec> windowSamples;
	mat previousCov, cov, covChol;
	colvec random_sample;

	int printAtrow;
	int ADAPTATION_WINDOW_LENGTH;
	int WINDOW_LENGTH;

	//bool customFunction;
	mat CholOfproposalDistributionCov;
	mat proposalDistributionCov;
	bool displayInfo;
	bool diagInfo;
	//logPosteriorFunc customFunc;
	bayesianPosterior func;

	mat samples_info;

   /*
	*	Convergence functions
	*/
	bool checkConvergence();
	bool checkRejectionRatio(double percRej);
	//bool checkKLDistance(const mat& previousCov, const mat& cov) ;

	/*
	*	Convergence flags and related variables
	*/
	bool checkRejectionRatioConvergence;
	bool checkKLDistanceConvergence;
	bool checkMinIterationsConvergence;
	bool checkMAPUnchangedConvergence;
	bool mapChanged;
	int minit; //required minimum number of iterations
	int minMAPNotChanged; //required minimum number of iterations where the map doesn't change
	double kldistance;
	double minrej, maxrej, gminrej, gmaxrej; //minimum and maximum rejection ratio
	int minIterations;
	int mapIterations;
	double Rtrace, Rdet, maxRtrace, maxRdet;

	void displayBurnInInfo();
	void storeBurnInInfo();
	void displayChainInfo();
	void storeChainInfo();


	bool saveMap;
	bool saveProposal;

	Samples chain;

	int DISPLAY_WINDOW_LENGTH = 1000;

	int dim;
	int length;
	double rejratio;

	double sd; //Rule of thumb Factor for Gaussian distributions.
	double ssd;

	void accept_sample(mcmcSample& sample);
	void accept_sample_dontsave(mcmcSample& sample);
	void propose_new_sample(mcmcSample& proposedSample);
	void propose_new_sample(const mcmcSample& mean, mcmcSample& proposedSample, const mat & CholOfCov);
	void scaled_propose_new_sample(mcmcSample& proposedSample, double scale);
	long double getDensityRatio(mcmcSample& x, mcmcSample& y);
	long double getAcceptanceProbability(mcmcSample& x, mcmcSample& y);

	//TODO: fix

	void adaptChain();

	//double chibJeliazkov(const mat & chain, int J, const colvec & parameter);
	//Helping function for chibJeliazkov to calculate the denominator
	long double chibJeliazkov(const colvec& thetaStar, const long double thetaStarDensity, const mat& propCovariance,  int J, int trim);
	long double chibJeliazkov_denominator(const colvec& sample,const long double thetaStarDensity, const mat& propCovariance, int J);
	long double chibJeliazkov_numerator(const colvec& thetaStar,const long double thetaStarDensity, const mat& propCovariance, int trim);

	//Store burnin info
	//std::ofstream burnInlogFile;
};

#endif

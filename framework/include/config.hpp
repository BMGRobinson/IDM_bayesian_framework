#pragma once
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <armadillo>
#include "pdf.hpp"
#include <libconfig.h++>
#include <iostream>
#include <fstream>
#include <limits>
#include <string.h>
using namespace libconfig;
using namespace arma;

/*
*	In settings for each proposed models, you give the liklihood function. If they are deterministic, no filters are necessary
*	if filters are necessary, need to construc statespace model and the filter
*
*/

//Dirty hack gloval variable
//static bool globalLogPrint;
//static bool globalConsolePrint;

struct mcmcParam {
	//Convergence
	int AP_PreRuns;
	int minIterations;
	int minMAPNotChangedIterations;
	int maxSamplesUntilConvergence;
  double rdet;
	double rtrace;

	//Initial
	colvec initialParameters;
	mat initialProposal;

	//Method MH, AM, AP, DRAM, etc.
	std::string method;
	std::string burnin_method;

	//Length of the chain
	int nruns;
	int window;
	int burnin_window;
	int nchains;

	//DRAM
	double DRProb;
	double DRScale;

	bool save_map;
	bool save_proposal;
	bool diaginfo;

};

struct tmcmcParam {

	//Length of the chain
	int window;
	int nprocs;
	int dim;
	double cov;
	double cov_tol;

	bool save_map;
	bool save_proposal;
	bool diaginfo;
};

struct evidenceParam {
	std::string method;	 //Chib or Quad

	//Chib-Jeliazkov
	int trim;

	//Gauss-Hermite quadrature
	colvec mu;
	mat sigma;
	int quadLevel;
	double quadTolerance;
};

struct genModelParam {
	double time;
	double dt;
	double NSR;
	int stepsBetweenMeasurements;
	colvec parameters;
	colvec initialState;
	std::string function_name;
	std::string function_handle;
	std::string folder;
};

struct proposedModels {
	bool run;
	bool doParameterEstimation;
	bool doEvidence;
	bool doStateEstimationRun;
	//bool state_estimation;
	double dt;
	int fStepsBetweenMeasurements;
	colvec initialState;
	mat initialStateVariance;
	mat modelCov;
	std::string function_name;
	std::string function_handle; //used for the likelihood function
	std::string folder;
	std::string data;
	std::string cov;
	std::string state_estimator;
	int nparticles;
	int nprocs;
	int parallelGroups;
	bool doOptimization;
	int nparameters;
	int nelderMeadMaxIt;
	std::vector<pdf1d*> priors;
	bool consoleprint;
	bool logprint;
	colvec parameters;
	//MCMC
	mcmcParam mcmcParameters;
	tmcmcParam tmcmcParameters;
  std::string mcmcMethod;

	//State-Estimation
	int seruns;
	//Evidence
	evidenceParam evidenceParameters;
	//Computing the error
	bool doStateEstimationError;
};
template <class V>
std::string convert(V val);
std::string convert(bool val);
void print(const std::string &text, const int id);


template <class T>
void getFlagWithDefault(Config& cfg, Setting& gm, std::string valueName, T& value, const T defaultValue, int procId  );
template <class U>
bool getFlagRequired(Config& cfg, Setting& gm, std::string valueName, U& value, int procId  );

void getFlagWithDefault(Config& cfg, Setting& gm, std::string valueName, std::string& value, const std::string defaultValue, int procId  );
bool getFlagRequired(Config& cfg, Setting& gm, std::string valueName, std::string& value, int procId  );

mat getMatrix(Setting& stgs,  std::string settingName, int size );
mat getMatrix(Config& cfg,  std::string settingName, int size );
colvec getVector(Config& cfg,  std::string settingName, int size );
colvec getVector(Setting& stgs,  std::string settingName, int size );

bool getProposedModelParameters(Config& cfg, std::vector<proposedModels>& propModelVector, const MPI::Intracomm _com );
bool getGeneratingModelParameters(Config& cfg, std::vector<genModelParam>& genModelVector );
bool getMCMCParameters(Config& cfg, Setting & gms, mcmcParam& _parameters, const MPI::Intracomm _com );
bool getEvidenceParameters(Config& cfg, Setting & gms, evidenceParam& _parameters, const MPI::Intracomm _com );
bool openConfigFile( Config& cfg,  const char * filename );
bool readVector( colvec& vector, Config& cfg, std::string filename );
bool readMatrix( mat& matrix, Config& cfg, std::string filename );
bool readVector( colvec& vector, Setting& stgs, std::string filename );
bool readMatrix( mat& matrix, Setting& stgs, std::string filename );
bool readPriors( std::vector<pdf1d*>& priors, Setting& stgs);
#endif // header

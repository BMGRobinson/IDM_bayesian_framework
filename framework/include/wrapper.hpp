#ifndef WRAPPER_HPP_
#define WRAPPER_HPP_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <mpi.h>
#include "config.hpp"
#include <functional>
#include "mcmc.hpp"
#include "pdf.hpp"
#include "samples.hpp"
#include "filters.hpp"
#include "statespace.hpp"
#include "tmcmc.hpp"
#include <armadillo>
#include <cmath>
#include <iomanip>      // std::setprecision
#include <functional>
#include "bayesianPosterior.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <dlfcn.h> //to load dynamic library
#include "optimization.hpp"
#pragma GCC diagnostic pop
using namespace arma;

/* typedef for function */
typedef std::function<colvec(const colvec & , const colvec &, const double , const double )> genModelFunc;

/* Helper methods to read config file */
bool readconfig(Config &cfg,  const char * filename,  std::vector<proposedModels>& propModels, int id);
bool readdataconfig(Config &cfg,  const char * filename, std::vector<genModelParam>& genModels, int id);

bool checkInput( int argc, int id);
bool getStateEstimator(state_estimator *&se, statespace &ss, Gaussian *&is,  const mat &modelMeasVariance,  proposedModels &model, const MPI::Intracomm& statecom,  int id);
double doParameterEstimationAnalytical2D(void *handle , proposedModels &model , const MPI::Intracomm& statecom, int rootid, double xl, double xr, int Nx, double yl, double yr, int Ny );

//int divideWork(std::vector<proposedModels>& propModelVector, MPI::Intracomm& group);
int divideWork(std::vector<proposedModels>& propModelVector, MPI_Comm & StateEstimatorCom, MPI_Comm & headnodescom );
int getNumberOfRequiredProcs(std::vector<proposedModels>& propModelVector);
double doMCMC(bayesianPosterior& bp , proposedModels &model , const MPI_Comm& statecom, const MPI_Comm& mcmccom, int chainId);
bool mcmcSetUp(bayesianPosterior& bp, mcmc& mychain, proposedModels &model, const MPI_Comm& statecom, const MPI_Comm& mcmccom, const int chainId);

//Transitional MCMC
double doTMCMC(bayesianPosterior& bp , proposedModels &model , const MPI_Comm& statecom, const MPI_Comm& mcmccom, int chainId);
bool tmcmcSetUp(bayesianPosterior& bp, tmcmc& mytmcmc, proposedModels &model, const MPI_Comm& statecom, const MPI_Comm& mcmccom, const int chainId);


double doParameterEstimationSingleChain(void *handle, proposedModels &model , const MPI::Intracomm& com, int rootid );
//double doStateEstimation(void *handle , proposedModels &model , const MPI::Intracomm& statecom, int rootid);
double doStateEstimation(const mat& data, state_estimator & se , proposedModels &model , const MPI_Comm& statecom, int id);
//double doEvidenceEstimation(void *handle , proposedModels &model , const MPI::Intracomm& statecom, int rootid);
double doEvidenceEstimation(bayesianPosterior& bp , proposedModels &model , const MPI_Comm& statecom, const MPI_Comm& mcmccom, int chainId);
bool optimize(bayesianPosterior& bp , proposedModels &model , const MPI::Intracomm& statecom, int rootid);
void doStateEstimationError(const mat& data, state_estimator & se , proposedModels &model , const MPI_Comm& mcmccom, const MPI_Comm& statecom, int id);
//Generate data function
bool generateData( void *handle , genModelParam &model, int id );

//Main wrapper method
bool wrapper( const char * filename );
bool wrapperdata( const char * filename );

#endif

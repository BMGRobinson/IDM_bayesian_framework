#include "../SOURCE/mcmc.hpp"
#include "../SOURCE/pdf.hpp"
#include "../SOURCE/samples.hpp"
#include "../SOURCE/filters.hpp"
#include "../SOURCE/statespace.hpp"
#include <armadillo>
#include <cmath> 
#include <iomanip>      // std::setprecision
#include <functional>
#include "../SOURCE/config.hpp"
#include "../SOURCE/bayesianPosterior.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h> //to load dynamic library
/* models for the problem */
//typedef std::function<colvec(const colvec & , const colvec &, const double , const double )> genModelFunc;
//typedef std::function<double(const colvec &)> logLikelihoodFunc; //std::function for loglikelihood

/* models for the problem */
//#include "proposedModels.hpp"
using namespace arma;

int main(int argc, char* argv[]) { 
	
	/*Armadillo error output to my_log.txt*/
	std::ofstream f("my_log.txt");
	set_stream_err2(f);

	/* MPI */
	MPI::Init ( argc, argv );
	int num_procs = MPI::COMM_WORLD.Get_size ( );
	int id = MPI::COMM_WORLD.Get_rank ( );
	
	/* Set the seed to a random value */ 
	arma_rng::set_seed_random();
	
	/*
	*	Checking the arguments passed to the software
	*
	*/
	bool abort = false;
	if (id == 0) {
		if (argc != 2) {
			std::cout << "Generate data usage:" << std::endl;
			std::cout << "mpirun -np 1 ./run.out configFile" << std::endl;
			std::cout << "Where" << std::endl;
			std::cout << "configFile : name of the config file" << std::endl;
			abort = true;
		}
	}

	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return 0;
	} 

	
	/* Open the library object. Quit if can load it */
	void    *handle; //Handle to the functions
	handle = dlopen("./libf.so", RTLD_LOCAL | RTLD_LAZY);
	if (!handle) {
		std::cout << "Cannot load library: " << dlerror() << std::endl;
		abort = true;
	}
	
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return 0;
	} 
	
	mat modelMeasVariance;
	
	if (id == 0) {
		std::cout << "=======================================================================" << std::endl;
		std::cout << "===================== Reading configuration file ======================" << std::endl << std::endl;
	}
	Config cfg;
	
	if (!openConfigFile(cfg, argv[1])) {
		std::cout << "Could not read config file " << argv[1] << std::endl;
		return 0;
	}
	if (id == 0) { std::cout << "Configuration file used : " << argv[1] << std::endl << std::endl; }
	std::vector<proposedModels> propModels;
	
	/* Getting configuration in generatingModels vector */
	bool status = getProposedModelParameters( cfg, propModels );
	
	if (!status) {
		std::cout << "Error in getting the generating model paramters" << std::endl;
		return 0;
	}
	
	int npm = propModels.size();
	if (id == 0) { std::cout << std::endl << std::endl << "Number of proposed models: " << npm << std::endl; }
	
	std::string path;
	std::string dataFileName;
	statespace ss;
	Gaussian * is;
	state_estimator* se;
	const char* dlsym_error;
	
	//For each proposed model with the run flag set to TRUE, perform model selection. Root assign a model to run.
	
	for (int i = 0; i < npm; i ++) {
				path = "./" + propModels[i].folder + "/";
				dataFileName = propModels[i].data;
				//std::cout << "Using data file " << dataFileName << " in " << path << std::endl;		
				
				/* Load the measurements and the variance */
				mat data;
				data.load(path + "/" + dataFileName);
				
				if (data.n_rows == 0) {
					std::cout << "Error" << std::endl;
					std::cout << "Could not load the data file " << dataFileName << " in " << path << std::endl;
					break;
				}
				
				modelMeasVariance.load(path + "/" + propModels[i].cov,raw_ascii); 

				/* Get the statespace */
				ss = *(statespace *)dlsym(handle, propModels[i].function_handle.c_str());
				dlsym_error = dlerror();
				if (dlsym_error) {
					std::cerr << "Cannot load symbol create: " << dlsym_error << '\n';
					return 1;
				}
				
				ss.setDt( propModels[i].dt );
				//Construct state estimator
				
				if ( propModels[i].state_estimator == "ekf" ) {
					//std::cout << "State estimation using EKF" << std::endl;
					/* Construct initial state, for now always Gaussian */
					is = new Gaussian(propModels[i].initialState, propModels[i].initialStateVariance  );
					se = new Ekf( is ,ss, propModels[i].modelCov, modelMeasVariance );
				} else if ( propModels[i].state_estimator == "deterministic" ) {
					se = new Deterministic(propModels[i].initialState, ss);
				} else if ( propModels[i].state_estimator == "pf" ) {
					//std::cout << "State estimation using PF" << std::endl;
					se = new PF( propModels[i].initialState, propModels[i].initialStateVariance ,50,ss, modelMeasVariance );
				} else {
					std::cout << "State estimation method not recognized." << std::endl;
					break;
				}

				/* Do state estimation at MAP */
				se->saveToFile(true, path + "/" + propModels[i].function_handle + "_state_estimation.dat" );
		} else {
			//do nothing
		}
		
	}
	MPI::Finalize();
	return 0;
}

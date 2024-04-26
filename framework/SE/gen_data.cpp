#include <armadillo>
#include <cmath> 
#include "generatingModels.hpp"
#include <iomanip>      // std::setprecision
#include <map>
#include <functional>
#include "../SOURCE/config.hpp"
 #include <sys/stat.h>
 #include <sys/types.h>
/* models for the problem */
using namespace arma;
typedef std::function<colvec(const colvec & , const colvec &, const double , const double )> genModelFunc;


//mat generateData(colvec (*func)(const colvec & parameters , const colvec & state, const double time, const double dt),  const colvec & parameters, 
//	const colvec & initialState, const double timeEnd, const double dt,  const double dty, const double NSR,std::string path, std::string filename ) {
bool generateData(genModelFunc func, const colvec & parameters, const colvec & initialState, const double timeEnd, const double dt,  const double dty, const double NSR,std::string path, std::string filename ) {
	int Nrows  = initialState.n_rows+1; //Extra row for random forcing
	double _var = 0.0;
	double _mean = 0.0;
	colvec state = zeros<colvec>(Nrows);
	/*
	* Initial conditions
	*/

	for (int l = 0; l < Nrows-1; l++) {
		state[l] = initialState[l];  //Copy initial state. The last row is not copied since it will contain the random forcing	
	}
	
	int j = 0;
	int i = 0;
	int s = 0;
	long double time = 0.0;
	int m = int(ceil(timeEnd / dty));
	int n = int(ceil(timeEnd/ dt));
	double temp;
	double sample = 0.0;
	mat data = zeros<mat>(2,m);
	mat signal = zeros<mat>(Nrows+1,n); //One extra row for the time, one extra row for the random forcing
	std::cout << "Generating " << m << " data points." << std::endl;
	std::cout << "Number of timesteps " << n << std::endl;
	

	while (i < n) {
		//std::cout << std::setprecision(9) << time << std::endl;
		//std::cout << "Value of i is " << i << " and time is " << time << std::endl;
		//std::cout << std::setprecision(16) << "Value of time < timeEnd (" << time << "," << timeEnd << ")  is " <<  (time < timeEnd) << std::endl;
		
		for (s = 0; s < Nrows; s++) {
			signal(s,i) = state[s]; //The state now contains the random forcing
		}
		signal(Nrows,i) = time;

		//Compute the mean and variance
		_mean += state[0];
		_var += state[0]*state[0];
		sample += 1.0;

		//Forecast the state
		state = func(parameters, state, time, dt);
		time += dt;
		//Do we have a measurement then?
		temp = time/dty;
		//Rework this solution
		//std::cout << "Time = " << time << ", dty = " << dty << ", fmod = " << fabsf(roundf(temp) - temp) <<  " ... " << (fabsf(roundf(temp) - temp) < 1e-10) << std::endl;
		if (std::abs(roundf(temp) - temp) < 1e-10) {
			//if (modf(time/dty, &temp) == 0.0 ) { //need to check this behaviour
			data(0,j) = state[0];// + std::sqrt(measVariance) * randn(); //(noise is added later)
			//std::cout << "Signal at that time is " << data(0,j) << std::endl;
			data(1,j) = time;
			j ++;
		}
		i++;
	}
	//std::cout << "Temp var is " << _var << " and temp mean is " << _mean << std::endl;
	//Based on the signal to noise ratio, compute the measurement variance error. The variance of the signal is 
	_var = 1.0/(sample-1.0) * (_var - _mean*_mean/sample);
	//Equivalent
	//_var = 1.0/(sample-1.0) * (_var - 2.0*_mean*_mean/sample + _mean*_mean/(sample*sample));
	//std::cout << "Signal variance is " << _var << std::endl;
	double measVariance = NSR*_var;
	mat _VARIANCE = {measVariance};
	//Generate the measurements
	for (j = 0; j < m; j++) {
		data(0,j) += std::sqrt(measVariance)*randn();
	}
	std::cout << "True signal saved as " << path << "true.dat" << std::endl;
	signal.save(path + "true.dat" , raw_ascii);
	std::cout << "Measurment variance is " << measVariance << " saved in " << path << "variance.dat" << std::endl;
	_VARIANCE.save(path + "variance.dat", raw_ascii);
	/* Record the variance measurement */

	//Saving the data
	std::cout << "Measurments saved in " << path << "data.dat" << std::endl;
	data.save(path + "data.dat", raw_ascii);
	return true;

}

int main(int argc, char* argv[]) { 
	
	if (argc != 2) {
		std::cout << "Generate data usage:" << std::endl;
		std::cout << "mpirun -np 1 ./generate_data.out configFile" << std::endl;
		std::cout << "Where" << std::endl;
		std::cout << "configFile : name of the config file" << std::endl;
		return 0;
	}

	//To check if the folder exists or not
	struct stat st;
	
	/* Building the MAP between function handles */
	typedef std::function<colvec(const colvec & , const colvec &, const double , const double )> genModelFunc;
	std::map<std::string, genModelFunc> func_map;
	
	/* Adding each model in the map */
	func_map.insert(std::make_pair("generating_model_1",generating_model_1));
	func_map.insert(std::make_pair("generating_model_2",generating_model_2));
	func_map.insert(std::make_pair("generating_model_3",generating_model_3));
	func_map.insert(std::make_pair("generating_model_4",generating_model_4));
	
	
	
	
	
	std::cout << "=======================================================================" << std::endl;
	std::cout << "===================== Reading configuration file ======================" << std::endl;
	Config cfg;
	
	if (!openConfigFile(cfg, argv[1])) {
		std::cout << "Could not read config file " << argv[1] << std::endl;
		return 0;
	}
	std::cout << "Configuration file : " << argv[1] << std::endl;
	std::vector<genModelParam> genModels;
	
	/* Getting configuration in generatingModels vector */
	bool status = getGeneratingModelParameters( cfg, genModels );
	
	if (!status) {
		std::cout << "Error in getting the generating model paramters" << std::endl;
		return 0;
	}
	
	int ngm = genModels.size();
	std::cout << "There are " << ngm << " generating models found." << std::endl;
	
	std::string path;
	std::string dataFileName;
	
	/* Set the seed to a random value */ 
	arma_rng::set_seed_random();

	for (int i = 0; i < ngm; i ++) {
		std::cout << std::endl << "=======================================================================" << std::endl;
		std::cout << "=============== Generating data for " << genModels[i].function_name << " ================" << std::endl;
		std::cout << "Function handle " << genModels[i].function_handle << std::endl;
		path = "./" + genModels[i].folder + "/";
		
		//If path doesn't exist, create it
		//st = {0};
		//if (stat(path.c_str(), &st) == -1) {
		//	mkdir(path.c_str(), 0700);
		//}

		
		dataFileName = genModels[i].function_handle + ".dat";
		std::cout << "Saving data file " << dataFileName << " in " << path << std::endl;		
		if (genModels[i].dt > genModels[i].time) {
		std::cout << "ABORTING ---> Time should be greater than dt" << std::endl;
		return 0;
		}	

		if (genModels[i].dt > genModels[i].dty) {
			std::cout << "ABORTING ---> dt should be greater than dty" << std::endl;
			return 0;
		}

		if (genModels[i].NSR <= 0) {
			std::cout << "ABORTING ---> NSR should be greater than 0" << std::endl;
			return 0;
		}
		
		generateData(func_map.find(genModels[i].function_handle)->second, genModels[i].initialParameters, genModels[i].initialState, genModels[i].time, genModels[i].dt, genModels[i].dty, genModels[i].NSR, path, dataFileName);
	}

	/*
	std::cout << "=======================================================================" << std::endl;
	std::cout << "=========== Loading true parameters for generating model 1 ============" << std::endl;
	std::cout << "---> Looking in file gen_model_1_param.dat" << std::endl;
	
	colvec trueP;
	status = trueP.load(path + "gen_model_1_param.dat", raw_ascii);
	if (!status) {
		std::cout << "ABORTING ---> Error loading file gen_model_1_param.dat" << std::endl;
		return 0;
	}

	if (trueP.n_rows  != gen_model1_parameters) {
		std::cout << "ABORTING ---> Incorrect number of parameters. " << trueP.n_rows << " were supplied but " << gen_model1_parameters << " are required." << std::endl;
		return 0;
	}

	std::cout << "Mass (m):" << trueP[0] << std::endl;
	std::cout << "Damping coefficient  (c):" << trueP[1] << std::endl;
	std::cout << "Stiffness (k):" << trueP[2] << std::endl;
	std::cout << "Random force coefficient (sigma):" << trueP[3] << std::endl;
	std::cout << "Harmonic loading amplitude  (A):" << trueP[4] << std::endl;
	std::cout << "Harmonic loading freq in Hz (f):" << trueP[5] << std::endl;
	std::cout << std::endl;

	std::cout << "=========== Loading initial state for generating model 1 ============" << std::endl;
	std::cout << "---> Looking in file gen_model_1_state.dat" << std::endl;
	
	colvec initialState;
	status = initialState.load(path + "gen_model_1_state.dat", raw_ascii);
	if (!status) {
		std::cout << "ABORTING ---> Error loading file gen_model_1_state.dat" << std::endl;
	}

	if (initialState.n_rows  != gen_model1_state_size) {
		std::cout << "ABORTING ---> Incorrect state dimension required. " << initialState.n_rows << " were supplied but " << gen_model1_state_size << " are required." << std::endl;
	}

	std::cout << "Initial displacement :" << initialState[0] << std::endl;
	std::cout << "Initial velocity :" << initialState[1] << std::endl;

	std::cout << std::endl << "=========== Generating the data using generating model 1 ============" << std::endl;
	std::cout << "Simulation time (T):" << T << std::endl;
	std::cout << "Simulation time step (dt):" << dt << std::endl;
	std::cout << "Measurements time step (dty):" << dty << std::endl;
	std::cout << "Noise to Signal Ratio (NSR):" << NSR << std::endl;
	
	generateData( generating_model_1, trueP, initialState, T, dt, dty, NSR,path, "generating_model_1.dat");
*/



/******************************************************************************************************
*******************************************************************************************************
*******************************************************************************************************
*								Generating model 2
*******************************************************************************************************
******************************************************************************************************/
/*
	std::cout << "=======================================================================" << std::endl;
	std::cout << "=========== Loading true parameters for generating model 2 ============" << std::endl;
	std::cout << "---> Looking in file gen_model_2_param.dat" << std::endl;
	
	status = trueP.load(path + "gen_model_2_param.dat", raw_ascii);
	if (!status) {
		std::cout << "ABORTING ---> Error loading file gen_model_2_param.dat" << std::endl;
		return 0;
	}

	if (trueP.n_rows  != gen_model2_parameters) {
		std::cout << "ABORTING ---> Incorrect number of parameters. " << trueP.n_rows << " were supplied but " << gen_model1_parameters << " are required." << std::endl;
		return 0;
	}

	std::cout << "Mass (m):" << trueP[0] << std::endl;
	std::cout << "Damping coefficient  (c):" << trueP[1] << std::endl;
	std::cout << "Stiffness (k):" << trueP[2] << std::endl;
	std::cout << "Damping for colored noise SDE (b2) :" << trueP[3] << std::endl;
	std::cout << "Stiffness for colored noise SDE (b1) :" << trueP[4] << std::endl;
	std::cout << "Random force coefficient for colored noise SDE (b0) :" << trueP[5] << std::endl;
	std::cout << "Random force coefficient (s):" << trueP[6] << std::endl;
	std::cout << "Harmonic loading amplitude  (A):" << trueP[7] << std::endl;
	std::cout << "Harmonic loading freq in Hz (f):" << trueP[8] << std::endl;
	std::cout << std::endl;

	std::cout << "=========== Loading initial state for generating model 2 ============" << std::endl;
	std::cout << "---> Looking in file gen_model_2_state.dat" << std::endl;
	
	status = initialState.load(path + "gen_model_2_state.dat", raw_ascii);
	if (!status) {
		std::cout << "ABORTING ---> Error loading file gen_model_2_state.dat" << std::endl;
	}

	if (initialState.n_rows  != gen_model2_state_size) {
		std::cout << "ABORTING ---> Incorrect state dimension required. " << initialState.n_rows << " were supplied but " << gen_model2_state_size << " are required." << std::endl;
	}

	std::cout << "Initial displacement :" << initialState[0] << std::endl;
	std::cout << "Initial velocity :" << initialState[1] << std::endl;
	std::cout << "Initial eta :" << initialState[2] << std::endl;
	std::cout << "Initial eta velocity :" << initialState[3] << std::endl;

	std::cout << std::endl << "=========== Generating the data using generating model 2 ============" << std::endl;
	std::cout << "Simulation time (T):" << T << std::endl;
	std::cout << "Simulation time step (dt):" << dt << std::endl;
	std::cout << "Measurements time step (dty):" << dty << std::endl;
	std::cout << "Noise to Signal Ratio (NSR):" << NSR << std::endl;
	
	generateData( generating_model_2, trueP, initialState, T, dt, dty, NSR, path, "generating_model_2.dat");



	std::cout << std::endl << "=========== Generating log file ============" << std::endl;
	std::string logfile;
	

*/


}
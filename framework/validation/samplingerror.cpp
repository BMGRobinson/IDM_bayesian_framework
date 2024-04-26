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

/* models for the problem */
#include "models.hpp"
using namespace arma;


int main() {
	/* Set the seed to a random value */ 
	arma_rng::set_seed_random();

	int NParticles[] = {10,20,50,100,200,400,500,1000,2000,4000,5000,10000,20000,50000,100000};

	mat Q = {{0.0,0.0},{0.0,1.0}};
	mat R = {0.001};

	/* Load the data */
	mat data;
	data.load("data.dat");

	colvec u = {0.0, 0.0};
	mat P = {{2.0,1.0},{1.0,2.0}};

		/* Initial Gaussian state */
	Gaussian * is = new Gaussian(u, P );
		
	state_estimator * se_ekf = new Ekf( is ,ss_ekf, Q, R );
	state_estimator * se_pf; 
	state_estimator * se_enkf = new Enkf( u, P, 500, ss_pf, H, R );

	bayesianPosterior bp_ekf = bayesianPosterior(data, se_ekf);
	bayesianPosterior bp_enkf = bayesianPosterior(data, se_enkf);
	bayesianPosterior bp_pf;

	wall_clock timer;
	

	std::cout << "Likelihood:" << std::endl;
	std::cout << "EKF: " << bp_ekf.evaluate() << std::endl;
	timer.tic();
	std::cout << "Enkf: " << bp_enkf.evaluate() << std::endl;
	std::cout << "Required time: "	<< 	timer.toc()/60.0 << " minutes." << std::endl;
	std::cout << "Required time per particles : "	<< 	timer.toc()/60.0/500.0 << " minutes." << std::endl;
	double temp;
	double temptime;
	int repetitions = 10;
	mat results = mat(10, repetitions+1);
	mat timeMatrix = mat(10,repetitions+1);
	for (int i = 0; i < 10; i ++) {
		se_pf = new PF( u, P , NParticles[i] , ss_pf, R );
		bp_pf = bayesianPosterior(data, se_pf);
		//se_enkf = new Enkf( u, P, NParticles[i], ss_pf, H, R );
		//bp_enkf = bayesianPosterior(data, se_enkf);
		results(i,0) = NParticles[i];
		timeMatrix(i,0) = NParticles[i];
		for (int s = 0; s < repetitions; s++) {
			timer.tic();
			temp = bp_pf.evaluate();
			//temp = bp_enkf.evaluate();
			temptime = timer.toc();
			std::cout << "PF[" << NParticles[i] << "] :" << temp << std::endl;
			std::cout << "Required time: "	<< 	temptime/60.0 << " minutes." << std::endl;
			std::cout << "Required time per particles (sec) : "	<< 	temptime/double(NParticles[i]) << " seconds." << std::endl;
			results(i,s+1) = temp;
			timeMatrix(i,s+1) = temptime;
		}
		std::cout << "***********************" << std::endl << std::endl << std::endl;
	}
	
	results.save("pf_results.dat", raw_ascii);
	timeMatrix.save("pf_results_time.dat", raw_ascii);
}
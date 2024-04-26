#include <mcmc.hpp>
#include <pdf.hpp>
#include <samples.hpp>
#include <filters.hpp>
#include <statespace.hpp>
#include <wrapper.hpp>
#include <armadillo>
#include <cmath>
#include <iomanip>      // std::setprecision
#include <functional>
//#include <config.hpp>
#include <bayesianPosterior.hpp>
#include <sys/stat.h>
#include <sys/types.h>
//#include <dlfcn.h> //to load dynamic library

/* models for the problem */
#include "models.hpp"
using namespace arma;


int main() {
	/* Set the seed to a random value */
	arma_rng::set_seed_random();
	MPI::Init ();
	int id = MPI::COMM_WORLD.Get_rank ();
	mat Q = {{0.0,0.0},{0.0,1.0}};
	mat R = {0.001};

	/* Load the data */
	mat data;
	data.load("data.dat");

	colvec u = {0.0, 0.0};
	mat P = {{2.0,1.0},{1.0,2.0}};

		/* Initial Gaussian state */
	Gaussian * is = new Gaussian(u, P );


	//Need to set the following
	ss_ekf.setMeasCov( R );
	ss_ekf.setDt( 0.001 );
	//Measurements each 0.05 seconds
	ss_ekf.setForecastStepsBetweenMeasurements( 50 );

	ss_enkf.setMeasCov( R );
	ss_enkf.setDt( 0.001 );
	ss_enkf.setForecastStepsBetweenMeasurements( 50 );

	ss_pf.setMeasCov( R );
	ss_pf.setDt( 0.001 );
	ss_pf.setForecastStepsBetweenMeasurements( 50 );

	// CREATING THE FILTERS
	state_estimator * se_ekf = new Ekf( is ,ss_ekf, Q, R );
	state_estimator * se_enkf = new Enkf( u, P, 500, ss_enkf );
	state_estimator * se_pf;

	bayesianPosterior bp_ekf = bayesianPosterior(data, se_ekf);
	bayesianPosterior bp_enkf = bayesianPosterior(data, se_enkf);
	bayesianPosterior bp_pf;

	wall_clock timer;

	double trueLog;

	timer.tic();
	trueLog = bp_ekf.evaluate();
	double temptime;
	int repetitions = 100;


	//Results will be of the following format for EnKF, PF and PF-Enkf
	//Time (Average) Time (min) Time (max) Time (var)  Error (Average) Error(min) Error(max) Error (var) Particles Simulations

	int NParticles[] = {100,200,400,500,1000,2000,5000};
	int N = 7;

	mat results_enkf = mat(N, 10);
	mat results_pf	 = mat(N, 10);
	mat results_pfenkf	 = mat(N, 10);

	//ENKF

	 running_stat<double> timing;
	 running_stat<double> error;
	 double meanerror;
	for (int i = 0; i < N; i++) {
		results_enkf(i,8) = NParticles[i];
		results_enkf(i,9) =  repetitions;
		timing.reset();
		error.reset();

		//Recreate the statespace and filter
		se_enkf = new EnkfMPI( u, P, NParticles[i], ss_enkf, MPI_COMM_WORLD );
		bp_enkf = bayesianPosterior(data, se_enkf);
		if (id == 0) std::cout << "Running Enkf with " << NParticles[i] << " particles." << std::endl;
		for (int s = 0; s < repetitions; s++) {
			if (id == 0) timer.tic();
			meanerror = pow(bp_enkf.evaluate() - trueLog, 2.0);
			if (id == 0) error(meanerror);
			if (id == 0) temptime = timer.toc();
			if (id==0) timing(temptime/60.0);
		}
		if (id == 0) {
		results_enkf(i, 0) = timing.mean();
		results_enkf(i, 1) = timing.min();
		results_enkf(i, 2) = timing.max();
		results_enkf(i, 3) = timing.var();

		results_enkf(i, 4) = error.mean();
		results_enkf(i, 5) = error.min();
		results_enkf(i, 6) = error.max();
		results_enkf(i, 7) = error.var();
	}
	}
	if (id == 0) {
	results_enkf.save("enkf_results.dat", raw_ascii);
}


	for (int i = 0; i < N; i++) {
		results_pf(i,8) = NParticles[i];
		results_pf(i,9) =  repetitions;
		timing.reset();
		error.reset();

		//Recreate the statespace and filter
		se_pf = new PFMPI( u, P, NParticles[i], ss_pf ,  MPI_COMM_WORLD);
		bp_pf = bayesianPosterior(data, se_pf);
		if (id==0) std::cout << "Running Pf with " << NParticles[i] << " particles." << std::endl;
		for (int s = 0; s < repetitions; s++) {
			if (id==0) timer.tic();
			meanerror = pow(bp_pf.evaluate() - trueLog, 2.0);
			if (id==0) error(meanerror);
			if (id==0) temptime = timer.toc();
			if (id==0) timing(temptime/60.0);
		}
		if (id==0)  {
		results_pf(i, 0) = timing.mean();
		results_pf(i, 1) = timing.min();
		results_pf(i, 2) = timing.max();
		results_pf(i, 3) = timing.var();

		results_pf(i, 4) = error.mean();
		results_pf(i, 5) = error.min();
		results_pf(i, 6) = error.max();
		results_pf(i, 7) = error.var();
	 }
	}

	if (id==0) results_pf.save("pf_results.dat", raw_ascii);
	double tempRes;

	for (int i = 0; i < N; i++) {
		results_pfenkf(i,8) = NParticles[i];
		results_pfenkf(i,9) =  repetitions;
		timing.reset();
		error.reset();

		//Recreate the statespace and filter
		se_pf = new PFEnkfMPI( u, P, NParticles[i], ss_enkf , MPI_COMM_WORLD );
		bp_pf = bayesianPosterior(data, se_pf);
		if (id == 0) std::cout << "Running Pf Enkf with " << NParticles[i] << " particles." << std::endl;
		for (int s = 0; s < repetitions; s++) {
			if (id == 0)  timer.tic();
			tempRes = bp_pf.evaluate();
			meanerror = pow(tempRes - trueLog, 2.0);
			if (id == 0 ) std::cout << "Temp result is " << tempRes << std::endl;
			if (id == 0)  error(meanerror);
			if (id == 0)  temptime = timer.toc();
			if (id == 0)  timing(temptime/60.0);
		}
		if (id == 0) {
		results_pfenkf(i, 0) = timing.mean();
		results_pfenkf(i, 1) = timing.min();
		results_pfenkf(i, 2) = timing.max();
		results_pfenkf(i, 3) = timing.var();

		results_pfenkf(i, 4) = error.mean();
		results_pfenkf(i, 5) = error.min();
		results_pfenkf(i, 6) = error.max();
		results_pfenkf(i, 7) = error.var();
	}
	}
if (id == 0) {
	results_pfenkf.save("pfenkf_results_noresampling.dat", raw_ascii);
}
	MPI::Finalize();
}

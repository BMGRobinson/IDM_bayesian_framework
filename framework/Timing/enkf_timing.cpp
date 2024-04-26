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

colvec h = zeros<colvec>(1);

colvec _h( const colvec & state, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

colvec _f( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double m = 1.0;
	double c = 0.1;
	double k = 2.0;
	double amplitude = 60.0;
	double forceFreq = 1.0;

	colvec temp = state;
	double x1 = state[0];
	double x2 = state[1];
	double force = std::sqrt(dt)*0.1*randn();

	/* Explicit */
	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt * (-k/m*x1 - c/m * x2 + amplitude * cos(2.0 * datum::pi * forceFreq * time)) + force;
	return temp;
}

void _f2( colvec & state, const double dt, double time, const colvec & parameters ) {
	double m = 1.0;
	double c = 0.1;
	double k = 2.0;
	double amplitude = 60.0;
	double forceFreq = 1.0;

	//colvec temp = state;
	double x1 = state[0];
	double x2 = state[1];
	double force = std::sqrt(dt)*0.1*randn();

	/* Explicit */
	state[0] = x1 + dt * x2;
	state[1] = x2 + dt * (-k/m*x1 - c/m * x2 + amplitude * cos(2.0 * datum::pi * forceFreq * time)) + force;
	//return temp;
}


int main(int argc, char* argv[]) {

	/*Armadillo error output to my_log.txt*/
	std::ofstream f("my_log.txt");
	set_stream_err2(f);

	/* MPI */
	MPI::Init ( argc, argv );
	int num_procs = MPI::COMM_WORLD.Get_size ( );
	int id = MPI::COMM_WORLD.Get_rank ( );

	if (argc < 2) {
		std::cout << "Please input the number of particles" << std::endl;
		return 0;
	}

	int N = strtol(argv[1], NULL, 0); //Number particles



	arma_rng::set_seed_random();

 statespace ss = statespace(_f, _h, 0.0001);

 colvec::fixed<2> u = {2.0,3.0};
 mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
 mat::fixed<1,1> R = {0.5};
 mat::fixed<1,2> H = {1.0,0.0};

//Creating the EnKF filter
 Enkf myenkf = Enkf(u,P,N,ss,H,R);
 EnkfMPI myenkfMPI = EnkfMPI(u,P,N,ss,H,R, MPI::COMM_WORLD);
colvec voidVector;
colvec measurement = {1.0};

int trials = 10000;
double serialTime, serialUpdateTime;
double parTime, parUpdateTime;
int i = 0;
double n;
wall_clock timer;

colvec A,B, C;
B = {1.0, 2.0};
if (id == 0) {
	std::cout << "Timing forecast for _f" << std::endl;

timer.tic();
for (i = 0; i < trials; i ++) {
	A = _f( B, 0.1, 1.0, C );
}
n = timer.toc();
 serialTime = n / double(trials);
	std::cout << "Average time is " << serialTime << " seconds [" << trials << "]" << std::endl << std::endl << std::endl;
}

if (id == 0) {
	std::cout << "Timing forecast for _fV2" << std::endl;

timer.tic();
for (i = 0; i < trials; i ++) {
	_f2( B, 0.1, 1.0, C );
}
n = timer.toc();
 serialTime = n / double(trials);
	std::cout << "Average time is " << serialTime << " seconds [" << trials << "]" << std::endl << std::endl << std::endl;
}



 if (id == 0) {
	 std::cout << "Timing forecast for ENKF [Serial]" << std::endl;

 timer.tic();
 for (i = 0; i < trials; i ++) {
	 myenkf.forecast(voidVector);
 }
 n = timer.toc();
	serialTime = n / double(trials);
	 std::cout << "Average time is " << serialTime << " seconds [" << trials << "]" << std::endl << std::endl << std::endl;
 }

 if (id == 0) {
	 std::cout << "Timing update for ENKF [Serial]" << std::endl;

 timer.tic();
 for (i = 0; i < trials; i ++) {
	 myenkf.update(measurement, voidVector);
 }
 n = timer.toc();
	serialUpdateTime = n / double(trials);
	 std::cout << "Average time is " << serialUpdateTime << " seconds [" << trials << "]" << std::endl << std::endl << std::endl;
 }



  if (id == 0) {
 	 std::cout << "Timing forecast for ENKF [Parallel " << num_procs << " process]" << std::endl;
  	timer.tic();
	}

  for (i = 0; i < trials; i ++) {
 	 myenkfMPI.forecast(voidVector);
  }

	if (id == 0) {
  n = timer.toc();
	parTime = n / double(trials);
 	 std::cout << "Average time is " << parTime << " seconds [" << trials << "]" << std::endl;
	 std::cout << "Speedup Estimate: " << std::setprecision(3) << serialTime / parTime << std::endl << std::endl << std::endl;
  }

	if (id == 0) {
 	 std::cout << "Timing Update for ENKF [Parallel " << num_procs << " process]" << std::endl;
  	timer.tic();
	}

  for (i = 0; i < trials; i ++) {
 	 myenkfMPI.update(measurement, voidVector);
  }

	if (id == 0) {
  n = timer.toc();
	parUpdateTime = n / double(trials);
 	 std::cout << "Average time is " << parUpdateTime << " seconds [" << trials << "]" << std::endl;
	 std::cout << "Speedup Estimate: " << std::setprecision(3) << serialUpdateTime / parUpdateTime << std::endl;
  }



	MPI::Finalize();
	return 0;
}

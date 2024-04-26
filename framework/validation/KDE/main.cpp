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

using namespace arma;


int main() {
	/* Set the seed to a random value */
	arma_rng::set_seed_random();
	MPI::Init ();
	int id = MPI::COMM_WORLD.Get_rank ();

	//Creating fix samples to compare serial vs parallel implementation of KDE

	colvec mean = {1.0,2.0};
	mat cov = {{1.0,0.2},{0.2,0.5}};
	Gaussian g = Gaussian(mean, cov);
	int N = 5000;
	mat samples = zeros<mat>(2,N);

	if (id == 0) {
		for (int i = 0; i < N; i ++){
			samples.col(i) = g.sample();
		}
	}

	MPI::COMM_WORLD.Bcast(samples.memptr(), 2*N, MPI::DOUBLE, 0  );

	WeightedSamples serial = WeightedSamples(samples);


	//WeightedSamplesMPI par = WeightedSamplesMPI(mean, cov, N, MPI::COMM_WORLD );
	WeightedSamplesMPI par = WeightedSamplesMPI(samples, MPI::COMM_WORLD );

	mat trial = {{0.8,0.9,1.0,1.1},{1.8,1.9,2.3,2.1}};
	std::cout << serial.evaluate_kde(trial, false) << std::endl;

	std::cout << par.evaluate_kde(trial, false) << std::endl;

	MPI::Finalize();
}

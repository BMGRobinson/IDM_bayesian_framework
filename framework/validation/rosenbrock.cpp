#include "../SOURCE/mcmc.hpp"
#include "../SOURCE/pdf.hpp"
#include "../SOURCE/samples.hpp"
#include "../SOURCE/filters.hpp"
#include "../SOURCE/statespace.hpp"
#include <armadillo>
#include <cmath> 

using namespace arma;

/*
* Rosenbrock pdf, the MAP occurs at (1,1) (when a = 1, b = 100)
*/
double log_rosenbrock_pdf(const colvec & parameters) {
	double a = 1.0;
	double b = 100.0;
	double x = parameters[0];
	double y = parameters[1];
	return -((a-x)*(a-x)+b*(y-x*x)*(y-x*x))/20.0;
}


int main() {

	/* Set the seed to a random value */ 	
	//arma_rng::set_seed_random();
	arma_rng::set_seed(2);
	int NSAMPLES = 50000;


	std::cout.precision(11);
	std::cout.setf(ios::fixed);

	//initial conditions
	mat proposal = {{0.1,0.0},{0.0,0.1}};
	colvec initial = {0.0,20.0};

	//Metroplis-Hastings
	mcmc metropolis = mcmc(NSAMPLES, proposal, log_rosenbrock_pdf);
	metropolis.setSartingPoint( initial );
	metropolis.metropolisHastings();
	metropolis.print(".","rosenbrock_mh");


	//Adaptive proposal
	mcmc ap = mcmc(NSAMPLES, proposal, log_rosenbrock_pdf);
	ap.setSartingPoint( initial );
	//ap.setSD(1.0);
	ap.checkMinMAPUnchangedIterations(20);
	if ( ap.adaptiveProposal(1000) ) {
		ap.metropolisHastings();
		ap.print(".","rosenbrock_ap");
	}
	
	//Adaptive metropolis
	mcmc am = mcmc(NSAMPLES, proposal, log_rosenbrock_pdf);
	am.checkMinMAPUnchangedIterations(20);
	am.setSartingPoint( initial );
	am.adaptiveMetropolisburnin(3, 100, 100000);
	am.adaptiveMetropolis(100);
	am.print(".","rosenbrock_am");
	
	//DRAM
	mcmc dram = mcmc(NSAMPLES, proposal, log_rosenbrock_pdf);
	dram.checkMinMAPUnchangedIterations(20);
	dram.setSartingPoint( initial );
	dram.DRAMburnin(3, 100, 100000, 0.5, 0.1);
	dram.DRAM(100, 0.5, 0.1);
	dram.print(".","rosenbrock_dram");
	
	return 0;

}
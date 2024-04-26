#include <mcmc.hpp>
#include <pdf.hpp>
#include <samples.hpp>
#include <filters.hpp>
#include <statespace.hpp>
#include <simulate.hpp>
#include <wrapper.hpp>
#include <IQAgent.hpp>
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
#include "functions.cpp"
using namespace arma;


int main() {
	/* Set the seed to a random value */
	arma_rng::set_seed_random();
	MPI::Init ();

	int id = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);

	//Need to set the following
	model1uSSs.setDt(0.0001);
	model1qSScs.setDt(0.0001);
	model1qSSs.setDt(0.0001);
	model2qSSs.setDt(0.0001);


	wall_clock timer;


	timer.tic();
	double temptime;

	//unsigned N = 10000;
	unsigned N = 20000;
	//unsigned N = 1*5;
	unsigned m = 300000; //Number of timesteps
	unsigned qm = 1; //Steps between quantiles
	bool shiftSignals = false;
  /*
	std::cout << "Testing Quantiles" << std::endl;
	IQAgent myagent = IQAgent();
	for (int i = 0; i < 100000; i ++) {
		myagent.add( randn() );

		if (i % 10000 == 0) {
			std::cout << "	Quantiles with " << i << " samples." << std::endl;
			std::cout << "p = 0.05 -> " << myagent.report(0.05) << std::endl;
			std::cout << "p = 0.25 -> " << myagent.report(0.25) << std::endl;
			std::cout << "p = 0.50 -> " << myagent.report(0.50) << std::endl;
			std::cout << "p = 0.75 -> " << myagent.report(0.75) << std::endl;
			std::cout << "p = 0.95 -> " << myagent.report(0.95) << std::endl;
		}
	}
 */

	//Load the mcmc chain
	mat temp;
	mat ref;
	//Quasi-steady + white
	ref.load("true.dat", raw_ascii);
	mat chain;
	colvec ic;
	//Generate N signals
	//temp.load("Model-Unsteady-1-000.dat", raw_ascii);

if (id == 0) {
	colvec parameters = {-1.25,-1.0, 100.0,-500.0, 0.2, 0.002};
	chain = repmat(parameters, 1, 2);
	ic = {0.0,0.0,0.0,0.0};
	simulate( N, m, qm, ic, chain, ref,  model1uSSs, "true-1", shiftSignals, 0.0, MPI_COMM_WORLD );
} if (id == 1) {
	temp.load("Model-Quasi-Steady-1-000.dat", raw_ascii);
	chain = trans(temp.cols(1,5));

	ic = {0.0,0.0,0.0};
	simulate( N, m, qm, ic, chain, ref, model1qSSs, "quasi-steady-1" , shiftSignals, 0.0, MPI_COMM_WORLD );

	//Quasi-steady + colored
} if (id == 2) {
	temp.load("Model-Quasi-Steady-1-C-000.dat", raw_ascii);
	chain = trans(temp.cols(1,6));

	ic = {0.0,0.0,0.0,0.0};
	simulate( N, m, qm, ic, chain, ref, model1qSScs, "quasi-steady-1-C", shiftSignals, 0.0, MPI_COMM_WORLD );

	//Unsteady + White
} if (id == 3) {
	temp.load("Model-Unsteady-1-000.dat", raw_ascii);
	chain = trans(temp.cols(1,6));
	ic = {0.0,0.0,0.0,0.0};
	simulate( N, m, qm, ic, chain, ref,  model1uSSs,"unsteady-1", shiftSignals, 0.0, MPI_COMM_WORLD );

	//temptime = timer.toc();
	//std::cout << "Time taken is " << temptime << std::endl;
} if (id == 4) {
	//Model 3
	temp.load("Model-Quasi-Steady-2-000.dat", raw_ascii);
	chain = trans(temp.cols(1,6));

	ic = {0.0,0.0,0.0};
	simulate( N, m, qm, ic, chain, ref, model2qSSs, "quasi-steady-2" , shiftSignals, 0.0, MPI_COMM_WORLD );
}
	MPI::Finalize();
}

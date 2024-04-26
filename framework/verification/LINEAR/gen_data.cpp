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
#include <config.hpp>
#include <bayesianPosterior.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h> //to load dynamic library

using namespace arma;

int main(int argc, char* argv[]) {

	/* MPI */
	MPI::Init ( argc, argv );
	int id = MPI::COMM_WORLD.Get_rank ();

	/* Set the seed to a random value */
	arma_rng::set_seed_random();

	bool abort = checkInput(argc, id);
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return 0;
	}


	wrapperdata(argv[1]);

	MPI::Finalize();
	return 0;
}

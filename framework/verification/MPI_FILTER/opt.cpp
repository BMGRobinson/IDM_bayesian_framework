#include "../SOURCE/mcmc.hpp"
#include "../SOURCE/pdf.hpp"
#include "../SOURCE/samples.hpp"
#include "../SOURCE/filters.hpp"
#include "../SOURCE/statespace.hpp"
#include "../SOURCE/wrapper.hpp"
#include <armadillo>
#include <cmath>
#include <iomanip>      // std::setprecision
#include <functional>
#include "../SOURCE/config.hpp"
#include "../SOURCE/bayesianPosterior.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h> //to load dynamic library

using namespace arma;

int main(int argc, char* argv[]) {

	/* Variables */
	mat modelMeasVariance;
	Config cfg;
	std::vector<proposedModels> propModels;
	void *handle;
	MPI::Intracomm group;
	int index;
	bool abort;

	/*Armadillo error output to my_log.txt*/
	std::ofstream f("armadilloLog.txt");
	set_stream_err2(f);

	/* MPI */
	MPI::Init ( argc, argv );
	int num_procs = MPI::COMM_WORLD.Get_size ( );
	int id = MPI::COMM_WORLD.Get_rank ();

	/* Set the seed to a random value */
	arma_rng::set_seed_random();

	abort = checkInput(argc, id);
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return 0;
	}

	/* Open the library object. Quit if can load it */
	handle = dlopen("./libf.so", RTLD_LOCAL | RTLD_LAZY);
	if (!handle) {
		if (id == 0){ std::cout << "Cannot load library: " << dlerror() << std::endl; }
		abort = true;
	}

	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return 0;
	}

	abort = readconfig(cfg, argv[1], propModels, id);
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return 0;
	}

	//Here we divide the work
	index = divideWork( propModels, group);
	optimize(handle , propModels[index],  group, id);
	MPI::Finalize();
	return 0;
}

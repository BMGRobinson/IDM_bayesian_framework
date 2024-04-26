#include "wrapper.hpp"

bool checkInput( int argc, int id) {
	if (id == 0) {
		if (argc != 2) {
			std::cout << "Usage:" << std::endl;
			std::cout << "mpirun -np NP ./run.out configFile" << std::endl;
			std::cout << "mpirun -np NP ./run.out configFile" << std::endl;
			std::cout << "Where" << std::endl;
			std::cout << "NP : number of processes" << std::endl;
			std::cout << "configFile : name of the config file" << std::endl;
			return true;
		}
  }
  return false;
}

bool readconfig(Config &cfg,  const char * filename, std::vector<proposedModels>& propModels, int id) {
  bool abort = false;
  if (id == 0) {
		std::cout << "=======================================================================" << std::endl;
		std::cout << "===================== Reading configuration file ======================" << std::endl << std::endl;
	}

	if (!openConfigFile(cfg, filename)) {
		if (id == 0){
			std::cout << "Could not read config file " << filename << std::endl;
		}
		abort = true;
	}

  MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return abort; }

	if (id == 0) { std::cout << "Configuration file used : " << filename << std::endl; }

	/* Getting configuration in generatingModels vector */
	bool status = getProposedModelParameters( cfg, propModels, MPI::COMM_WORLD );

	if (!status) {
		if (id == 0){ std::cout << "Could not read config file " << filename << std::endl; }
		abort = true;
	}

	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return abort; }

	int npm = propModels.size();
	if (id == 0) { std::cout << "Number of proposed models: " << npm << std::endl; }

	if (id == 0 & npm == 0) {
		std::cout << "No proposed models. Stopping simulation." << std::endl;
		abort = true;
	}
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return abort; }

  return false;
}

bool readdataconfig(Config &cfg,  const char * filename, std::vector<genModelParam>& genModels, int id) {
  bool abort = false;
  if (id == 0) {
		std::cout << "=======================================================================" << std::endl;
		std::cout << "===================== Reading configuration file ======================" << std::endl << std::endl;
	}

	if (!openConfigFile(cfg, filename)) {
		if (id == 0){
			std::cout << "Could not read config file " << filename << std::endl;
		}
		abort = true;
	}

  MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return abort; }

	if (id == 0) { std::cout << "Configuration file used : " << filename << std::endl << std::endl; }

	/* Getting configuration in generatingModels vector */
	bool status = getGeneratingModelParameters( cfg, genModels );

	if (!status) {
		if (id == 0){ std::cout << "Could not read config file " << filename << std::endl; }
		abort = true;
	}

	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return abort; }

	int ngm = genModels.size();
	if (id == 0) { std::cout << std::endl << std::endl << "Number of generating models: " << ngm << std::endl; }

	if (id == 0 & ngm == 0) {
		abort = true;
	}
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return abort; }

	return false;
}

//This method divides modelCom into the required communicators
void assignWork(const int parState, const int nChains, MPI_Comm & StateEstimatorCom, MPI_Comm & headnodescom, const MPI_Comm & modelCom) {
  int requiredProc = parState*nChains; //Required number of processors for this model
  //There will be nChains subgroups, each containing requiredProc
  int id;
  MPI_Comm_rank(modelCom, &id);
  //Create communicator for state estimation. There will be perState number of colors
  //Each color corresponds to a chain
	int color = id % nChains;

  //MPI_Group headnodesgroup, worldgroup;
  MPI_Comm_split(modelCom, color, id, &StateEstimatorCom);
  //group = MPI::COMM_WORLD.Split(i, MPI::COMM_WORLD.Get_rank());
  int stateId;
  MPI_Comm_rank(StateEstimatorCom, &stateId);
  if (stateId == 0) {
    color = 0;
  } else {
    color = MPI_UNDEFINED;
  }
	MPI_Comm_split(modelCom, color, id, &headnodescom);
}

//Get the number of required processors
int getNumberOfRequiredProcs(std::vector<proposedModels>& propModelVector) {
	int npm = propModelVector.size();
	int procs = 0;
	for ( int i = 0; i < npm; i ++) {
			procs += propModelVector[i].nprocs*propModelVector[i].parallelGroups;
	}
	return procs;
}
//Divide number of process based on the number of models, number of process per filter and number of chains and total
int divideWork(std::vector<proposedModels>& propModelVector, MPI_Comm & StateEstimatorCom, MPI_Comm & headnodescom ) {
  int nrank = MPI::COMM_WORLD.Get_size();
  int myid = MPI::COMM_WORLD.Get_rank();
  int i;
  //Get the total number of process required
  int npm = propModelVector.size();
  int requiredProc = 0;
	int requiredProcs[npm];
	int requiredProcsCS[npm];

	int procs;
  for ( i = 0; i < npm; i ++) {
			procs = propModelVector[i].nprocs*propModelVector[i].parallelGroups;
    	requiredProcs[i] = procs;
    	requiredProc += procs;
    	requiredProcsCS[i] = requiredProc;
  }

  //There will be nChains subgroups, each containing requiredProc
  i = 0;

  while (myid >= requiredProcsCS[i]) {
    //std::cout << "My id is " << myid << " >= " << requiredProcsCS[i] << std::endl;
    i++;
  }
	//Need to divide proc per model
	MPI_Comm modelCom;
	MPI_Comm_split(MPI_COMM_WORLD, i, myid, &modelCom);
	assignWork(propModelVector[i].nprocs, propModelVector[i].parallelGroups , StateEstimatorCom, headnodescom, modelCom);
	return i;
}

bool optimize(bayesianPosterior& bp , proposedModels &model , const MPI::Intracomm& statecom, int rootid) {

  int id = statecom.Get_rank();
  int n = model.nparameters;
	std::string path = "./" + model.folder + "/";

  nelderMead opt = nelderMead(bp);
  opt.setStateEstimatorCom(statecom);
  mat points = zeros<mat>(n,n+1);
  for (int j = 0; j < n+1; j++) {
      for (int s = 0; s < n; s++) {
        points(s,j) = model.priors[s]->sample();
  		}
  }
	points.print("THE POINTS");
  if (id == 0) {
    points.print("Starting points");
    opt.print = true;
  }
  colvec point = opt.optimize( model.nelderMeadMaxIt, points );
  if (id == 0) {
      //point.print("Recommended starting point");
      point.save(path + "IC/start-"+ model.function_name +".dat", raw_ascii);
  }
	return true;
}

bool wrapperdata( const char * filename ) {
	mat modelMeasVariance;
	Config cfg;
	std::vector<genModelParam> genModels;
	void *handle;
	MPI::Intracomm group;
	int index;
	bool abort = false;
	int num_procs = MPI::COMM_WORLD.Get_size ( );
	int id = MPI::COMM_WORLD.Get_rank ();

	/* Set the seed to a random value */
	arma_rng::set_seed_random();


	/* Open the library object containing the functions. Quit if can load it */
	handle = dlopen("./libf.so", RTLD_LOCAL | RTLD_LAZY);
	if (!handle) {
		if (id == 0){
			std::cout << "Cannot load library: " << dlerror() << std::endl;
			std::cout << "Did you remember to compile them?" << std::endl;
		}
		abort = true;
	}

	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) {
		MPI::Finalize();
		return false;
	}

	abort = readdataconfig(cfg, filename, genModels, id);
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);

	if (abort) {
		MPI::Finalize();
		return false;
	}
	//Serial data generation only
	if (id == 0 ) {
		for (int i = 0; i < genModels.size(); i ++ ) {
			generateData(handle, genModels[i], id);
		}
	}
}

bool wrapper( const char * filename ) {
	Config cfg;
	std::vector<proposedModels> propModels;
	void *handle;
	const char* dlsym_error;
	int index;
	bool abort = false;
	int num_procs = MPI::COMM_WORLD.Get_size ( );
	int id = MPI::COMM_WORLD.Get_rank ();
	MPI_Comm statecom, mcmccom;
	/* Set the seed to a random value */
	arma_rng::set_seed_random();


	/* Open the library object containing the functions. Quit if can load it */
	handle = dlopen("./libf.so", RTLD_LOCAL | RTLD_LAZY);
	if (!handle) {
		if (id == 0){
			std::cout << "Cannot load library: " << dlerror() << std::endl;
			std::cout << "Did you remember to compile them?" << std::endl;
		}
		abort = true;
	}
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return false;}

	abort = readconfig(cfg, filename, propModels, id);
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return false;}
	//We abort if we don't have the correct number of cores available
	if (id == 0) {
		if (num_procs != getNumberOfRequiredProcs(propModels)) {
			std::cout << "Error: requires " << getNumberOfRequiredProcs(propModels) << " procs and " << num_procs << " available." << std::endl;
			abort = true;
		}
	}
	MPI::COMM_WORLD.Bcast(&abort, 1, MPI::BOOL, 0);
	if (abort) { return false;}

	//Here we divide the work based on the number of procs required by each models.
	index = divideWork( propModels, statecom, mcmccom);




	//Loading the data
	std::string path = "./" + propModels[index].folder + "/";
	std::string dataFileName = propModels[index].data;
	bayesianPosterior bp;
	Gaussian * is;
	state_estimator* se;

	/* Load the measurements and the variance */
	mat data, modelMeasVariance;
	data.load(path + "/" + dataFileName);

	if (data.n_rows > data.n_cols) {
		std::cout << "It seems that data points are stored in rows. Transposing the matrix" << std::endl;
		data = trans(data);
	}

	modelMeasVariance.load(path + "/" + propModels[index].cov,raw_ascii);

	//Loading the statespace
	auto temp = dlsym(handle, propModels[index].function_handle.c_str());
	dlsym_error = dlerror();

	if (dlsym_error) {
		std::cout << "Cannot load the statespace named " << propModels[index].function_handle << '\n';
		return false;
	}

	statespace ss = *(statespace *)temp;
	ss.setMeasCov( modelMeasVariance );
	ss.setDt( propModels[index].dt );
	ss.setForecastStepsBetweenMeasurements(propModels[index].fStepsBetweenMeasurements);

	int stateId;
  MPI_Comm_rank(statecom, &stateId);
	//std::cout << "I am id " << id << " and my index is " << index << " and my state id is " << stateId << std::endl;
	//Construct state estimator
	abort = getStateEstimator(se, ss, is, modelMeasVariance, propModels[index], statecom, stateId);

	//Construct the posterior function depending if deterministic or no
  bp = bayesianPosterior(data, se, propModels[index].priors );

	if (propModels[index].doOptimization) {
		optimize( bp , propModels[index] , statecom, stateId);
		return true; //TODO should this point be automatically used by MCMC ?
	}

	if (propModels[index].doStateEstimationRun)  {
		//doStateEstimation( ss , propModels[index] , group , id);
		doStateEstimation(data, *se, propModels[index], statecom, stateId  );
	}

  if (propModels[index].doStateEstimationError) {
		doStateEstimationError(data, *se , propModels[index], mcmccom, statecom, stateId);
  }

	//Staring main work. First do we perform parameter estimation ?
	//This gives MCMC parameters, MAP, and optimal proposal covariance
	//They are saved on the disk
	if (propModels[index].doParameterEstimation)  {
			if (propModels[index].mcmcMethod == "MCMC") {
				doMCMC( bp , propModels[index], statecom, mcmccom, index);
			} else if (propModels[index].mcmcMethod == "TMCMC") {
				doTMCMC( bp , propModels[index], statecom, mcmccom, index);
			} else {
				//should never be here. No parameter estimation!
			}
	}

	//Perform state estimation at MAP or other provided sample
	if (propModels[index].doEvidence)  {
		doEvidenceEstimation( bp , propModels[index], statecom, mcmccom, index);
	}
	return true;
}

double doParameterEstimationAnalytical2D(void *handle , proposedModels &model , const MPI::Intracomm& statecom, int rootid, double xl, double xr, int Nx, double yl, double yr, int Ny ) {
  std::string path = "./" + model.folder + "/";
  std::string dataFileName = model.data;
  bayesianPosterior bp;
  Gaussian * is;
  state_estimator* se;
  colvec evidence = colvec(1);
  const char* dlsym_error;
  bool abort = false;

  /* MPI */
  int id = statecom.Get_rank();

  /* Load the measurements and the variance */
  mat data, modelMeasVariance;
  data.load(path + "/" + dataFileName);

  if (data.n_rows == 0) {
    if (id == 0) {
      std::cout << "Error: "
      << "Could not load the data file " << dataFileName << " in " << path << std::endl;
    }
    return 1.0;
  }
  modelMeasVariance.load(path + "/" + model.cov,raw_ascii);

  /* Get the statespace */
  auto temp = dlsym(handle, model.function_handle.c_str());
  dlsym_error = dlerror();

  if (dlsym_error) {
    if (id == 0) {
      std::cout << "Cannot load the statespace named " << model.function_handle << '\n';
    }
    abort = true;
  }
  statecom.Bcast(&abort, 1, MPI::BOOL, 0);
  if (abort) { return 1.0; }

  statespace ss = *(statespace *)temp;
	ss.setMeasCov( modelMeasVariance );
  ss.setDt( model.dt );
  ss.setForecastStepsBetweenMeasurements(model.fStepsBetweenMeasurements);
  //Construct state estimator
  abort = getStateEstimator(se, ss, is, modelMeasVariance, model, statecom, id);

  statecom.Bcast(&abort, 1, MPI::BOOL, 0);
  if (abort) { return 1.0; }
  //Construct the posterior function depending if deterministic or no
  bp = bayesianPosterior(data, se, model.priors );

  //Need to clean that part. Only do after all process reach that point
  statecom.Barrier();
  if (id == 0) system("clear");
  statecom.Barrier();

  mat posterior = bp.posterior2D( Nx, xl, xr, Ny, yl, yr);

  posterior.save("posterior_pdf.dat", raw_ascii);
  return 0.0;
}

bool tmcmcSetUp(bayesianPosterior& bp, tmcmc& mytmcmc, proposedModels &model, const MPI_Comm& statecom, const MPI_Comm& mcmccom, const int chainId) {
	/* MPI */
  int stateId, mcmcId, stateSize, numChains;

  MPI_Comm_rank(statecom, &stateId);
	MPI_Comm_size(statecom, &stateSize);

	//The following communicator is only composed of the root nodes of state estimation comnmunicator
	if (stateId == 0) {
		MPI_Comm_rank(mcmccom, &mcmcId);
		MPI_Comm_size(mcmccom, &numChains);
	} else {
		mcmcId = -1;
	}


	if (stateId == 0) {
		struct stat st;
		std::string folderpath = "./" + model.folder + "/chains";
		stat(folderpath.c_str(), &st);
		//If folder doesn't exists
		if(!S_ISDIR(st.st_mode)) {
				mkdir(folderpath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
		}
	}

  if (mcmcId == 0) system("clear");

	//mychain = mcmc(model.mcmcParameters.nruns, , model.mcmcParameters.burnin_window,  model.mcmcParameters.initialProposal, bp  );

	//int nprocs;

	//bool diaginfo;

	/* Configuration of TMCMC chain */
		mytmcmc = tmcmc(model.tmcmcParameters.window, model.tmcmcParameters.dim, model.tmcmcParameters.cov, model.tmcmcParameters.cov_tol, bp , mcmccom );


		//We are in presence of parallel state estimation
		if (stateSize > 1) {
			//mytmcmc.setStateEstimatorCom(statecom);
		}
		if (stateId == 0) {
			//mytmcmc.setParallelChainsCom(mcmccom);
		}

		//mytmcmc.save_map( model.tmcmcParameters.save_map );
		//mytmcmc.save_proposal( model.tmcmcParameters.save_proposal );

	 //int firstConsoleLine = chainId*6+1;
	 //int firstConsoleLine = chainId+1;
	// mytmcmc.setPrintAtrow(firstConsoleLine);

	 mytmcmc.setPath("./" + model.folder, "chains", "evidence", model.function_name);
	return true;
}

bool mcmcSetUp(bayesianPosterior& bp, mcmc& mychain, proposedModels &model, const MPI_Comm& statecom, const MPI_Comm& mcmccom, const int chainId) {
	/* MPI */
  int stateId, mcmcId, stateSize, numChains;

  MPI_Comm_rank(statecom, &stateId);
	MPI_Comm_size(statecom, &stateSize);
	//The following communicator is only composed of the root nodes of state estimation comnmunicator
	if (stateId == 0) {
		MPI_Comm_rank(mcmccom, &mcmcId);
		MPI_Comm_size(mcmccom, &numChains);
	} else {
		mcmcId = -1;
	}


	if (stateId == 0) {
		struct stat st;
		std::string folderpath = "./" + model.folder + "/burnin";
		stat(folderpath.c_str(), &st);
		//If folder doesn't exists
		if(!S_ISDIR(st.st_mode)) {
				mkdir(folderpath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
		}

		folderpath = "./" + model.folder + "/chains";
		stat(folderpath.c_str(), &st);
		//If folder doesn't exists
		if(!S_ISDIR(st.st_mode)) {
				mkdir(folderpath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
		}
	}

  if (mcmcId == 0) system("clear");

	/* Configuration of MCMC chain */
	mychain = mcmc(model.mcmcParameters.nruns, model.mcmcParameters.window, model.mcmcParameters.burnin_window,  model.mcmcParameters.initialProposal, bp  );

	//We are in presence of parallel state estimation
	if (stateSize > 1) {
		mychain.setStateEstimatorCom(statecom);
	}
	if (stateId == 0) {
		mychain.setParallelChainsCom(mcmccom);
	}
	mychain.save_map( model.mcmcParameters.save_map );
	mychain.save_proposal( model.mcmcParameters.save_proposal );


	int dim = model.mcmcParameters.initialParameters.n_rows;
	colvec point = zeros<colvec>(dim);
	colvec randVec = zeros<colvec>(dim);
	std::vector<pdf1d*> priors = bp.getPriors();
	int d;

	bool validSample = false;
	int maxtries = 200;
	int it = 0;
	//Chain 0 will start at the specified point
	//Other chains will start from a point samples from the prior
	mat cholDec =  trans(chol(model.mcmcParameters.initialProposal));
	while (!validSample && it < maxtries) {
		if (mcmcId == 0) {
			point = model.mcmcParameters.initialParameters;
		} else {
			randVec.randn();
			point = model.mcmcParameters.initialParameters + cholDec * randVec;
		}
		//Try to evaluate the likelihood x prior at this point
		//returns true is can be evaluated (i.e. it's a number not -infinity)
		validSample = mychain.setStartingPoint( point );
		it ++;
	}

	if (!validSample) {
		if (stateId == 0) { std::cout << "[" << model.function_name << "] Can't evaluate the likelihood at the starting parameter - Exiting" << std::endl; }
		return false;
	}

 //int firstConsoleLine = chainId*6+1;
 int firstConsoleLine = chainId+1;
 mychain.setPrintAtrow(firstConsoleLine);

 /* MCMC convergence criteria */
 mychain.checkMinIterations(model.mcmcParameters.minIterations);
 mychain.checkMinMAPUnchangedIterations(model.mcmcParameters.minMAPNotChangedIterations);

 mychain.setBGR(model.mcmcParameters.rdet, model.mcmcParameters.rtrace);


	//Adaptation
	if (model.mcmcParameters.burnin_method == "AM") {
		mychain.setAM();
	} else if (model.mcmcParameters.burnin_method == "DRAM") {
		mychain.setDRAM(model.mcmcParameters.DRProb, model.mcmcParameters.DRScale);
	} else if (model.mcmcParameters.burnin_method == "AP") {
		mychain.setAP();
	}

	if (model.mcmcParameters.diaginfo) {
		mychain.enableDiagInfo();
	}

	mychain.setPath("./" + model.folder, "burnin", "chains", "evidence", model.function_name);
	return true;
}

double doTMCMC(bayesianPosterior& bp , proposedModels &model , const MPI_Comm& statecom, const MPI_Comm& mcmccom, int chainId) {
  tmcmc mytmcmc;
	bool success = tmcmcSetUp(bp, mytmcmc, model, statecom, mcmccom, chainId);
	mytmcmc.run();
	return 0.0;
}

double doMCMC(bayesianPosterior& bp , proposedModels &model , const MPI_Comm& statecom, const MPI_Comm& mcmccom, int chainId) {
  mcmc mychain;

	bool success = mcmcSetUp(bp, mychain, model, statecom, mcmccom, chainId);

	//Creating folders for burnin, chains

  if (model.mcmcParameters.AP_PreRuns > 0) {
		mychain.ap_preruns(model.mcmcParameters.AP_PreRuns);
  }

	if (model.mcmcParameters.burnin_method != "None") {
		mychain.burnin();
	}
	mychain.run();
	//Save the chain
	mychain.print();

	return 0.0;
}

double doEvidenceEstimation(bayesianPosterior& bp , proposedModels &model , const MPI_Comm& statecom, const MPI_Comm& mcmccom, int chainId) {
	//Create evidence folder
	int stateId;
  MPI_Comm_rank(statecom, &stateId);
	if (stateId == 0) {
		struct stat st;
		std::string folderpath = "./" + model.folder + "/evidence";
		stat(folderpath.c_str(), &st);
		//If folder doesn't exists
		if(!S_ISDIR(st.st_mode)) {
				mkdir(folderpath.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
		}
	}
	//Gauss-Hermite
	if (model.evidenceParameters.method == "GH" ) {
		long double logEvidence;
	  int order = model.evidenceParameters.quadLevel;
		int dim =  model.evidenceParameters.mu.n_rows;
		//int dim = covMatrix.n_rows;
		GaussHermite gh = GaussHermite(bp, order, dim);

		gh.setInitialMean(model.evidenceParameters.mu);
		gh.setInitialSigma(model.evidenceParameters.sigma);

		std::cout << "Starting mu " << std::endl << model.evidenceParameters.mu << std::endl;
		std::cout << "Starting sigma " << std::endl << model.evidenceParameters.sigma << std::endl;

		logEvidence = gh.quadrature(1.0e-10, 10000);
		std::cout << "Estimated value : " << logEvidence << std::endl;
		std::ofstream log_file("./" + model.folder + "/" + model.function_name + "-logevidence.dat", std::ios_base::out | std::ios_base::app );
		std::time_t mytime = std::time(nullptr);
		log_file << std::endl << "**********************************" << std::endl;
		log_file << "Simulation time : " << std::ctime(&mytime); // << std::endl;
		log_file << "log evidence (Gauss-Hermite quad) : " << logEvidence << std::endl;
		return 0.0;
  }

	//Chib-Jeliazkov
	mcmc mychain;
	bool success = mcmcSetUp(bp, mychain, model, statecom, mcmccom, chainId);
	mychain.load();

	long double logEvidence, logGoodnessOfFit, EIG;
	mychain.logEvidenceAtMAP(model.mcmcParameters.nruns * model.mcmcParameters.window/model.evidenceParameters.trim, logEvidence, logGoodnessOfFit, EIG, model.evidenceParameters.trim);
	/*
	std::ofstream log_file(path + "/" + model.function_name + "-logevidence.dat", std::ios_base::out | std::ios_base::app );
	std::time_t mytime = std::time(nullptr);
	log_file << std::endl << "**********************************" << std::endl;
	log_file << "Simulation time : " << std::ctime(&mytime); // << std::endl;
	log_file << "log evidence (Chib-Jeliazkov): " << logEvidence << std::endl;
	log_file << "log Goodness of fit : " << logGoodnessOfFit << std::endl;
	log_file << "log Expected Information Gain : " << EIG << std::endl;
	*/
}

double doStateEstimation(const mat& data, state_estimator & se , proposedModels &model , const MPI_Comm& statecom, int id) {

  wall_clock timer;
	running_stat<double> stats;
	timer.tic();
	int sruns = model.seruns;
	double l;

	for (int i = 0; i < sruns; i ++) {

		l = se.logLikelihood( data, model.parameters );
		if (id == 0) {
			std::cout << l << std::endl;
			stats(l);
		}
	}
	double n = timer.toc()/60.0;

 	//Save to file the last one
	se.saveToFile(true, "./" + model.folder + "/" + model.function_name + "-state-estimation", 1 );
	se.state_estimation( data, model.parameters , true );


	if (id == 0) {
		std::cout << " Averaging " << sruns << " simulations." << std::endl;
		std::cout << " Average value " << stats.mean() << std::endl;
		std::cout << " Variance:  " << stats.var() << std::endl;
		std::cout << " COV (%) " << 100.0*std::sqrt(stats.var())/stats.mean() << std::endl;
		std::cout << " Max  " << stats.max() << std::endl;
		std::cout << " Min  " << stats.min() << std::endl;
		std::cout << " Total time taken " << n << " minutes " << std::endl;
		std::cout << " Average time taken " << n*60.0/double(sruns) << " seconds" << std::endl;
	}
}

//Method that computes the error of a model. Currently only serial implementation
void doStateEstimationError(const mat& data, state_estimator & se , proposedModels &model , const MPI_Comm& mcmccom, const MPI_Comm& statecom, int id) {
	//To do get size of statespace, get number of repetitions
	//implement sampling strategy with or without replacement
	//Load the parameters
	int mcmcId, nprocs;
	MPI_Comm_rank(mcmccom, &mcmcId);
	MPI_Comm_size(mcmccom, &nprocs);
	mat result;
	mat reference;
	std::string mcmcIds = std::to_string(mcmcId);

	//Append "0" to the left
	if (mcmcIds.length() == 1) {
		mcmcIds = "00" + mcmcIds;
	} else if (mcmcIds.length() == 2) {
		mcmcIds = "0" + mcmcIds;
	}

	int nsize = 2; //Only do theta and theta dot. Not idea. Maybe a better idea woulld be to add an user defined error function
  int N = 20000;

	//Proc load matrix containing samples (including extra info)
	result.load("./" + model.folder + "/chains/"  + model.function_name + "-" + mcmcIds + ".dat",auto_detect);
	reference.load("./" + model.folder + "/true.dat");

	int length = result.n_rows;
	int dim = result.n_cols - 2;

	mat _samples = zeros<mat>(dim, length);

	//Copy the samples
	for (int j = 1; j < dim+1; j++) {
		_samples.row(j-1) = result.unsafe_col(j).t();
	}
	//Each proc has it own list
	//Samples parameterList = Samples( _samples );

	Samples parameterList( _samples );
	colvec errors = zeros<colvec>(nsize);
	colvec param;
	for (int i = 0; i < N; i ++) {
			param = parameterList.drawWithoutReplacement();
			errors += se.getSquareError(data, reference, param);
	}
	errors /= double(N);

	//Get the average errors

	if (id == 0) {
		if (mcmcId == 0) MPI_Reduce(MPI_IN_PLACE, errors.memptr() , nsize, MPI_DOUBLE, MPI_SUM, 0, mcmccom);
		if (mcmcId != 0) MPI_Reduce(errors.memptr(), errors.memptr() , nsize, MPI_DOUBLE, MPI_SUM, 0, mcmccom);
	}
	//mcmccom.Reduce(MPI::IN_PLACE, errors.memptr(), nsize, MPI::DOUBLE, MPI::SUM, 0);


	//Write the error
	if (mcmcId == 0) {
	errors /= double(nprocs);
	std::ofstream log_file("./Errors/Error_" + model.function_name + ".txt", std::ios_base::out | std::ios_base::trunc );
		log_file << "MSE for " << model.function_name << std::endl;
		//log_file << std::setw(15) << "Chain Id" << std::setw(15) << "Samples"  << std::setw(15) << "Reject Ratio" << std::endl;
		for (int s = 0; s < nsize; s ++) {
			log_file << std::setw(15) << errors[s] << std::endl;
		}
		log_file.close();
	}
}
//State estimator is a reference to a pointer
bool getStateEstimator(state_estimator *&se, statespace &ss, Gaussian *&is,  const mat &modelMeasVariance,  proposedModels &model, const MPI::Intracomm& statecom,  int id) {
  bool abort = false;
  if ( model.state_estimator == "ekf" ) {
    is = new Gaussian(model.initialState, model.initialStateVariance  );
    se = new Ekf( is ,ss, model.modelCov, modelMeasVariance );
  } else if ( model.state_estimator == "deterministic" ) {
    se = new Deterministic(model.initialState, ss);
  } else if ( model.state_estimator == "pf" ) {
    if (id == 0) {std::cout << "State estimation using Particle Filter." << std::endl;}
    se = new PF( model.initialState, model.initialStateVariance ,model.nparticles,ss );
	} else if ( model.state_estimator == "pfmpi" ) {
		if (id == 0) {std::cout << "State estimation using MPI Particle Filter." << std::endl;}
		se = new PFMPI( model.initialState, model.initialStateVariance ,model.nparticles, ss, statecom );
	} else if ( model.state_estimator == "enkf" ) {
    if (id == 0) {std::cout << "State estimation using Enkf." << std::endl;}
    se = new Enkf(model.initialState, model.initialStateVariance,model.nparticles,ss );
  } else if ( model.state_estimator == "enkfmpi" ) {
    if (id == 0) {std::cout << "State estimation using Enkf MPI." << std::endl;}
    se = new EnkfMPI(model.initialState, model.initialStateVariance,model.nparticles,ss, statecom );
	} else if ( model.state_estimator == "pfenkfmpi" ) {
		if (id == 0) {std::cout << "State estimation using PF-Enkf MPI." << std::endl;}
		se = new PFEnkfMPI(model.initialState, model.initialStateVariance,model.nparticles,ss, statecom );
  } else {
    if (id == 0) {std::cout << "State estimation method not recognized." << std::endl;
			std::cout << "Extended kalman filter (Serial) --> ekf" << std::endl;
			std::cout << "Ensemble kalman filter (Serial) --> enkf" << std::endl;
			std::cout << "Ensemble kalman fliter (MPI) --> enkfmpi" << std::endl;
			std::cout << "Particle filter (Serial) --> pf" << std::endl;
			std::cout << "Particle filter (MPI) --> pfmpi" << std::endl;
			std::cout << "Particle filter Enkf (MPI) --> pfenkfmpi" << std::endl;
			}
    abort = true;
  }
  return abort;
}

//So far only handles scalar measurements. Serial implementation.
bool generateData( void *handle , genModelParam &model, int id ) {
	colvec initialState = model.initialState;
	colvec parameters = model.parameters;
	int Nrows  = initialState.n_rows+1; //Extra row for random forcing
	double _var = 0.0;
	double _mean = 0.0;
	colvec state = zeros<colvec>(Nrows);
	for (int l = 0; l < Nrows-1; l++) {
		state[l] = initialState[l];  //Copy initial state. The last row is not copied since it will contain the random forcing
	}

	double dt = model.dt;
	double timeEnd = model.time;
	int stepsBetweenMeasurements = model.stepsBetweenMeasurements;
	double NSR = model.NSR;
	const char* dlsym_error;

	//Indices
	int j = 0;
	int i = 0;
	int s = 0;
	long double time = 0.0;
	int n = int(ceil(timeEnd/ dt));		 //Number of time steps
	int m = n/stepsBetweenMeasurements + 1;  //Number of measurements

	double sample = 0.0;
	mat data = zeros<mat>(1,m);
	mat dataTime = zeros<mat>(1,m);
	mat signal = zeros<mat>(Nrows+1,n+1); //One extra row for the time, one extra row for the random forcing
	std::cout << "Generating " << m << " data points." << std::endl << "Number of timesteps " << n << std::endl;


	//Loading function
	auto temp = dlsym(handle, model.function_handle.c_str());
  dlsym_error = dlerror();

  if (dlsym_error) {
      std::cout << "Cannot load the function named " << model.function_handle << '\n';
			return false;
  }

	genModelFunc func = *(colvec (*)(const colvec&, const colvec&, const double, const double))temp;

	while (i <= n) {
		for (s = 0; s < Nrows; s++) {
			signal(s,i) = state[s]; //The state now contains the random forcing
		}
		signal(Nrows,i) = time;

		//Compute the mean and variance of the signal to compute the Noise to Signal ratio
		_mean += state[0];
		_var += state[0]*state[0];
		sample += 1.0;

		if ((i % stepsBetweenMeasurements) == 0) { //Measurement at this time instant
			data(0,j) = state[0]; //Generating clean measurement of the state
			dataTime(0,j) = time;			//Recording the time at which the measurement is taken
			j ++;
		}

		//Forecast the state
		state = func(parameters, state, time, dt);
		time += dt;

		i++;
	}
	//Based on the signal to noise ratio, compute the measurement variance error. The variance of the signal is
	_var = 1.0/(sample-1.0) * (_var - _mean*_mean/sample);
	//Equivalent: _var = 1.0/(sample-1.0) * (_var - 2.0*_mean*_mean/sample + _mean*_mean/(sample*sample));

	double measVariance = NSR*_var; //Find the measurement error variance based on the specified Noise to Signal ratio.
	mat _VARIANCE = {measVariance};

	//Generate the noisy measurements
	for (j = 0; j < m; j++) {
		data(0,j) += std::sqrt(measVariance)*randn();
	}

	//If folder doesn't exist,  it is created.
	struct stat st;
	std::string path = "./" + model.folder;
	stat(path.c_str(), &st);
	//If folder doesn't exists
	if(!S_ISDIR(st.st_mode)) {
		mkdir(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
	}



	/** Saving the true signal, the measurement variance and the data */
	std::cout << "True signal saved as " << path << "/true.dat" << std::endl;
	signal.save(path + "/true.dat" , raw_ascii);
	std::cout << "Measurment variance is " << measVariance << " saved in " << path << "/variance.dat" << std::endl;
	_VARIANCE.save(path + "/variance.dat", raw_ascii);
	std::cout << "Measurments saved in " << path << "/data.dat" << std::endl;
	data.save(path + "/data.dat", raw_ascii);
	dataTime.save(path + "/timedata.dat", raw_ascii);
	return true;
}



// Generate data points
// Run state estimation using all models

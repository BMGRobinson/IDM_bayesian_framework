#include "config.hpp"
template <class V>
std::string convert(V val) {
	std::string stringval;          // string which will contain the result
	std::ostringstream conv;   // stream used for the conversion
	conv << val;
	stringval = conv.str(); // set 'Result' to the contents of the stream
	return stringval;
}

std::string convert(bool val) {
	if (val) {
		return "True";
	} else {
		return "False";
	}
}


void print(const std::string &text, const int id) {
	if (id == 0) {
		//std::cout << text << std::endl;
		std::ofstream log_file(
		"proposedmodels.txt", std::ios_base::out | std::ios_base::app );
		log_file << text << std::endl;
		log_file.close();
	}
}

void print(const std::string &text, const colvec & V, const int id) {
	if (id == 0) {
		//std::cout << text << std::endl << V << std::endl;
		std::ofstream log_file("proposedmodels.txt", std::ios_base::out | std::ios_base::app );
		log_file << text << " : ";

		for (int i = 0; i < V.n_rows; i ++) {
			log_file << V[i] << " ";
		}
		log_file << std::endl;
		log_file.close();
	}
}

void print(const std::string &text, const mat & M, const int id) {
	if (id == 0) {
		//std::cout << text << std::endl << M << std::endl;
		std::ofstream log_file( "proposedmodels.txt", std::ios_base::out | std::ios_base::app );
		log_file << text << " : " << std::endl;
		for (int i = 0; i < M.n_rows; i ++) {
			log_file << "\t\t";
			for (int j = 0; j < M.n_cols; j ++) {
				log_file << M(i,j) << " ";
			}
			log_file << std::endl;
		}
		log_file << std::endl;
		log_file.close();
	}
}

void padUpTo(std::string & str, const size_t num)
{
    if(num > str.size()) {
    	str.insert(str.size(), num - str.size(), '.');
		}
}


template <class T>
void getFlagWithDefault(Config& cfg, Setting& gm, std::string valueName, T& value, const T defaultValue, int procId  ) {
	std::string def = " specified.";

	//Check if it is in the global settings and of the correct type
	if (gm.exists(valueName) && gm.lookupValue(valueName, value) ) {
		//nothing to do
	} else if (cfg.exists(valueName) && cfg.lookupValue(valueName, value) ) { //Look into local settings
		//nothing to do
	} else {
		def = " not found or invalid type. Using default value.";
		value = defaultValue;
	}
	std::string str;
	str = "\t" + valueName + def;
	padUpTo(str, 80);
	print( str + convert(value) ,procId);
}
//Should throw an exception
template <class U>
bool getFlagRequired(Config& cfg, Setting& gm, std::string valueName, U& value, int procId  ) {
	bool ok = true;
	std::string def = " specified.";

	//Check if it is in the global settings and of the correct type
	if (gm.exists(valueName) && gm.lookupValue(valueName, value) ) {
		//nothing to do
	} else if (cfg.exists(valueName) && cfg.lookupValue(valueName, value) ) { //Look into local settings
		//nothing to do
	} else {
		def = " not found or bad type. ERROR. This value is required.";
		ok = false;
	}
	std::string str;
	str = "\t" + valueName + def;
	padUpTo(str, 80);
	print( str + convert(value) ,procId);
	return ok;
}

//String version otherwise, Template can't differentiate between string and char*
void getFlagWithDefault(Config& cfg, Setting& gm, std::string valueName, std::string& value, const std::string defaultValue, int procId  ) {
	std::string def = " specified.";

	//Check if it is in the global settings and of the correct type
	if (gm.exists(valueName) && gm.lookupValue(valueName, value) ) {
		//nothing to do
	} else if (cfg.exists(valueName) && cfg.lookupValue(valueName, value) ) { //Look into local settings
		//nothing to do
	} else {
		def = " not found or bad type. Using default value.";
		value = defaultValue;
	}
	std::string str;
	str = "\t" + valueName + def;
	padUpTo(str, 80);
	print( str +  "\"" + value + "\"" ,procId);
}
//Should throw an exception
bool getFlagRequired(Config& cfg, Setting& gm, std::string valueName, std::string& value, int procId  ) {
	bool ok = true;
	std::string def = " specified.";

	//Check if it is in the global settings and of the correct type
	if (gm.exists(valueName) && gm.lookupValue(valueName, value) ) {
		//nothing to do
	} else if (cfg.exists(valueName) && cfg.lookupValue(valueName, value) ) { //Look into local settings
		//nothing to do
	} else {
		def = " not found or bad type. *** ERROR *** This value is required.";
		ok = false;
	}
	std::string str;
	str = "\t" + valueName + def;
	padUpTo(str, 80);
	print( str + "\"" + value + "\"" ,procId);
	return ok;
}

bool getTMCMCParameters(Config& cfg, Setting & gms, tmcmcParam& _parameters, const MPI::Intracomm _com ) {
	int id = _com.Get_rank();
	bool flag;
	std::string localPath;
	colvec tempVector;

	//Diagnostic info
	getFlagWithDefault(cfg, gms, "Diagnostic", _parameters.diaginfo, false, id  );

	flag = getFlagRequired(cfg, gms, "dim", _parameters.dim, id  );

	//Number of samples
	getFlagWithDefault(cfg, gms, "window", _parameters.window, 5000, id  );

	getFlagWithDefault(cfg, gms, "COV", _parameters.cov , 1.0, id  );
	getFlagWithDefault(cfg, gms, "COV_TOL", _parameters.cov_tol , 1.0e-4, id  );

	return flag;
}

bool getMCMCParameters(Config& cfg, Setting & gms, mcmcParam& _parameters, const MPI::Intracomm _com ) {

	int id = _com.Get_rank();
	std::string localPath;
	colvec tempVector;

	//Diagnostic info
	getFlagWithDefault(cfg, gms, "Diagnostic", _parameters.diaginfo, false, id  );

	//Starting point
	if ( gms.exists("starting_point") ) {
		_parameters.initialParameters = getVector(gms, "starting_point", 100 );
		print("\tMCMC Starting point", _parameters.initialParameters, id );
	} else {
		print("\tError: MCMC Starting point not found.", id );
		return false;
	}

	//Initial Proposal Covariance Matrix for the Metropolis-Hastings algorithm
	double f;
	if ( gms.exists("initialProposal") ) {
		print("\tReading initial proposal covariance from configuration file" , id);
		_parameters.initialProposal = getMatrix(gms, "initialProposal", _parameters.initialParameters.n_rows );
		print("\tInitial covariance proposal", _parameters.initialProposal, id );
	} else if ( gms.exists("eye") ) {
		print("\tSetting initial MH proposal covariance to scaled Identity" , id);
		getFlagWithDefault(cfg, gms, "eye", f, 1.0, id  );
		_parameters.initialProposal = f * eye<mat>(_parameters.initialParameters.n_rows , _parameters.initialParameters.n_rows );
	} else {
		print("\tSetting initial MH proposal covariance to Identity" , id);
		_parameters.initialProposal = eye<mat>(_parameters.initialParameters.n_rows , _parameters.initialParameters.n_rows );
	}

	//Burn-in will be performed
	if (_parameters.burnin_method != "None") {
		getFlagWithDefault(cfg, gms, "minIterations", _parameters.minIterations, 10, id  );
		getFlagWithDefault(cfg, gms, "minMAPNotChangedIterations", _parameters.minMAPNotChangedIterations, 5, id  );
		getFlagWithDefault(cfg, gms, "maxSamplesUntilConvergence", _parameters.maxSamplesUntilConvergence, 100000, id  );
		//getFlagWithDefault(cfg, gms, "adaptationWindowLength", _parameters.adaptationWindowLength, 500, id  );
	}

	/*
	if (_parameters.burnin_method == "AM" || _parameters.burnin_method == "DRAM") {
		getFlagWithDefault(cfg, gms, "startAdaptationIndex", _parameters.startAdaptationIndex, 200, id  );
	}
	*/

	getFlagWithDefault(cfg, gms, "AP_PreRuns", _parameters.AP_PreRuns, 0, id  ); //Perform how many pre-runs
	getFlagWithDefault(cfg, gms, "method", _parameters.method, "MH", id  );
	getFlagWithDefault(cfg, gms, "burnin", _parameters.burnin_method, "None", id  );
	getFlagWithDefault(cfg, gms, "window", _parameters.window, 5000, id  );
	getFlagWithDefault(cfg, gms, "burnin_window", _parameters.burnin_window, 5000, id  );
	getFlagWithDefault(cfg, gms, "runs", _parameters.nruns, 10, id  );
	getFlagWithDefault(cfg, gms, "save_proposal", _parameters.save_proposal, true, id  );
	getFlagWithDefault(cfg, gms, "save_map", _parameters.save_map, true, id  );

	getFlagWithDefault(cfg, gms, "rdet", _parameters.rdet, 1.5, id  );
	getFlagWithDefault(cfg, gms, "rtrace", _parameters.rtrace, 1.1, id  );

	if (_parameters.burnin_method == "DRAM") {
		getFlagWithDefault(cfg, gms, "DRProb", _parameters.DRProb, 0.5, id  );
		getFlagWithDefault(cfg, gms, "DRScale", _parameters.DRScale, 0.1, id  );
	}

	return true;
}

bool getEvidenceParameters(Config& cfg, Setting & gms, evidenceParam& _parameters, const MPI::Intracomm _com ) {
	int id = _com.Get_rank();
	std::string localPath;

	getFlagWithDefault(cfg, gms, "evidenceMethod", _parameters.method, "NA", id  );

	if (_parameters.method == "CJ") {
		getFlagWithDefault(cfg, gms, "trim", _parameters.trim, 5, id  );
	}

	if (_parameters.method == "GH") {
		//Initial starting point
		if ( gms.exists("mu") ) {
			_parameters.mu = getVector(gms,  "mu", 10 );
			print("\tGauss-Hermite initial mean", _parameters.mu, id );
		} else {
			print("\tCouldn't find Gauss-Hermite initial mean", id );
			return false;
		}

		//Initial covariance
		if ( gms.exists("sigma") ) {
			_parameters.sigma = getMatrix(gms, "sigma", _parameters.mu.n_rows );
			print("\tGauss-Hermite initial sigma", _parameters.sigma, id );
		} else {
			_parameters.sigma = eye<mat>(_parameters.mu.n_rows, _parameters.mu.n_rows);
			print("\tGauss-Hermite initial sigma not found. Using Identity.", _parameters.sigma, id );
		}

		getFlagWithDefault(cfg, gms, "quadLevel", _parameters.quadLevel, 3, id  );
		getFlagWithDefault(cfg, gms, "quadTolerance", _parameters.quadTolerance,1.0e-10, id  );
	}

	return true;
}

bool getProposedModelParameters(Config& cfg, std::vector<proposedModels>& propModelVector, const MPI::Intracomm _com) {
	std::string function_name, function_handle, folder, datapath, covfile;
	colvec initialParam, initialState;
	mat proposalCov;
	mat initialCov;
	bool flag;

	//Checking if proposed_models can be found.
	if (!cfg.exists("proposed_models") ) {
		if (_com.Get_rank() == 0) { std::cout << "-----------> Error. Could not find any proposed models. Please check the config file.";}
		return false;
	}

	Setting &gms = cfg.lookup("proposed_models");
	int ngm = gms.getLength();
	int id = _com.Get_rank();
	proposedModels temp;
	std::string localPath;
	for (int i = 0; i < ngm; i++){
		temp = {}; //Reset the structure

		getFlagWithDefault(cfg, gms[i], "run", temp.run, true, id  );

		if (temp.run == true) { //Will actually run this model
			flag = gms[i].lookupValue("name", temp.function_name);
			if (!flag) {
				if (id == 0 ) {	std::cout << "Error in reading model " << i << ": name is required." << std::endl; }
				return false;
			}

			print( "Reading model " + temp.function_name , id);
			print( "\tSimulation parameters" , id);

			flag = getFlagRequired(cfg, gms[i], "dt", temp.dt, id  ) && getFlagRequired(cfg, gms[i], "fStepsBetweenMeasurements", temp.fStepsBetweenMeasurements, id  );
			if (!flag) {
				if (id == 0 ) {
					std::cout << "Error in reading model " << i << ": dt (" << temp.dt << ") and fStepsBetweenMeasurements ("<< temp.fStepsBetweenMeasurements << ") are required." << std::endl;
				}
				return false;
			}

			flag = getFlagRequired(cfg, gms[i], "folder", temp.folder, id  ) && getFlagRequired(cfg, gms[i], "data", temp.data, id );
			if (!flag) {
				if (id == 0 ) {	std::cout << "Error in reading model " << i << ": folder and data are required." << std::endl; }
				return false;
			}

			flag = getFlagRequired(cfg, gms[i], "handle", temp.function_handle, id  );
			if (!flag) {
				if (id == 0 ) {	std::cout << "Error in reading model " << i << ": handle is required." << std::endl; }
				return false;
			}

			flag = getFlagRequired(cfg, gms[i], "measurementCov", temp.cov, id  ) && getFlagRequired(cfg, gms[i], "state_estimator", temp.state_estimator, id );
			if (!flag) {
				if (id == 0 ) {	std::cout << "Error in reading model " << i << ": measurementCov and state_estimator are required." << std::endl; }
				return false;
			}

			print( "\n\tState Estimation parameters" , id);
			getFlagWithDefault(cfg, gms[i], "State_Estimation", temp.doStateEstimationRun, false, id  );
			//We are performing state estimation

			if (temp.doStateEstimationRun) {
				temp.parameters = getVector(gms[i], "parameters", 100 );
				getFlagWithDefault(cfg, gms[i], "seruns", temp.seruns, 1, id  );
				print("\tModel Parameters for state estimation ", temp.parameters, id );

				if (temp.parameters.n_rows == 0) {
					if (id == 0 ) {	std::cout << "Error in reading model " << i << ": missing parameters for state estimation." << std::endl; }
					return false;
				}
			}

			getFlagWithDefault(cfg, gms[i], "parallelGroups", temp.parallelGroups, 1, id  );
			getFlagWithDefault(cfg, gms[i], "nprocs", temp.nprocs, 1, id  );
			getFlagWithDefault(cfg, gms[i], "nparticles", temp.nparticles, 1000, id  );


			//Get initial state and initial parameter
			flag = readVector(temp.initialState , gms[i], "initialState" );
			if (!flag) {
				print("\tError: Initial state not found.", id );
				return false;
			} else {
				print("\tInitial state", temp.initialState, id );
			}

			if (gms[i].exists("initialStateVariance")) {
				temp.initialStateVariance = getMatrix(gms[i], "initialStateVariance", temp.initialState.n_rows );
				print("\tInitial state variance", temp.initialStateVariance, id );

				if (temp.initialStateVariance.n_rows == 0) {
					if (id == 0 ) {	std::cout << "Error in reading model " << i << ": can't read initialStateVariance ." << std::endl; }
					return false;
				}
			} else {
				print("\tSetting initial state variance covariance to Idenity" , id);
				temp.initialStateVariance = eye<mat>(temp.initialState.n_rows , temp.initialState.n_rows );
			}

			//Process noise covariance
			if (gms[i].exists("process_noise_covariance")) {
				temp.modelCov = getMatrix(gms[i], "process_noise_covariance", temp.initialState.n_rows );
				print("\tprocess_noise_covariance", temp.modelCov, id );
			} else {
				print("\tSetting process noise covariance to zero matrix with last term 1.0"  , id);
				temp.modelCov = zeros<mat>(temp.initialState.n_rows , temp.initialState.n_rows );
				temp.modelCov(temp.initialState.n_rows-1,temp.initialState.n_rows-1) = 1.0;
			}

			print( "\n\tMCMC parameters" , id);
			if ( gms[i].exists("MCMC_CONFIG") ) {
				getFlagWithDefault(cfg, gms[i], "Parameter_Estimation", temp.doParameterEstimation, true, id  );
				getFlagWithDefault(cfg, gms[i].lookup("MCMC_CONFIG"), "Method", temp.mcmcMethod, "MCMC" , id  );
				if (temp.mcmcMethod == "MCMC") {
					flag = getMCMCParameters(cfg, gms[i].lookup("MCMC_CONFIG") , temp.mcmcParameters , _com );
				} else if (temp.mcmcMethod == "TMCMC") {
					flag = getTMCMCParameters(cfg, gms[i].lookup("MCMC_CONFIG") , temp.tmcmcParameters , _com );
				} else {
					print("\tMCMC METHOD INVALID. Currently only supporting MCMC and TMCMC. Will not perform parameter estimation" , id);
					temp.doParameterEstimation = false;
				}
			} else {
				print("\tMCMC_CONFIG flag not found. Will not perform parameter estimation using MCMC." , id);
				temp.doParameterEstimation = false;
			}

			if (!flag) {
				if (id == 0 ) {	std::cout << "Error in reading MCMC parameters. Check proposedmodels.txt"  << std::endl; }
				return false;
			}

			print( "\n\tEvidence Estimation parameters" , id);
			getFlagWithDefault(cfg, gms[i], "Evidence_Estimation", temp.doEvidence, false, id  );
			if (temp.doEvidence) {
				flag = flag && getEvidenceParameters(cfg, gms[i], temp.evidenceParameters , _com );
			} else {
				print("\tEVIDENCE_CONFIG flag not found. Will not estimate the evidence." , id);
			}


			print( "\n\tOptimization parameters" , id);
			getFlagWithDefault(cfg, gms[i], "doOptimization", temp.doOptimization, false, id  );
			//Used in optimization
			if (temp.doOptimization) {
				getFlagWithDefault(cfg, gms[i], "nparameters", temp.nparameters, 0, id  );
				getFlagWithDefault(cfg, gms[i], "nelderMeadMaxIt", temp.nelderMeadMaxIt, 20, id  );
			}

			print("\n\tState Estimation Error", id);
			getFlagWithDefault(cfg, gms[i], "doStateEstimationError", temp.doStateEstimationError, false, id  );


			//Get the priors
			if ( gms[i].exists("prior") ) {
				print("\n\tUsing prior distribution for parameters.", id);
				flag = flag && readPriors(temp.priors, gms[i]);
			}

			if (!flag) {
				if (id == 0 ) {	std::cout << "Error in reading proposed models. Check proposedmodels.txt"  << std::endl; }
				return false;
			}
			propModelVector.push_back( temp );
		}
	}
	return true;
}

bool getGeneratingModelParameters(Config& cfg, std::vector<genModelParam>& vecModelParam ) {
	bool flag;
	std::string function_name, function_handle, folder;
	colvec parameters, initialState;
	Setting &gms = cfg.lookup("generating_models");
	int ngm = gms.getLength();

	genModelParam temp;
	for (int i = 0; i < ngm; i++){

		flag = 	gms[i].lookupValue("time", temp.time) &&
		gms[i].lookupValue("dt", temp.dt)	&&
		gms[i].lookupValue("stepsBetweenMeasurements", temp.stepsBetweenMeasurements) &&
		gms[i].lookupValue("name", temp.function_name) &&
		gms[i].lookupValue("NSR", temp.NSR) &&
		gms[i].lookupValue("handle", temp.function_handle) &&
		gms[i].lookupValue("folder", temp.folder);


		//Optional value

		//getFlagWithDefaultBool(cfg, gms[i], "shiftTime", temp.shiftTime, false, MPI::COMM_WORLD.Get_rank()  );
		//Get initial state and initial parameter
		flag = flag && readVector(initialState , gms[i], "initialState" );
		flag = flag && readVector(parameters , gms[i], "parameters" );

		temp.initialState = initialState;
		temp.parameters = parameters;
		vecModelParam.push_back( temp );

		if (!flag) {
			std::cout << "Error in reading generating models. Valid parameters (case sensitive) are " << std::endl
			<< "\t time" << std::endl
			<< "\t dt" << std::endl
			<< "\t name" << std::endl
			<< "\t NSR" << std::endl
			<< "\t handle" << std::endl
			<< "\t folder" << std::endl
			<< "\t initialState" << std::endl
			<< "\t stepsBetweenMeasurements" << std::endl
			<< "\t parameters" << std::endl;
			return false;
		}
	}
	return true;
}

bool openConfigFile( Config& cfg,  const char * filename ) {
	// Read the file. If there is an error, report it and exit.
	try
	{
		cfg.readFile(filename);
	}
	catch(const FileIOException &fioex)
	{
		std::cout << "I/O error while reading file." << std::endl;
		return false;
	}
	catch(const ParseException &pex)
	{
		std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
		return false;
	}

	std::ofstream log_file(	"proposedmodels.txt", std::ios_base::out | std::ios_base::trunc );
	log_file.close();
	return true;
}

bool readPriors(std::vector<pdf1d*>& _priors, Setting& stgs) {
	const Setting &stg = stgs.lookup( "prior" );

	int nrows = stg.getLength();
	int i = 0;

	while (i < nrows) {
		std::string temp = stg[i];
		if ( temp.compare("Gaussian") == 0 ) {
			_priors.push_back( new Gaussian1d(double(stg[i+1]),double(stg[i+2]) ));
			i = i+3;
		} else if ( temp.compare("Uniform") == 0 ) {
			//std::cout << "Uniform(" << double(stg[i+1]) << "," << double(stg[i+2]) << ")" << std::endl;
			_priors.push_back( new Uniform1d(double(stg[i+1]),double(stg[i+2])) );
			i = i+3;
		} else if ( temp.compare("Log-Normal") == 0 ) {
			_priors.push_back( new LogNormal1d(double(stg[i+1]),double(stg[i+2])) );
			i = i+3;
		} else if ( temp.compare("Reciprocal") == 0 ) {
			_priors.push_back( new reciprocal() );
			i ++; //no need to skip the other values
		} else {
			std::cout << "Error reading priors. Currently only unidimensional Gaussian, Uniform, Log-Normal and reciprocal are supported." << std::endl;
			return false;
		}
	}
	return true;
}

bool readVector( colvec& vector, Config& cfg, std::string filename ) {
	const Setting &stg = cfg.lookup( filename );
	int count = stg.getLength();
	vector.set_size( count );
	for (int i = 0; i < count; i++) {
		vector[i] = stg[i];
	}
	return true;
}

bool readVector( colvec& vector, Setting& stgs, std::string filename ) {
	const Setting &stg = stgs.lookup( filename );
	int count = stg.getLength();
	vector.set_size( count );
	for (int i = 0; i < count; i++) {
		vector[i] = stg[i];
	}
	return true;
}

mat getMatrix(Setting& stgs,  std::string settingName, int size ) {
	Setting &stg = stgs.lookup( settingName );
	mat toReturn;
	std::string temp,formated;
	switch(stg.getType() ) {
	case (Setting::TypeList) :  //Manually copy the matrix
		{
			int nrows = stg.getLength();
			int ncols = stg[0].getLength();

			toReturn.set_size( nrows, ncols );
			for (int i = 0; i < nrows; i++) {
				for (int j = 0; j < ncols; j++) {
					toReturn(i,j) = (stg[i])[j];
				}
			}
			break;       // and exits the switch
		}
	case (Setting::TypeString) :
		{
			stgs.lookupValue(settingName, formated);
			temp = formated;
			for (auto & c: formated) c = toupper(c);
			if (formated == "EYE" || formated == "IDENTITY") {
				toReturn = eye<mat>( size , size );
			} else if (formated == "NOISE" ) {
				toReturn = zeros<mat>( size , size );
				toReturn(size-1,size-1) = 1.0;
			} else {
				std::size_t found = formated.find(".DAT");
				if (found!=std::string::npos) {
					toReturn.load(temp ,raw_ascii);
				}}
			break;
		}
		default :
		toReturn = zeros<mat>( size , size );
	}
	return toReturn;
}

mat getMatrix(Config& cfg,  std::string settingName, int size ) {
	Setting &stg = cfg.lookup( settingName );
	mat toReturn;
	std::string temp,formated;
	switch(stg.getType() ) {
	case (Setting::TypeList) :  //Manually copy the matrix
		{
			int nrows = stg.getLength();
			int ncols = stg[0].getLength();

			toReturn.set_size( nrows, ncols );
			for (int i = 0; i < nrows; i++) {
				for (int j = 0; j < ncols; j++) {
					toReturn(i,j) = (stg[i])[j];
				}
			}
			break;       // and exits the switch
		}
	case (Setting::TypeString) :
		{
			cfg.lookupValue(settingName, formated);
			temp = formated;
			for (auto & c: formated) c = toupper(c);
			std::cout << "Type string" << std::endl;
			if (formated == "EYE" || formated == "IDENTITY") {
				toReturn = eye<mat>( size , size );
			} else if (formated == "NOISE" ) {
				toReturn = zeros<mat>( size , size );
				toReturn(size-1,size-1) = 1.0;
			} else {
				std::size_t found = formated.find(".DAT");
				if (found!=std::string::npos) {
					toReturn.load(temp ,raw_ascii);
				}}
			break;
		}
		default :
		std::cout << " Default case " << std::endl;
	}
	return toReturn;
}

colvec getVector(Config& cfg,  std::string settingName, int size ) {
	Setting &stg = cfg.lookup( settingName );
	colvec toReturn;
	std::string temp,formated;
	switch(stg.getType() ) {
	case (Setting::TypeList) :  //Manually copy the matrix
		{
			int count = stg.getLength();
			toReturn.set_size( count );
			for (int i = 0; i < count; i++) {
				toReturn[i] = stg[i];
			}
			break;     // and exits the switch
		}
	case (Setting::TypeString) :
		{
			cfg.lookupValue(settingName, formated);
			temp = formated;
			for (auto & c: formated) c = toupper(c);
			if (formated == "ZERO" ) {
				toReturn = zeros<colvec>( size );
			} else {
				std::size_t found = formated.find(".DAT");
				if (found!=std::string::npos) {
					toReturn.load(temp ,raw_ascii);
				}}
			break;
		}
		default :
		std::cout << " Default case " << std::endl;
	}
	return toReturn;
}
//TODO::error message when can't load matrix. Fix issue with size....make it optional?
colvec getVector(Setting& stgs,  std::string settingName, int size ) {
	Setting &stg = stgs.lookup( settingName );
	colvec toReturn;
	std::string temp,formated;
	switch(stg.getType() ) {
	case (Setting::TypeList) :  //Manually copy the matrix
		{
			int count = stg.getLength();
			toReturn.set_size( count );
			for (int i = 0; i < count; i++) {
				toReturn[i] = stg[i];
			}
			break;     // and exits the switch
		}
	case (Setting::TypeString) :
		{
			stgs.lookupValue(settingName, formated);
			temp = formated;
			for (auto & c: formated) c = toupper(c);
			if (formated == "ZERO" ) {
				toReturn = zeros<colvec>( size );
			} else {
				std::size_t found = formated.find(".DAT");
				if (found!=std::string::npos) {
					toReturn.load(temp ,raw_ascii);

				}}
			break;
		}
		default :
		std::cout << " Default case " << std::endl;
	}
	return toReturn;
}

bool readMatrix( mat& matrix, Setting& stgs, std::string filename ) {
	const Setting &stg = stgs.lookup( filename );
	//matrix.print("Before entering:");
	int nrows = stg.getLength();
	int ncols = stg[0].getLength();
	//std::cout << "Nrows is " << nrows << " and ncols is " << ncols << std::endl;
	matrix.set_size( nrows, ncols );
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			//std::cout << (stg[i])[j] << std::endl;
			matrix(i,j) = (stg[i])[j];
		}
	}
	//matrix.print("Before leaving:");
	return true;
}

bool readMatrix( mat& matrix, Config& cfg, std::string filename ) {
	const Setting &stg = cfg.lookup( filename );

	int nrows = stg.getLength();
	int ncols = stg[0].getLength();

	matrix.set_size( nrows, ncols );
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			matrix(i,j) = (stg[i])[j];
		}
	}
	return true;
}

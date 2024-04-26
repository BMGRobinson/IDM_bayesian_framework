#include "filters.hpp"
using namespace arma;

//FirstState will not work!!!

/*****************************************************************
Helper methods
*****************************************************************/
//Method that fills Y and dpVY since these do not depends on the right hand side.
//All process will share them
/**
* Pre-process parallel function for Sherman-Morrison solver
* @param A    A matrix that needs to be diagonal
* @param U    U matrix
* @param V    V matrix
* @param Y    Y matrix
* @param dpVY vector containing the denominator. Used in the solver
* @param _com MPI communicator
*/
void shermanMorrisonMPIPreProc(const mat& A, const mat& U, const mat& V, mat& Y, vec& dpVY, const MPI::Intracomm _com) {

	int m = A.n_rows; //number of measurements
	int n = U.n_cols; //number of columns (number of particles for Enkf)

	int i,j,k;	/**< Indices */
	double num, den;
	int r = 0;
	int index[n];
	int nproc = _com.Get_size();
	int pid = _com.Get_rank();

	//Solve for the Ys A y = U
	for (j = 0; j < n; j++) {
		for (i = 0; i < m; i++) {
			Y(i,j) = U(i,j) / A(i,i);
		}
	}

	//Get the index. For each index, tell which process is responsible for it
	index[0] = 0; //first index belongs to root
	for (i = 1; i < n; i ++) {
		//If the previous index is equal to the number of process
		if (index[i-1] == (nproc-1)) {
			r = 0;
		} else {
			r ++;
		}
		index[i] = r;
	}

	//Solve the Y's in parallel. I will share the current Y(:,i) with everyone
	//Bcast(void* buffer, int count, const MPI::Datatype& datatype, int root) const = 0
	for (i = 0; i < n; i ++) {
		_com.Bcast( Y.colptr(i) , m, MPI::DOUBLE, index[i] );
		den = 1.0 + dot(Y.unsafe_col(i), V.unsafe_col(i));
		dpVY(i) = den;
		for (k = i + 1; k < n; k++) {
			if (index[k] == pid) {
				num = dot(Y.unsafe_col(k), V.unsafe_col(i));
				Y.unsafe_col(k) = Y.unsafe_col(k) - num/den * Y.unsafe_col(i);
			}
		}
	}
}
//Returns the solution of (A + UV^T) z = d. Stores it in z
void simplifiedShermanMorrison(const mat& A, const mat& V, const mat& Y, const vec& dpVY, colvec& z, const colvec& d, const MPI::Intracomm _com) {
	int m = A.n_rows; //number of measurements
	int n = V.n_cols; //number of columns (number of particles for Enkf)
	int i;

	//First solve the diagonal matrix. A z = d
	for ( i = 0; i < m; i++) {
		z(i) = d(i) / A(i,i);
	}
	//Work on the subset of particles
	for (i = 0; i < n; i ++) {
		z = z - dot(V.unsafe_col(i), z)/dpVY(i) * Y.unsafe_col(i);
	}
}
/*****************************************************************
STATE ESTIMATOR CLASS
*****************************************************************/

state_estimator::state_estimator() {
	//default constructor, do nothing
	save_to_file = false;
	id = 0;
	output = false;
}

state_estimator::state_estimator(statespace _ss) {
	save_to_file = false;
	ss = _ss;
	id = 0;
}

state_estimator::~state_estimator() {
	id = 0; //default value
}

void state_estimator::saveToFile(const bool save, std::string filename, int _timestep = 1) {
	save_to_file = save;
	state_estimation_filename = filename;
	save_to_file_step = _timestep;
}

	void state_estimator::disableSaveToFile() {
		save_to_file = false;
	}

/*****************************************************************
DETERMINISTIC CLASS
*****************************************************************/

Deterministic::Deterministic() {}

Deterministic::Deterministic( const Deterministic &other ) {
	ss = other.ss;
	firstState = other.firstState;
	state = other.state;
}

Deterministic::Deterministic( colvec& initial_state, statespace _ss) : state_estimator::state_estimator(_ss) {
	firstState = initial_state;
	state = initial_state;
}

double Deterministic::logLikelihood(const mat& timedData, const colvec&  parameters ) {
	std::cout << "Return ERROR. Need to fix see logLikelihood from filters " << std::endl;
	assert(false);
	reset(); //reset filter to original state
	long double time = 0.0;
	double dt = ss.getDt();
	ss.resetTime();
	double logLik = 0.0;
	double currentLogLik;
	double lastMeasurementTime = timedData(timedData.n_rows-1, timedData.n_cols-1);
	colvec currentObs = colvec(timedData.n_rows-1);
	int j = 0;

	//Only used when saving to file
	int s = 0;
	int Cs = std::ceil(lastMeasurementTime/dt)/save_to_file_step + 1; //One time for each timestep.
	int Is = timedData.n_rows - 1 + 1 + getCurrentState().n_rows + 1; //size of obs vector + time + size of state vector x 2 (mean + covariance) + 1 for lik

	mat stateEstimate;
	if (id == 0 && save_to_file) {
		stateEstimate = mat(Is,Cs);
	}
	colvec tempState;

	colvec obsTemp = colvec(timedData.n_rows-1);
	obsTemp.fill(-999);
	//while (time <= lastMeasurementTime)
	for (int iteration= 0;  iteration <= std::ceil(lastMeasurementTime/dt) ; iteration ++) {
		if (id == 0 && save_to_file && (iteration % save_to_file_step) == 0) {
			//Record the state and covariance
			tempState = getCurrentState();
			stateEstimate(span(0, tempState.n_rows-1),s ) = tempState;
			stateEstimate(span(tempState.n_rows,tempState.n_rows+timedData.n_rows-2) , s) = obsTemp;
			stateEstimate(Is-2, s) = -1999;
			stateEstimate(Is-1, s) = time; //Time
			s++;
		}
		//measurement is available
		if ( std::abs( time - timedData(timedData.n_rows-1,j) ) < 0.00000001 ) {
			currentObs = timedData(span(0,timedData.n_rows-2), j);
			/*
			*  since we are taking the log of the density ln(0) is not defined, thus will return NaN. In that case we simply return that value.
			*  no need to complete the state estimation
			*/
			currentLogLik = logLikelihoodOfMeasurement(currentObs, parameters);
			if (std::isnan(currentLogLik)) {
				return currentLogLik;
			}
			logLik += currentLogLik;
			j++;
		}
		//forecast to the next time step
		forecast(parameters);
		ss.timeIncrement();
		time += dt;
	}

	if (id == 0 && save_to_file) {
		mat temp = trans(stateEstimate);
		temp.save(state_estimation_filename, raw_ascii);
	}

	return logLik;
}

double Deterministic::logLikelihoodOfMeasurement(const colvec& measurement) {
	return logLikelihoodOfMeasurement(measurement, colvec() );
}

double Deterministic::logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters) {
	return ss.evaluateloglikelihood(measurement, state, parameters );
}

void Deterministic::forecast(const colvec&  parameters) {
	state = ss.evaluatef( state, parameters );
}

void Deterministic::reset() {
	state = firstState;
}

void Deterministic::print() {
	//do nothing
}

//void initializeFilter( problem* _problem);
Deterministic::~Deterministic() {
	//do nothing
}

colvec& Deterministic::getState() {
	return state;
}

colvec Deterministic::getCurrentState() {
	return state;
}

double Deterministic::state_estimation( const mat& data, const colvec&  parameters , bool save ) {
	std::cout << "State Estimation is NOT IMPLEMENTED" << std::endl;
	assert(false);
}

colvec Deterministic::getSquareError( const mat & data, const mat & reference, const colvec& parameters) {
	std::cout << "getSquareError is NOT IMPLEMENTED" << std::endl;
	assert(false);

}

/*****************************************************************
FILTER CLASS
*****************************************************************/

filter::filter() : state_estimator() {
}

filter::filter(statespace _ss) : state_estimator( _ss ) {
}

filter::~filter() {
}

//make that functoin parallel (so that only root prints)

/**
* Returns the loglikelihood.
* @param  timedData  Measurements and the corresponding time at which they were taken.
* @param  parameters Parameter vector
* @return            loglikelihood
*/
double filter::logLikelihood(const mat& data, const colvec&  parameters ) {
	reset(); //reset filter to original state
	int Nm = ss.getForecastStepsBetweenMeasurements();
	ss.resetTime();
	double logLik = 0.0;
	double currentLogLik;
	colvec currentObs;
	int j = 0;
	int s = 0;
	int numberOfMeasurements = data.n_cols;

 	while(j < numberOfMeasurements) {
		if (s == Nm || j == 0) {
			currentObs = data.col(j);
			/*
			*  since we are taking the log of the density ln(0) is not defined, thus will return NaN. In that case we simply return that value.
			*  no need to complete the state estimation
			*/
			currentLogLik = logLikelihoodOfMeasurement(currentObs, parameters);
			//std::cout <<"Current log lik at data point " << j << " is " << currentLogLik << " at time " << ss.getTime() << std::endl;
			if (std::isnan(currentLogLik)) {
				return currentLogLik;
			}
			logLik += currentLogLik;
			update(currentObs, parameters);
			j++; 	//increase measurement index
			s = 0; 	//Reset s
		}
		forecast(parameters); 		//forecast to the next time step
		s++;
		ss.timeIncrement();
	}

	return logLik;
}

double filter::state_estimation( const mat& data, const colvec&  parameters , bool save ) {
	reset(); //reset filter to original state
	int Nm = ss.getForecastStepsBetweenMeasurements();
	ss.resetTime();
	colvec currentObs;
	int j = 0;
	int s = 0;
	int n;
	int m;
	int l;
	int numberOfMeasurements = data.n_cols;
	l = parameters.n_rows;
	colvec temp;
	double logLik = 0.0;
	double currentLogLik;
	std::ofstream se;

	//Saving state estimation to a file
	//if (id == 0) {
	se = std::ofstream( state_estimation_filename + std::to_string(id) + ".xml", std::ios_base::out | std::ios_base::trunc );
	se << "<\?xml version=\"1.0\"\?>" << std::endl << "<root>" <<std::endl;
	se << "<parameters>";
	for (int i = 0; i < l; i ++){
		se << " " << parameters[i];
	}
	se << "</parameters>" << std::endl;
	se << "<dataPoints>" << numberOfMeasurements << "</dataPoints>" << std::endl;
	se << "<states>" << std::endl;

	int index = 0;
 	while(j < numberOfMeasurements) {
		if (save) {
			se << "<state>" << std::endl;
			se << "<time>" << ss.getTime() << "</time>" << std::endl;
			se << "<index>" << index << "</index>" << std::endl;
			getStateXML( se );
			se << "</state>" << std::endl;
			index ++;
		}



		//Is there a measurement at this time instant?
		if (s == Nm || j == 0) {
			currentObs = data.col(j);
			/*
			*  since we are taking the log of the density ln(0) is not defined, thus will return NaN. In that case we simply return that value.
			*  no need to complete the state estimation
			*/
			currentLogLik = logLikelihoodOfMeasurement(currentObs, parameters);
			//std::cout <<"Current log lik at data point " << j << " is " << currentLogLik << " at time " << ss.getTime() << std::endl;
			if (std::isnan(currentLogLik)) {
				return currentLogLik;
			}
			logLik += currentLogLik;
			update(currentObs, parameters);
			j++; 	//increase measurement index
			s = 0; 	//Reset s

			if (save) {
				se << "<state>" << std::endl;
				se << "<time>" << ss.getTime() << "</time>" << std::endl;
				se << "<index>" << index << "</index>" << std::endl;
				//Record the measurement if there is any
				se << "<measurement>";
				m = currentObs.n_rows;
				for (int i = 0; i < m; i++) {
					se << " " << currentObs[i];
				}
				se << "</measurement>" << std::endl;
				se << "<loglik>" << currentLogLik << "</loglik>" << std::endl;
				//All filters have their own version of this
				getStateXML( se );
				se << "</state>" << std::endl;
				index ++;
			}
		}

		//forecast to the next time step
		forecast(parameters);
		s++;
		ss.timeIncrement();
	}
	se << "</states>" << std::endl << "</root>";
	se.close();
	cout << "Brandon says the log-likelihood is" << logLik;
	return logLik;
}
//Return error vector {e_1^2, e_2^2,...,e_n^2}
colvec filter::getSquareError( const mat & data, const mat & reference, const colvec& parameters) {
	reset(); //reset filter to original state
	int Nm = ss.getForecastStepsBetweenMeasurements();
	ss.resetTime();
	colvec currentObs;
	int j = 0;
	int s = 0;
	int i;
	int index = 0;
	//int n = ss.getStateVectorSize();
	//int n = reference.n_rows;
	int n = 2;
	int numberOfMeasurements = data.n_cols;
	colvec temp;
	colvec error = zeros<colvec>(n);
	for (i = 0; i < n; i ++) {error[i] = 0.0;	} //Error is 0 initially

 	while(j < numberOfMeasurements) {

		//Is there a measurement at this time instant?
		if (s == Nm || j == 0) {

			currentObs = data.col(j);
			update(currentObs, parameters);
			j++; 	//increase measurement index
			s = 0; 	//Reset s

			//Only compute the error after an observation
			temp = getCurrentState();
			for (i = 0; i < n; i ++) {
				error[i] += (temp[2*i] - reference.at(i,index))*(temp[2*i]-reference.at(i,index));
			}
		}

		//Get the error at this time instant. Get current state regurns (state 1 var 1...state n var n)
		//TODO: do it better


		//forecast to the next time step
		forecast(parameters);
		s++;
		index ++;
		ss.timeIncrement();
	}

	return error / double(numberOfMeasurements);
}


/*****************************************************************
EKF CLASS
*****************************************************************/

Ekf::Ekf() : filter::filter() {
	//do nothing
}

Ekf::Ekf(const Ekf &other) : filter(other) {
	ss = other.ss;
	firstState = Gaussian(other.firstState);
	state = Gaussian(other.state);
	K = other.K;
}

Ekf::Ekf( Gaussian* _initial_state, statespace _ss, const mat& _Q, const mat& _R) : filter::filter(_ss) {
	firstState = Gaussian(*_initial_state);
	state = Gaussian(*_initial_state);
	Q = _Q;
	R = _R;
	//assert( checkMatrixSize()); //Make sure the matrix size are ok doesn't work when there are parameters
}

void Ekf::reset() {
	state = firstState;
}

colvec Ekf::getCurrentState() {
	colvec tempMean = state.getMean();
	mat tempCov = state.getCovariance();
	int n = tempMean.n_rows;
	colvec temp = colvec(2*n);
	for (int i = 0; i < n; i++) {
		temp(2*i) = tempMean(i);
		temp(2*i+1) = tempCov(i,i);
	}
	return temp;
}

void Ekf::forecast(const colvec&  parameters) {
	mat _dfdx = ss.evaluatedfdx( state.getMean(), parameters );
	mat _dfde = ss.evaluatedfde( state.getMean(), parameters );
	mat temp1 = _dfdx * state.getCovariance() * trans(_dfdx);
	mat temp2 = _dfde * trans(_dfde);
	//mat temp2 = _dfde * Q * trans(_dfde);

	//std::cout << "Temp 1 = " << temp1 << std::endl;
	//std::cout << "Temp 2 = " << temp2 << std::endl;
	//state.setCovariance( _dfdx * state.getCovariance() * trans(_dfdx) + _dfde * Q * trans(_dfde) );
	state.setCovariance( _dfdx * state.getCovariance() * trans(_dfdx) + _dfde * trans(_dfde) );
	state.setMean( ss.evaluatef( state.getMean() , parameters ) );
}
//It might be useful to set P = 0.5*P+0.5*trans(P) -> it forces symmetry
/**
* Update procedure for the Extended Kalman filter. See equations []
* @param _d         Measurement vector
* @param parameters Parameter vector
*/
void Ekf::update(const colvec& _d, const colvec&  parameters) {
	mat pkf = state.getCovariance();
	mat xkf = state.getMean();

	colvec residual = _d - ss.evaluateh( xkf, parameters);

	mat _dhdx = ss.evaluatedhdx( xkf, parameters );
	mat _dhde = ss.evaluatedhde( xkf, parameters );
	S = _dhdx * pkf * trans(_dhdx) + _dhde * R * trans(_dhde);
	K = ( pkf * trans(_dhdx)) * inv_sympd(S);
	state.setMean(xkf + K * residual );
	state.setCovariance( pkf - K*_dhdx*pkf);
}

mat Ekf::getKalmanGain() {
	return K;
}

/**
* Returns the state as a Gaussian distribution
* @return pointer of a gaussian object
*/
Gaussian * Ekf::getState() {
	return &state;
}

/**
* Check the matrix size to make sure they are consistent
* @return True if the matrix size are ok
*/
bool Ekf::checkMatrixSize() {
	//Funciton that checks the size of all the matrix to make sure they are compatible
	bool sizeOk = true;
	colvec x = state.getMean();
	colvec obs = ss.evaluateh(x);
	mat cov = state.getCovariance();
	mat _dfdx = ss.evaluatedfdx(x);
	mat _dfde = ss.evaluatedfde(x);
	mat _dhdx = ss.evaluatedhdx(x);
	mat _dhde = ss.evaluatedhde(x);
	unsigned int n,m;

	n = x.n_rows;
	m = obs.n_rows;
	//Check 1, state is N by 1
	if (x.n_cols > 1) {
		std::cout << "ERROR: state should be " << n << " x 1. Current size is (" << x.n_rows << " x " << x.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 2, covariance is symetric
	if (cov.n_cols != cov.n_rows) {
		std::cout << "ERROR: Covariance should be " << n << " x " <<  n << ". Current size is (" << cov.n_rows << " x " << cov.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 3 covariance size matches with the state
	if (cov.n_cols != n) {
		std::cout << "ERROR: Covariance should be " << n << " x " <<  n << ". Current size is (" << cov.n_rows << " x " << cov.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 4, model noise is n by n
	if (Q.n_cols != n || Q.n_rows != n) {
		std::cout << "ERROR: Model error covariance should be " << n << " x " <<  n << ". Current size is (" << Q.n_rows << " x " << Q.n_cols << ")" << std::endl;
		sizeOk = false;
	}

	//Check 5, measurement noise is
	if (R.n_cols != m || R.n_rows != m) {
		std::cout << "ERROR: measurement error covariance should be " << m << " x " <<  m << ". Current size is (" << R.n_rows << " x " << R.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 6 dfdx is n by n
	if (_dfdx.n_cols != n || _dfdx.n_rows != n) {
		std::cout << "ERROR: Jacobian df/dx should be " << n << " x " <<  n << ". Current size is (" << _dfdx.n_rows << " x " << _dfdx.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 7 dfde is n by n
	if (_dfde.n_cols != n || _dfde.n_rows != n) {
		std::cout << "ERROR: Jacobian df/de should be " << n << " x " <<  n << ". Current size is (" << _dfde.n_rows << " x " << _dfde.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 8 dhdx m by n
	if (_dhdx.n_cols != n || _dhdx.n_rows != m) {
		std::cout << "ERROR: Jacobian dh/dx should be " << m << " x " <<  n << ". Current size is (" << _dhdx.n_rows << " x " << _dhdx.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	//Check 9 dhde m by n
	if (_dhde.n_cols != m || _dhde.n_rows != m) {
		std::cout << "ERROR: Jacobian dh/de should be " << m << " x " <<  m << ". Current size is (" << _dhde.n_rows << " x " << _dhde.n_cols << ")" << std::endl;
		sizeOk = false;
	}
	return sizeOk;
}
/**
* Set the measurement covariance
* @param _R Covariance of the measurement matrix
*/
void Ekf::setMeasurementCovariance( const mat& _R) {
	R = _R;
}


//See eqn 13 of Nonlinear Dynamics paper
double Ekf::logLikelihoodOfMeasurement(const colvec& measurement) {
	return logLikelihoodOfMeasurement(measurement, colvec() );
}

/**
* Returns the loglikelihood when state estimation is performed using EKF
* @param  measurement Observation vector
* @param  parameters  Parameter vector
* @return             Returns the log likelihood
*/
double Ekf::logLikelihoodOfMeasurement(const colvec& measurement, const colvec& parameters) {
	//In the case of EKF this corresponds to evaluating the density of a normal distribution
	//having a mean of h(xk) and covariance Gamma = dhdx P dhdx^T + dhde R dhde^T
	colvec x = state.getMean();
	mat _dhdx = ss.evaluatedhdx(x,parameters);
	mat _dhde = ss.evaluatedhde(x,parameters);
	mat gamma = _dhdx * state.getCovariance() * trans(_dhdx) + _dhde * R * trans(_dhde);
	//gamma.print("Gamma");
	//Invalid covariance matrix
	if ( gamma.has_nan() ) {
			return std::numeric_limits<double>::quiet_NaN();
	}

	if ( det(gamma) < 0 ) {
			return std::numeric_limits<double>::quiet_NaN();
	}
	Gaussian logLik = Gaussian( ss.evaluateh(x,parameters) , gamma);
	return logLik.getLogDensity( measurement );
}
/**
* Display the mean and covariance of EKF
*/
void Ekf::print() {
	std::cout << "EKF:" << std::endl;
	//std::cout << "State of the system at time = " << this->time << "s" << std::endl;
	state.getMean().print("Mean:");
	state.getCovariance().print("Covariance:");
	std::cout << std::endl;
}


void Ekf::getStateXML(std::ofstream & se) {
	se << "<mean>" << std::endl;
	int n = state.getMean().n_rows;
	for (int i = 0; i < n; i ++) {
		se << " " << state.getMean()[i];
	}
	se << " </mean>" << std::endl;
	se << "<covariance>" << state.getCovariance() << "</covariance>" << std::endl;
}

Ekf::~Ekf() {}


/*****************************************************************
Ensemble Kalman filter CLASS
*****************************************************************/
Enkf::Enkf() : filter::filter() {
	//do nothing
}

Enkf::Enkf( colvec& _mean,mat& _cov, int _N, statespace _ss): filter::filter(_ss) {
	//Creates the samples based on Gaussian distribution
	N = _N;
	initialMean = _mean;
	initialCov = _cov;
	state = Samples(_mean, _cov, N);
	first = 0;
	last = N-1;
	int m = _ss.getMeasurementVectorSize();
	pmeasurements = Samples( zeros<colvec>(m) , eye<mat>(m,m), N); //Pertubed measurements
}
/**
 * Forecast each Monte-Carlo sample
 * @param parameters Parameter vector
 */
void Enkf::forecast(const colvec&  parameters) {
	for (int i = first; i <= last; i++) {
		state.setSampleAt(i, ss.evaluatef( state.getSampleAt(i) , parameters ) );
	}
}

/**
 * Enkf Update. Is used for linear measurement. Might need to change so we perturb the sample instead.
 * @param _d         Measurement vector at this timestep
 * @param parameters Measurement vector
 */
void Enkf::update(const colvec& _d, const colvec&  parameters ) {

	//Eqs 40 - 45 Nonlinear dynamics paper
	for (int i = first; i <= last; i++){
		//artificial measurements
		pmeasurements.setSampleAt(i, ss.evaluateh(state.getSampleAt(i), parameters)); //Eq. 40
	}
	Pxd = state.getCrossCovariance( pmeasurements ); //Eq. 43
	Pdd = pmeasurements.getCovariance();						 //Eq. 44
	kalman = Pxd * inv(Pdd);	//Eq. 45

	for (int i = first; i <= last; i++){
		state.setSampleAt(i, state.getSampleAt(i) + kalman*(_d - pmeasurements.getSampleAt(i)) );
	}
}

double Enkf::logLikelihoodOfMeasurement(const colvec& measurement) {
	return logLikelihoodOfMeasurement(measurement, colvec() );
}

/**
* Returns the loglikelihood when state estimation is performed using EnKF.
* The loglikelihood is the sum of the likelihood of each particle.
* @param  measurement Observation vector
* @param  parameters  Parameter vector
* @return             Returns the log likelihood
*/
double Enkf::logLikelihoodOfMeasurement(const colvec& _d, const colvec&  parameters) {
	long double lik = 0.0;
	long double particlelogLikelihood;
	//Gaussian logLik = Gaussian( _d , R);
	for (int i = first; i <= last; i++) {
		//particlelogLikelihood = logLik.getLogDensity( ss.evaluateh( state.getSampleAt(i) , parameters ) );
		particlelogLikelihood = ss.evaluateloglikelihood(_d, state.getSampleAt(i) , parameters );
		if (std::isnan(particlelogLikelihood)) {
			//lik  += 0;  exp(-infinity) = 0
		} else {
			lik += std::exp(particlelogLikelihood);
		}
	}
	lik = lik / double(N);
	return std::log(lik);
}

/**
 * Reset the filter. Restore the original samples
 */
void Enkf::reset() {
	//state = firstState;
	state = Samples(initialMean, initialCov, N);
}

/**
 * Return the mean value of the samples
 * @return mean vector
 */
//colvec Enkf::getMean() {
//	return state.getMean();
//}

/**
 * Save the current samples
 * @param filename string of the filneame where the samples will be saved
 */
void Enkf::saveSamples(std::string filename) {
	state.getSamples().save(filename, raw_ascii);
}

/**
 * Returns the covariance of the samples
 * @return Matrix containing the covariance
 */
//mat Enkf::getCovariance() {
//	return state.getCovariance();
//}

/**
 * Print state mean and covariance at current timestep
 */
void Enkf::print() {
	std::cout << "EnKF:" << std::endl;
	state.getMean().print("Mean:");
	state.getCovariance().print("Covariance:");
	std::cout << std::endl;
}

Samples& Enkf::getState() {
	return state;
}

/**
 * Print summary of the state
 * @return Returns a flattened vector containing the mean and the covariance
 * but not the cross covariance
 */
colvec Enkf::getCurrentState() {
	colvec tempMean = state.getMean();
	mat tempCov = state.getCovariance();
	int n = tempMean.n_rows;
	colvec temp = colvec(2*n);
	for (int i = 0; i < n; i++) {
		temp(2*i) = tempMean(i);
		temp(2*i+1) = tempCov(i,i);
	}
	return temp;
}

void Enkf::getStateXML(std::ofstream & se) {
	//Get each particle and its id
	int j,k;
	int l = state.getSampleAt(first).n_rows;
	colvec temp;
	mat tempCov;
	//Save every 50 particles
	temp = state.getMean();
	tempCov = state.getCovariance();
	se << "<mean>";
	for (j = 0; j < l; j ++) {
		se << temp[j] << " ";
	}
	se << "</mean>" << std::endl << "<covariance>";
	for (j = 0; j < l; j ++) {
			for (k = 0; k < l; k ++) {
				se << tempCov(j,k) << " ";
			}
	}
	se << "</covariance>" << std::endl;
	for (int i = first; i <= last; i = i + 50){
		se << "<p>" << std::endl;
		se << "<id>" << i << "</id>" << std::endl << "<val>" ;
		temp = state.getSampleAt(i);
		for (j = 0; j < l; j ++) {
			se << temp[j] << " ";
		}
		se << "</val>" << std::endl;
		se << "</p>" << std::endl;
	}
}

Enkf::~Enkf() {}


/*****************************************************************
Ensemble Kalman filter MPI
*****************************************************************/

EnkfMPI::EnkfMPI() : Enkf::Enkf() {};

EnkfMPI::EnkfMPI( colvec& _mean, mat& _cov, int _N, statespace  _ss, const MPI::Intracomm _com) {
	com = _com;	//Communicator
	nproc = com.Get_size();
	id = com.Get_rank();
	N = _N; //Total number of particles
	int sub = N / nproc; //Each process will have sub elements
	first = id*sub; 			//Index of the first element belonging to the current process
	last = first + sub - 1;		//Index of the last element belonging to the current process (inclusive)
	int m = _ss.getMeasurementVectorSize();
	state = SamplesMPI(_mean, _cov, N, com);
  pmeasurements = SamplesMPI( zeros<colvec>(m) , eye<mat>(m,m), N, com); //Pertubed measurements
	ss = _ss;
	//Initialize memory for Kalman gain
	kalman = zeros<mat>(_ss.getStateVectorSize(), _ss.getMeasurementVectorSize());
	initialMean = _mean;
	initialCov = _cov;

}

SamplesMPI& EnkfMPI::getState() {
	return state;
}

void EnkfMPI::reset() {
	state = SamplesMPI(initialMean, initialCov, N, com);
}
/**
 * Parallel forecast for Enkf. Each process has it own first and last
 * @param parameters Parameter vector
 */
void EnkfMPI::forecast(const colvec&  parameters) {
	for (int i = first; i <= last; i++) {
		state.setSampleAt(i, ss.evaluatef( state.getSampleAt(i) , parameters ) );
	}
}

//colvec EnkfMPI::getMean() {
//	return state.getMean();
//}
/**
 * Parellel implementation of getCovariance
 * @return [description]
 */
//mat EnkfMPI::getCovariance() {
//	return state.getCovariance();
//}

colvec EnkfMPI::getCurrentState() {
	colvec tempMean = state.getMean();
	mat tempCov = state.getCovariance();
	int n = tempMean.n_rows;
	colvec temp = colvec(2*n);
	for (int i = 0; i < n; i++) {
		temp(2*i) = tempMean(i);
		temp(2*i+1) = tempCov(i,i);
	}
	return temp;
}

double EnkfMPI::logLikelihoodOfMeasurement(const colvec& measurement) {
	return logLikelihoodOfMeasurement(measurement, colvec() );
}

double EnkfMPI::logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters) {
	//In the case of sample based filter, it is a sum of the likelihood
	long double lik = 0.0;
	long double particlelogLikelihood;
	//Gaussian logLik = Gaussian( measurement , R);
	for (int i = first; i <= last; i++) {
		//particlelogLikelihood = logLik.getLogDensity( ss.evaluateh( state.getSampleAt(i) , parameters ) );
		particlelogLikelihood = ss.evaluateloglikelihood(measurement, state.getSampleAt(i) , parameters );
		lik += std::exp(particlelogLikelihood);
	}
	//std::cout << "Before allreduce" << std::endl;
	//Add the likelihood of each process. TODO test wether I should do an all reduce or not
	com.Allreduce(MPI::IN_PLACE, &lik, 1, MPI::LONG_DOUBLE, MPI::SUM);
	//std::cout << "after allreduce" << std::endl;
	lik = lik / double(N);
//	std::cout << std::setprecision(8) << "I am id " << id << " and lik is " << lik  << std::endl << std::flush;
	//std::cout << "I am id " << id << " and N is " << N << std::endl << std::flush;
	return std::log(lik);
}

void EnkfMPI::saveSamples(std::string filename) {
	mat * temp = new mat(state.getDim() , N );
	*temp = state.getSamplesAtRoot();
	if ( id ==0 ) {
		temp->save(filename, raw_ascii);
	}
}

//Update step. TODO Possible mistake here. evaluateh shouldn't be corrupted by noise
void EnkfMPI::update(const colvec& measurement, const colvec&  parameters ) {
	//Eqs 40 - 45 Nonlinear dynamics paper
	for (int i = first; i <= last; i++){
		pmeasurements.setSampleAt(i, ss.evaluateh(state.getSampleAt(i), parameters)); //Eq. 40
	}
	Pxd = state.getCrossCovarianceAtRoot( pmeasurements ); //Eq. 43
	Pdd = pmeasurements.getCovarianceAtRoot();						 //Eq. 44

	if (id == 0) {
		kalman = Pxd * inv(Pdd);	//Eq. 45
	}
	com.Bcast(kalman.memptr(), kalman.n_rows * kalman.n_cols, MPI::DOUBLE, 0);
	for (int i = first; i <= last; i++){
		state.setSampleAt(i, state.getSampleAt(i) + kalman*(measurement - pmeasurements.getSampleAt(i)) );
	}
}

void EnkfMPI::getStateXML(std::ofstream & se) {
	//Get each particle and its id
	int j,k;
	int l = state.getSampleAt(first).n_rows;
	colvec temp;
	mat tempCov;
	//Save every 50 particles
	temp = state.getMeanAtRoot();
	tempCov = state.getCovarianceAtRoot();
	if (id == 0) {
	se << "<mean>";
	for (j = 0; j < l; j ++) {
		se << temp[j] << " ";
	}
	se << "</mean>" << std::endl << "<covariance>";
	for (j = 0; j < l; j ++) {
			for (k = 0; k < l; k ++) {
				se << tempCov(j,k) << " ";
			}
	}
	se << "</covariance>" << std::endl;
}
	for (int i = first; i <= last; i = i + 50){
		se << "<p>" << std::endl;
		se << "<id>" << i+first << "</id>" << std::endl << "<val>" ;
		temp = state.getSampleAt(i);
		for (j = 0; j < l; j ++) {
			se << temp[j] << " ";
		}
		se << "</val>" << std::endl;
		se << "</p>" << std::endl;
	}
}


/*****************************************************************
Ensemble Kalman filter MPI - Sherman-Morrison Update
*****************************************************************/

//EnkfSMMPI::EnkfSMMPI() : EnkfMPI::EnkfMPI() {};


/*
*	Here we proposed a different way to perform the update. Instead of inverting the matrix
* We solve the linear problem.
* X^a = X^f + P^f H^T ( H P^f H + Gamma)^{-1} [D - ]
*
*/
/*
EnkfSMMPI::EnkfSMMPI( colvec& _mean, mat& _cov, int N, statespace _ss, const rowvec& _H, const mat& _R, const MPI::Intracomm _com) : EnkfMPI::EnkfMPI(_mean, _cov, N, _ss, _com) {
	//Nothing to do here
	//Y = zeros<mat>(dim, N);
	//dpVY = zeros<vec>(N);
}

//Update step. Sherman-Morrison in parallel
void EnkfSMMPI::update(const colvec& measurement, const colvec&  parameters ) {
	//Every process compute the Kalman gain
	//P = getCovariance();
	//kalman = P*trans(H)*inv(H*P*trans(H)+R);

	//shermanMorrisonMPIPreProc( R , const mat& U, const mat& V, Y, dpVY, com)

	//Generate the data vector, need to make sure each process don't generate from the same seed
	Gaussian meas = Gaussian(measurement, R);
	colvec z;


	for (int i = first; i <= last; i++){
		z = meas.sample() - ss.evaluateh(state.getSampleAt(i), parameters);
		state.setSampleAt(i, state.getSampleAt(i) + kalman*z);
	}
}
*/
/*****************************************************************
PF-Bootstrap CLASS
*****************************************************************/
//PF::PF( colvec& _mean,mat& _cov, int N, statespace _ss, const mat& _R, MPI_Comm _com): filter::filter(_ss) {
PF::PF() : filter::filter() {
	//do nothing
}

PF::PF( colvec& _mean,mat& _cov, int _N, statespace _ss): filter::filter(_ss) {
	//Creates the samples based on Gaussian distribution
	state = WeightedSamples(_mean, _cov, _N);
	first = 0;
	last = _N;
	N = _N;
	initialMean = _mean;
	initialCov = _cov;
}

void PF::forecast(const colvec&  parameters) {
	//Monte-Carlo Forecast
	for (int i = first; i < last; i++) {
		state.setSampleAt(i, ss.evaluatef( state.getSampleAt(i) , parameters ) );
	}
}

void PF::update(const colvec& _d, const colvec&  parameters ) {
	double tempWeight;
	double sumWeight = 0.0;
	double logoldw,logneww;
	for (int i = first; i < last; i++ ) {
		logoldw = log(state.getWeightAt(i));
		logneww = logoldw+ss.evaluateloglikelihood(_d, state.getSampleAt(i) , parameters );
		tempWeight = std::exp(logneww);
		state.setWeightAt(i, tempWeight);
		sumWeight += tempWeight;
	}



	//Normalize the weights
	for (int i = first; i < last; i++ ) {
		state.normalizeWeightAt(i, sumWeight);
	}

	if (state.getEffectiveSize() < 0.33*double(N)) {
		state.systematic_resampling();
	}
}

double PF::logLikelihoodOfMeasurement(const colvec& measurement) {
	return logLikelihoodOfMeasurement(measurement, colvec() );
}


double PF::logLikelihoodOfMeasurement(const colvec& _d, const colvec&  parameters) {
	long double w;
	long double lik = 0.0;
	long double particlelogLikelihood;
	for (int i = first; i < last; i++) {
		w = state.getWeightAt(i);
	  particlelogLikelihood = ss.evaluateloglikelihood(_d, state.getSampleAt(i) , parameters );
		lik += std::exp(log(w) + particlelogLikelihood);
	}
	return log(lik);
}

void PF::reset() {
	state = WeightedSamples(initialMean, initialCov, N);
}

void PF::print() {
	std::cout << "PF:" << std::endl;
	state.getMean().print("Mean:");
	state.getCovariance().print("Covariance:");
	std::cout << std::endl;
}

colvec PF::getCurrentState() {
	colvec tempMean = state.getMean();
	mat tempCov = state.getCovariance();
	int n = tempMean.n_rows;
	colvec temp = colvec(2*n);
	for (int i = 0; i < n; i++) {
		temp(2*i) = tempMean(i);
		temp(2*i+1) = tempCov(i,i);
	}
	return temp;
}

void PF::getStateXML(std::ofstream & se) {
	int j,k;
	int l = state.getSampleAt(first).n_rows;
	colvec temp;
	mat tempCov;
	//Save every 50 particles
	temp = state.getMean();
	tempCov = state.getCovariance();
	se << "<mean>";
	for (j = 0; j < l; j ++) {
		se << temp[j] << " ";
	}
	se << "</mean>" << std::endl << "<covariance>";
	for (j = 0; j < l; j ++) {
			for (k = 0; k < l; k ++) {
				se << tempCov(j,k) << " ";
			}
	}
	se << "</covariance>" << std::endl;
	for (int i = first; i <= last; i = i + 50){
		se << "<p>" << std::endl;
		se << "<id>" << i << "</id>" << std::endl << "<val>" ;
		temp = state.getSampleAt(i);
		for (j = 0; j < l; j ++) {
			se << temp[j] << " ";
		}
		se << "</val>" << std::endl;
		se << "<w>" << state.getWeightAt(i) << "</w>" << std::endl;
		se << "</p>" << std::endl;
	}
}

PF::~PF() {}

/*****************************************************************
Ensemble Kalman filter MPI
*****************************************************************/
PFMPI::PFMPI() {}

PFMPI::PFMPI( colvec& _mean, mat& _cov, int _N, statespace  _ss, const MPI::Intracomm _com) {
	com = _com;	//Communicator
	nproc = com.Get_size();
	id = com.Get_rank();
	N = _N;
	int sub = N / nproc;
	first = id*sub; 			//Index of the first element belonging to the current process
	last = first + sub - 1;		//Index of the last element belonging to the current process (inclusive)
	state = WeightedSamplesMPI(_mean, _cov, N, com);
	ss = _ss;
	initialMean = _mean;
	initialCov = _cov;
}


void PFMPI::reset() {
	state = WeightedSamplesMPI(initialMean, initialCov, N, com);
}

/**
 * Parallel forecast for Enkf. Each process has it own first and last
 * @param parameters Parameter vector
 */

void PFMPI::forecast(const colvec&  parameters) {
	for (int i = first; i <= last; i++) {
		state.setSampleAt(i, ss.evaluatef( state.getSampleAt(i) , parameters ) );
	}
}

//Saves the state at specific timestep as following
//[state] w_0
//....
//[state_n] w_n
colvec PFMPI::getCurrentState() {
	//colvec tempMean = state.getMeanAtRoot();
	//mat tempCov = state.getCovarianceAtRoot();
	colvec tempMean = state.getMean();
	mat tempCov = state.getCovariance();
	int n = tempMean.n_rows;
	colvec temp = colvec(2*n);
	for (int i = 0; i < n; i++) {
		temp(2*i) = tempMean(i);
		temp(2*i+1) = tempCov(i,i);
	}
	return temp;
}

void PFMPI::getStateXML(std::ofstream & se) {
	int j,k;
	int l = state.getSampleAt(first).n_rows;
	colvec temp;
	colvec tempMode;
	mat tempCov;
	//Save every 50 particles
	temp = state.getMeanAtRoot();
	tempMode = state.getModeAtRoot();
	tempCov = state.getCovarianceAtRoot();
	if (id == 0) {
	se << "<mean>";
	for (j = 0; j < l; j ++) {
		se << temp[j] << " ";
	}
	se << "</mean>" << std::endl << "<covariance>";
	for (j = 0; j < l; j ++) {
			for (k = 0; k < l; k ++) {
				se << tempCov(j,k) << " ";
			}
	}
	se << "</covariance>" << std::endl << "<mode>";
	for (j = 0; j < l; j ++) {
		se << tempMode[j] << " ";
	}
	se << "</mode>" << std::endl;
}
	for (int i = first; i <= last; i = i + 50){
		se << "<p>" << std::endl;
		se << "<id>" << i + first << "</id>" << std::endl << "<val>" ;
		temp = state.getSampleAt(i);
		for (j = 0; j < l; j ++) {
			se << temp[j] << " ";
		}
		se << "</val>" << std::endl;
		se << "<w>" << state.getWeightAt(i) << "</w>" << std::endl;
		se << "</p>" << std::endl;
	}
}


double PFMPI::logLikelihoodOfMeasurement(const colvec& measurement) {
	return logLikelihoodOfMeasurement(measurement, colvec() );
}

double PFMPI::logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters) {
	//In the case of sample based filter, it is a sum of the likelihood
	long double lik = 0.0;
	long double w = 0;
	long double particlelogLikelihood;

	for (int i = first; i <= last; i++) {
   	w = state.getWeightAt(i);
	  particlelogLikelihood = ss.evaluateloglikelihood(measurement, state.getSampleAt(i) , parameters );
		lik += std::exp(log(w) + particlelogLikelihood);
	}

	com.Allreduce(MPI::IN_PLACE, &lik, 1, MPI::LONG_DOUBLE, MPI::SUM);
	return std::log(lik);
}

void PFMPI::saveSamples(std::string filename) {
	mat * temp = new mat(state.getDim() , N );
	*temp = state.getSamplesAtRoot();
	if ( id ==0 ) {
		temp->save(filename, raw_ascii);
	}
}

WeightedSamplesMPI& PFMPI::getState() {
	return state;
}

void PFMPI::update(const colvec& measurement, const colvec&  parameters ) {
	//Update the weights - here we don't use the full state to create the KDE. Each process estimate will be based on the local samples
	double tempWeight;
	double sumWeight = 0.0;
	double logoldw,logneww;
	for (int i = first; i <= last; i++ ) {
		logoldw = log(state.getWeightAt(i));
		logneww = logoldw+ss.evaluateloglikelihood(measurement, state.getSampleAt(i) , parameters );
		tempWeight = std::exp(logneww);
		state.setWeightAt(i, tempWeight);
		sumWeight += tempWeight;
	}
	//Get the total weight
	com.Allreduce(MPI::IN_PLACE, &sumWeight, 1, MPI::DOUBLE, MPI::SUM);

	//Normalize the weights
	for (int i = first; i <= last; i++ ) {
		state.normalizeWeightAt(i, sumWeight);
	}
	//double seff = state.getEffectiveSize();
	//if (id == 0) {
	//std::cout << "effectiveSize is " << seff << std::endl;}
	if (state.getEffectiveSize() < 0.33*double(N)) {
		state.systematic_resampling();
	}

}

/*****************************************************************
Ensemble Kalman filter MPI
*****************************************************************/

PFEnkfMPI::PFEnkfMPI() : PFMPI::PFMPI() {};

PFEnkfMPI::PFEnkfMPI( colvec& _mean, mat& _cov, int _N, statespace  _ss, const MPI::Intracomm _com) : PFMPI(_mean, _cov, _N, _ss, _com) {
	int m = _ss.getMeasurementVectorSize();

  pmeasurements = WeightedSamplesMPI( zeros<colvec>(m) , eye<mat>(m,m), N, com); //Pertubed measurements
	//Reserve memory space for all procs
	kalman = zeros<mat>(_ss.getStateVectorSize(), _ss.getMeasurementVectorSize());
}

//Update step. TODO Possible mistake here. evaluateh shouldn't be corrupted by noise
void PFEnkfMPI::update(const colvec& measurement, const colvec&  parameters ) {
	//Eqs 50 - 60 Nonlinear dynamics paper

	//1) Make a copy of the samples and their weights
	old = state;
	//2) Run the EnKF update, using noisy measurements.
	for (int i = first; i <= last; i++){
		pmeasurements.setSampleAt(i, ss.evaluateh(state.getSampleAt(i), parameters)); //Eq. 40
	}
	Pxd = state.getCrossCovarianceAtRoot( pmeasurements ); //Eq. 43
	Pdd = pmeasurements.getCovarianceAtRoot();						 //Eq. 44

	if (id == 0) {
		kalman = Pxd * inv(Pdd);	//Eq. 45
	}

	com.Bcast(kalman.memptr(), ss.getStateVectorSize() * ss.getMeasurementVectorSize(), MPI::DOUBLE, 0);
	for (int i = first; i <= last; i++){
		state.setSampleAt(i, state.getSampleAt(i) + kalman*(measurement - pmeasurements.getSampleAt(i)) );
		//Reset the weights
		state.setWeightAt(i, 1.0/double(N));
	}

	//Update the weights - here we don't use the full state to create the KDE. Each process estimate will be based on the local samples
	//Gather the sample
	old.shareSamples();
	state.shareSamples();
	logpriors = old.evaluate_kde( state.getSamples(), true );
	logproposals = state.evaluate_kde( state.getSamples(), false );

	double tempWeight;
	double sumWeight = 0.0;
	double logoldw,logneww,logprior, logproposal;
	for (int i = first; i <= last; i++ ) {
		logprior = logpriors[i-first];
		logproposal = logproposals[i-first];
		logoldw = log(old.getWeightAt(i));
		logneww = logoldw+ss.evaluateloglikelihood(measurement, state.getSampleAt(i) , parameters ) + logprior - logproposal;
		tempWeight = std::exp(logneww);
		state.setWeightAt(i, tempWeight);
		sumWeight += tempWeight;
	}
	//Get the total weight
	com.Allreduce(MPI::IN_PLACE, &sumWeight, 1, MPI::DOUBLE, MPI::SUM);

	//Normalize the weights
	for (int i = first; i <= last; i++ ) {
		state.normalizeWeightAt(i, sumWeight);
		pmeasurements.setWeightAt(i, state.getWeightAt(i)); //Copy the weights to pmeasurements
		//std::cout << "New weight for particle " << i << " is " << state.getWeightAt(i) << std::endl;
	}

	if (state.getEffectiveSize() < 0.33*double(N)) {
		state.systematic_resampling();
	}

}

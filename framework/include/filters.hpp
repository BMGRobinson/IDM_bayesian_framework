/*
 * TODO: add a forecast with no parameters
 *
 */

#ifndef FILTERS_HPP_
#define FILTERS_HPP_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include "armadillo"
#include <mpi.h>
#pragma GCC diagnostic pop
#include <iomanip>
#include "pdf.hpp"
#include "samples.hpp"
#include "statespace.hpp"

using namespace arma;
class pdf;

/* helper methods */
//subroutine ShermanMorrisonSolver(A0, U, V, d, z, Y, dpVY, nmeas, N)
/* Solve a system of the form (A + UV^T) z = d */
//void shermanMorison(const mat& A, const mat& U, const mat& V, const vec& z, const vec& d, const mat& Y, const vec& dpVY );
void shermanMorrisonMPIPreProc(const mat& A, const mat& U, const mat& V, mat& Y, vec& dpVY, const MPI::Intracomm _com);
void simplifiedShermanMorrison(const mat& A, const mat& V, const mat& Y, const vec& dpVY, colvec& z, const colvec& d, const MPI::Intracomm _com);



/**
Filter class representing a general filter. Is overloaded. It has three main functions. Predict (forecast), Update and evaluate the likelihood p(D | M, Parameters)
*/

class state_estimator {
	public:
		state_estimator();
		state_estimator( statespace _ss);
		virtual ~state_estimator();
		virtual void forecast(const colvec&  parameters) = 0;
		void saveToFile(const bool save, std::string filename, int _timestep);
		virtual double state_estimation( const mat& data, const colvec&  parameters , bool save ) = 0;
		//Returns a vector of the errors
		virtual colvec getSquareError( const mat & data, const mat & reference, const colvec& parameters) = 0;
		void disableSaveToFile();
		virtual void reset() = 0;
		virtual double logLikelihoodOfMeasurement(const colvec& measurement) = 0;
		virtual double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters) = 0; 		/**< Evaluate likelihood of the filter at current time step */
		virtual double logLikelihood(const mat& data, const colvec&  parameters ) = 0;

	protected:
		int id;
		statespace ss;
		bool save_to_file;
		int save_to_file_step ;
		virtual colvec getCurrentState() = 0;
		std::string state_estimation_filename;
		bool output;
};

class filter : public state_estimator {
public:
  filter(); //default constructor
	filter( statespace _ss);
	virtual ~filter();
	double state_estimation( const mat& data, const colvec&  parameters , bool save );
	colvec getSquareError( const mat & data, const mat & reference, const colvec& parameters);
	virtual void update(const colvec& measurement, const colvec&  parameters) = 0; 					/**< Update step of the filter */
	double logLikelihood(const mat& data, const colvec&  parameters );
	virtual void print() = 0;
protected:
	virtual void getStateXML(std::ofstream & se ) = 0;
};

//State estimator with a deterministic model
class Deterministic : public state_estimator {
public:
	Deterministic(); //default constructor
	Deterministic( const Deterministic &other ); //copy constructor
	Deterministic( colvec& initial_state, statespace _ss );
	double logLikelihood(const mat& data, const colvec&  parameters );
	double logLikelihoodOfMeasurement(const colvec& measurement);
	double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters);
	double state_estimation( const mat& data, const colvec&  parameters , bool save );
	colvec getSquareError( const mat & data, const mat & reference, const colvec& parameters);
	void forecast(const colvec&  parameters);
	void reset();
	void print();
	colvec& getState();
	colvec getCurrentState();
	~Deterministic();
private:
	colvec firstState;
	colvec state;
};


/**
Serial implementation of the Extended Kalman Filter
*/

class Ekf : public filter {
public:
	Ekf(); //default constructor
	Ekf( const Ekf &other ); //copy constructor
	Ekf( Gaussian* initial_state, statespace _ss, const mat& _Q, const mat& _R);
	double logLikelihoodOfMeasurement(const colvec& measurement);
	double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters);
	void forecast(const colvec&  parameters);
	void update(const colvec& measurement, const colvec&  parameters );
	bool checkMatrixSize();
	void reset();
	void print();
	mat getKalmanGain();
	Gaussian * getState();
	colvec getCurrentState();
	void getStateXML(std::ofstream & se );
	void setMeasurementCovariance( const mat& _R);
	~Ekf();
private:
	//private values to speed up computation
	mat Q; //model covariance
	mat R; //measurement covariance
	mat S;
	mat K;    																							    /**< Kalman Gain */
	Gaussian firstState;
	Gaussian state;
};

class PF : public filter {
public:
	PF();
	PF( colvec& _mean, mat& _cov, int N, statespace _ss);
	double logLikelihoodOfMeasurement(const colvec& measurement);
	double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters);
	//double evaluate_integral(const colvec& measurement);
	void forecast(const colvec&  parameters);
	void update(const colvec& measurement, const colvec&  parameters );
	void reset();
	colvec getCurrentState();
	void print();
	~PF();
protected:
	void getStateXML(std::ofstream & se );
	int getDim();
	int first;
	int last;
	int N;
	colvec initialMean;
	mat initialCov;

private:
	WeightedSamples state;
};

//Basic implementation of the Ensemble Kalman filter
class Enkf : public filter {
public:
	Enkf();
	Enkf( colvec& _mean, mat& _cov, int _N, statespace _ss ); //, const rowvec& _H, const mat& _R);
	double logLikelihoodOfMeasurement(const colvec& measurement);
	double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters);
	void forecast(const colvec&  parameters); //Forecast step; embarassingly parallel
	void update(const colvec& measurement, const colvec&  parameters );	//Update step
	void reset();
	void saveSamples(std::string filename);
	colvec getCurrentState();
	void print();
	Samples& getState();
	~Enkf();
protected:
	void getStateXML(std::ofstream & se );
	int first; 	//Index of the first element belonging to the current process
	int last;	//Index of the last element belonging to the current process
	mat Pdd;    //perturbed measurements covariance
	mat Pxd;		//Cross covariance
	mat kalman;			//Kalman gain matrix
	int N;
	colvec initialMean;
	mat initialCov;
private:
	Samples state; 		//Current ensemble state
	Samples pmeasurements; //Perturbed measurements

};

class EnkfMPI : public Enkf {
public:
	EnkfMPI();
	EnkfMPI( colvec& _mean, mat& _cov, int N, statespace  _ss, const MPI::Intracomm _com);
	double logLikelihoodOfMeasurement(const colvec& measurement);
	double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters);
	void forecast(const colvec&  parameters);
	void update(const colvec& measurement, const colvec&  parameters );	//Update step.
	void saveSamples(std::string filename);
	void reset();
	colvec getCurrentState();
	SamplesMPI& getState();
	void getStateXML(std::ofstream & se );
protected:
	int nproc;
	MPI::Intracomm com;
	SamplesMPI state;
	SamplesMPI pmeasurements;
};
//Using Sherman-Morrison formula for the update. Almost matrix free. Not yet implemented
/*
class EnkfSMMPI : public EnkfMPI {
public:
	EnkfSMMPI();
	EnkfSMMPI( colvec& _mean, mat& _cov, int N, statespace _ss, const rowvec& _H, const mat& _R, const MPI::Intracomm _com);
	void update(const colvec& measurement, const colvec&  parameters );	//Update step. Using Sherman-Morrison formula
private:
	mat Y;
	vec dpVY;
};
*/

class PFMPI : public PF {
public:
	PFMPI();
	PFMPI( colvec& _mean, mat& _cov, int N, statespace  _ss, const MPI::Intracomm _com);
	double logLikelihoodOfMeasurement(const colvec& measurement);
	double logLikelihoodOfMeasurement(const colvec& measurement, const colvec&  parameters);
	void forecast(const colvec&  parameters);
	void update(const colvec& measurement, const colvec&  parameters );	//Update step.
	void saveSamples(std::string filename);
	void reset();
	colvec getCurrentState();
	WeightedSamplesMPI& getState();
	void getStateXML(std::ofstream & se );
protected:
	int nproc;
	MPI::Intracomm com;
	WeightedSamplesMPI state;
};

class PFEnkfMPI : public PFMPI {
public:
	PFEnkfMPI();
	PFEnkfMPI( colvec& _mean, mat& _cov, int N, statespace  _ss, const MPI::Intracomm _com);
	void update(const colvec& measurement, const colvec&  parameters );	//Update step.
protected:
	mat kalman;
	mat Pxd;
	mat Pdd;
	colvec logpriors, logproposals;
	WeightedSamplesMPI old;
	WeightedSamplesMPI pmeasurements;
};

/*
//Right now can only be used for additive Gaussian noise
class Ukf: public filter {
public:
	Ukf( Gaussian* initial_state, statespace* ss, double dt, double _alpha, double _beta, double _kappa);
	void forecast();
	void update(const colvec& d );
	double evaluate_integral(const colvec& measurement);
	void resetFilter();
	void print();
	void sigma();
	Gaussian * getState();
	colvec getMean();
	mat getCovariance();
	//void initializeFilter( problem* _problem);
	~Ukf();
private:
	mat sigma_points;
	mat hsigma_points;
	colvec sigma_weights_m;
	colvec sigma_weights_c;
	int getDim();
	int num_sigma_points;
	double n; //double version of sigma points, used to calculate the weights
	double alpha;
	double beta;
	double kappa;
	double lambda;
	//private values to speed up computation
	mat S;
	mat K;
	mat cholP;    																							//Kalman Gain
	Gaussian * firstState;
	Gaussian * state;
};

class PFEnkf : public PF {
public:
	PFEnkf( Samples* initial_state, statespace* ss, double dt, MPI_Comm _com);
	void update( const colvec& measurement );
private:
	mat K;
	Samples * oldstate;
	void enkf_update( const colvec& measurement );
};

class UPF: public PF {
public:
	UPF( Samples* initial_state, statespace* ss, double dt, MPI_Comm _com);
	void forecast();
	Samples * oldstate;
	void update( const colvec& measurement );
	//void ukf_update( const colvec& measurement );
private:
	std::vector<Ukf> ukfs;
};
*/
/**
Serial implementation of the Ensemble Kalman Filter
*/
/*
class Enkf: public filter {
public:
	//default constructor
	Enkf(int _ensemble_size );
	Enkf(int _ensemble_size, mat& intial_ensemble );
	Enkf(int _ensemble_size, colvec& _state, mat& _cov);
	void predict( problem* _problem);
	void update(problem* _problem, colvec& measurement );
	void load_ensemble_x( mat& new_ensemble);
	void load_ensemble_noise(mat& ensemble_noise);
	void load_measurement_noise(mat& ensemble_noise);
	double evaluate_integral(const Measurement& meas_model,const colvec& measurement);
	void calculateCovariance();
	void initializeFilter( problem* _problem);
	void resetFilter(problem* _problem);
protected:
	mat ensemble;
	int ensemble_size;
	mat ensemble_noise;
	mat measurement_noise;
	//void calculateCovariance();

	//matrix that are used in calculations so they don't need to be created each time
	//Contains possible observation with no noise
	mat t_obs;
	//Contains
	mat D;
	//N x N matrix containing the weight of each sample (1/N)
	mat In;
	mat ensemble_bar;
	mat ensemble_prime;
	mat t_obs_prime;
	mat S;
	mat C;
	mat K;
};

class PF :public filter {
private:
	double bandwidth_ensemble_size;
public:
	//default constructor
	PF(int _ensemble_size, mat& initial_ensemble );
	PF(int _ensemble_size, colvec& _state, mat& _cov);
	void predict( problem* _problem);
	void update(problem* _problem, colvec& measurement );
	void load_ensemble_x( mat& new_ensemble);
	void load_ensemble_noise(mat& ensemble_noise);
	void load_measurement_noise(mat& ensemble_noise);
	//double getLikelihood(problem* _problem);
	//double evaluate_prob_of_value(T value, int flag);
	double evaluate_integral(const Measurement& meas_model,const colvec& measurement);
	void update_state_covariance();
	void resetFilter(problem* _problem);
	//mat getEnsemble() { return ensemble; };
	//mat getWeights() { return ensemble_w; };
	colvec getState();
	mat printFilter();
	mat getCovariance();
	void setMPI( MPI::Intracomm _com ) { com_internal = _com; };
	//double evaluate_gaussian_prob( const colvec& state, const colvec& mean, const mat& cov, const mat& cov_inv, const double cov_det);
	~PF();
protected:
	MPI::Intracomm com_internal;
	void resample( double ratio );
	void multimonial_resampling( double ratio );
	mat ensemble;
	int ensemble_size;
	mat ensemble_noise;
	mat measurement_noise;
	rowvec ensemble_w;
	void calculateCovariance();
	double effective_ensemble_size;
};

//particle filter with EnKF proposal

class PF_EnKF : public PF {
public:
		PF_EnKF(int _ensemble_size, mat& initial_ensemble );
		PF_EnKF(int _ensemble_size, colvec& _state, mat& _cov);
		void update(problem* _problem, colvec& measurement );
		void copyToPrevious();
		void initializeFilter( problem* _problem);
private:
		mat previous_ensemble;
		mat t_obs;
		mat D;
		mat ensemble_bar;
		mat ensemble_prime;
		mat t_obs_prime;
		mat t_obs_bar;
		mat S;
		mat C;
		mat K;
};

class PF_mpi : public PF {
public:
		PF_mpi(int _ensemble_size, mat& initial_ensemble, MPI::Intracomm com );
		PF_mpi(int _ensemble_size, colvec& _state, mat& _cov,MPI::Intracomm com);
		void predict(problem* _problem);
		void update(problem* _problem, colvec& measurement );
		double evaluate_integral(const Measurement& meas_model,const colvec& measurement);
		void copyToPrevious();
		void initializeFilter( problem* _problem);
private:
		int chunk;
		void synchronize(problem* _problem);
		void multimonial_resampling_mpi( double ratio, problem* _problem );
		void rejection_resampling_mpi();
		colvec getMean_mpi();
		mat previous_ensemble;
		MPI::Intracomm com;
};

class PF_EnKF_mpi : public PF {
public:
		PF_EnKF_mpi(int _ensemble_size, mat& initial_ensemble, MPI::Intracomm com );
		PF_EnKF_mpi(int _ensemble_size, colvec& _state, mat& _cov,MPI::Intracomm com);
		void predict(problem* _problem);
		void update(problem* _problem, colvec& measurement );
		double evaluate_integral(const Measurement& meas_model,const colvec& measurement);
		void copyToPrevious();
		void initializeFilter( problem* _problem);
private:
		int chunk;
		void synchronize(problem* _problem);
		void multimonial_resampling_mpi( double ratio, problem* _problem );
		colvec getMean_mpi();
		mat previous_ensemble;
		mat t_obs;
		mat D;
		mat ensemble_bar;
		mat ensemble_prime;
		mat t_obs_prime;
		mat t_obs_bar;
		mat S;
		mat C;
		mat K;
		MPI::Intracomm com;
};

class FP: public filter {
public:
	FP( mat& _domain,vec& _pdf, int length, double _delta_x );
	int length;
	double delta_x;
	mat D;
	mat domain;
	vec pdf;
	vec initial;
	colvec getState();
	mat getCovariance();
	double evaluate_integral(const Measurement& meas_model,const colvec& measurement);
	void predict(problem* _problem);
	void update(problem* _problem, colvec& measurement );
	void resetFilter(problem* _problem);
	~FP();
};

*/

#endif

#include <pdf.hpp>
#include <samples.hpp>
#include <filters.hpp>
#include <statespace.hpp>
#include <wrapper.hpp>
#include <armadillo>
#include <cmath>
#include <iomanip>      // std::setprecision
#include <functional>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h> //to load dynamic library

using namespace arma;
//Set constant value of timestep
const double dt = 0.001;
//How many steps between measurements
const int fStepsBetweenMeasurements = 100;

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	double mu = parameters[0];
	double amp = parameters[1];
	double omega = parameters[2];

	double x1 = state[0];
	double x2 = state[1];

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*mu*(1.0-x1*x1)*x2 - dt*x1 + dt*amp*sin(omega * time);
	return temp;
}

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {
	double mu = parameters[0];

	double x1 = state[0];
	double x2 = state[1];
	return {{1.0,dt},{dt*(mu*(-2.0*x1)*x2-1.0), 1.0+dt*mu*(1.0-x1*x1)}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = parameters[3];
	return {{0.0,0.0},{0.0,std::sqrt(dt)*sigma}};
}

mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0,0.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

int main(int argc, char* argv[]) {
	
		

	/**** Ideally you would call this block only once ****/
	colvec initialState = {0.0, -0.2};
	mat initialStateVariance = {{0.1, 0.0}, {0.0, 0.1}};
	mat modelCov = {{0.0, 0.0},{0.0, 1.0}};
	mat data, modelMeasVariance;
  	data.load("./Data/data.dat");
	modelMeasVariance.load("./Data/variance.dat",raw_ascii);

	//Statespace
	statespace ss = statespace( model1, dfdx, dfde, _h, dhdx, dhde, dt, 2, 1);
	ss.setMeasCov( modelMeasVariance );
	ss.setForecastStepsBetweenMeasurements(fStepsBetweenMeasurements);

	//Build state estimator
    Gaussian * is = new Gaussian(initialState, initialStateVariance  );
    state_estimator * se = new Ekf( is ,ss, modelCov, modelMeasVariance );
	

	/**** Ideally you would call this block only once ****/

	//Input that will change
	colvec parameters = {4.2, 0.1, 0.1, 0.1};
	
	double loglik =  se -> logLikelihood( data, parameters );
	std::cout << "loglikelihood = " << loglik << std::endl;
	return loglik;
	
	
}

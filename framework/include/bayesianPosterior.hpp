#ifndef BAYESIANPOSTERIOR_HPP_
#define BAYESIANPOSTERIOR_HPP_

#include "filters.hpp"
#include "pdf.hpp"
#include <vector>
#include "armadillo"
using namespace arma;
typedef std::function<double(const colvec &)> logPosteriorFunc;
 /*
	Wrapper around state_estimator to calculate the unormalized log likelihod.
	Combines the prior, the data and the likelihood function
	Can also include a specific function to calculate the value
*/

class bayesianPosterior {
	private:
		mat data;
		state_estimator * se; //Pointer to type state_estimator
		std::vector<pdf1d*> priors; //vector of abstract pdfs1d
		logPosteriorFunc customFunc;
		bool customFunction;

	public:
		bayesianPosterior();
		long double evaluate( const colvec& parameters );
		long double evaluateLogLikelihood( const colvec& parameters );
		long double evaluatePrior( const colvec& parameters );
		long double evaluate();
		bayesianPosterior( logPosteriorFunc _func );
		bayesianPosterior(const mat& data, state_estimator* se);
		bayesianPosterior(const mat& data, state_estimator* se, const std::vector<pdf1d*>& _priors);
		//long double quadrature1D( const int Nx, const double xl, const double xr);
		mat posterior1D( const int Nx, const double xl, const double xr);
		mat posterior2D( const int Nx, const double xl, const double xr, const int Ny, const double yl, const double yr);
		std::vector<pdf1d*> getPriors();
		void setPriors(const std::vector<pdf1d*>& _priors);
};
#endif

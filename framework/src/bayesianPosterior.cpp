#include "bayesianPosterior.hpp"

bayesianPosterior::bayesianPosterior() {}

bayesianPosterior::bayesianPosterior( logPosteriorFunc _func ) {
	customFunction = true;
	customFunc = _func;
}

bayesianPosterior::bayesianPosterior(const mat& _data, state_estimator * _se ) {
	customFunction = false;
	data = _data;
	se = _se;
}

bayesianPosterior::bayesianPosterior(const mat& _data, state_estimator * _se , const std::vector<pdf1d*>& _priors):bayesianPosterior(_data,_se) {
	priors = _priors;
}

void bayesianPosterior::setPriors(const std::vector<pdf1d*>& _priors) {
	priors = _priors;
}

std::vector<pdf1d*> bayesianPosterior::getPriors() {
	return priors;
}

/**
 * Returns the unormalized posterior density.
 * This function should be called when the parameter vector doesn't need to be specified.
 * @return log( p(parameters | data) p(parameters))
 */
long double bayesianPosterior::evaluate() {
	//std::cout << " You are calling evaluate() with no arguments." << std::endl;
	if (customFunction) {
		return customFunc( colvec () );
	} else {
		return se->logLikelihood( data, colvec() );
	}
}
/**
 * returns the unormalized posterior density.
 * @param  parameters Parameter vector at which the unormalized posterior density is evaluated.
 * @return            log( p(parameters | data) p(parameters))
 */
long double bayesianPosterior::evaluate( const colvec& parameters ) {
	//Is there a prior specified?
	long double logPrior = 0.0;
	long double logLik = 0.0;
	if ( !priors.empty() ) {
		for (int i = 0; i < parameters.n_rows; i ++) {
			logPrior += priors[i]->getLogDensity(parameters[i]);
		}
	}
	//If prior is out of bounds, no need to evaluate logLikelihood
	if ( std::isnan(logPrior) ) {
		return logPrior;
	}

	//Case where the posterior function is specified (no state estimation!)
	if (customFunction) {
		logLik =  customFunc( parameters );
	} else {
		logLik = se->logLikelihood( data, parameters);
	}
	return logLik + logPrior;
}


/**
 * returns the log logLikelihood density.
 * @param  parameters Parameter vector at which the unormalized posterior density is evaluated.
 * @return            log( p(parameters | data) p(parameters))
 */
long double bayesianPosterior::evaluateLogLikelihood( const colvec& parameters ) {
	long double logLik = 0.0;

	//Case where the posterior function is specified (no state estimation!)
	if (customFunction) {
		logLik =  customFunc( parameters );
	} else {
		logLik = se->logLikelihood( data, parameters);
	}
	return logLik;
}


long double bayesianPosterior::evaluatePrior( const colvec& parameters ) {
	long double logPrior = 0.0;
	if ( !priors.empty() ) {
		for (int i = 0; i < parameters.n_rows; i ++) {
			logPrior += priors[i]->getLogDensity(parameters[i]);
		}
	}
	return logPrior;
}

//Use trapezoidal method to estimate the evidence
mat bayesianPosterior::posterior1D( const int Nx, const double xl, const double xr) {
	mat posterior = zeros<mat>(1, Nx+1);

	double h = (xr - xl)/double(Nx);
	double result = 0.0;
	double temp;
	colvec param = zeros<colvec>(1);
	//Nx + 1 evaluations
	param[0] = xl;
	temp = std::exp(evaluate(param));
	posterior(0,0) = temp;
	result = temp/2.0;
	for (int i = 1; i < Nx; i ++) {
		param[0] += h;
		temp  = std::exp(evaluate(param));
		posterior(0,i) = temp;
		result += temp;
	}
	param[0] = xr;
	temp = std::exp(evaluate(param));
	posterior(0,Nx) = temp;
	result += temp/2.0;
	result *= h;
	double logev = log(result);
	std::cout << "log evidence is " << logev << std::endl;
	//Normalize by the evidence
	for (int i = 0; i <= Nx; i ++) {
		posterior(0,i) = std::exp(log(posterior(0,i)) - logev);
	}
	return posterior;
}

mat bayesianPosterior::posterior2D( const int Nx, const double xl, const double xr, const int Ny, const double yl, const double yr) {
	//Initialize matrix containing posterior distribution
	mat posterior = zeros<mat>(Nx+1, Ny+1);

	double hx = (xr - xl)/double(Nx);
	double hy = (yr - yl)/double(Ny);

	colvec param = zeros<colvec>(2);
	//Nx + 1 evaluations
	param[0] = xl;
	param[1] = yl;

	//Store the partial results of int f(x,y) dx
	double result = 0.0;
	double results[Ny+1];
	double temp;

	param[1] = yl;
	for (int j = 0; j <= Ny; j ++) {
		param[0] = xl;
		temp = std::exp(evaluate(param));
		posterior(0,j) = temp;
		result = temp/2.0;
		for (int i = 1; i < Nx; i ++) {
			param[0] += hx;
			temp = std::exp(evaluate(param));
			posterior(i,j) = temp;
			result += temp;
		}
		param[0] = xr;
		temp = std::exp(evaluate(param));
		posterior(Nx, j) = temp;
		result += temp/2.0;
		results[j] = result * hx;
		param[1] += hy;
	}

	result = 0.0;
	for (int j = 0; j <= Ny; j ++) {
		result += results[j];
	}
	result = result - results[0]/2.0 - results[Ny]/2.0;
	result *= hy;
	double logev = log(result);
	std::cout << "log - Evidence is " << logev << std::endl;

	for (int i = 0; i <=Nx; i ++) {
		for (int j = 0; j <=Ny; j ++) {
			posterior(i,j) = std::exp(log(posterior(i,j)) - logev);
		}
	}

	return posterior;
}

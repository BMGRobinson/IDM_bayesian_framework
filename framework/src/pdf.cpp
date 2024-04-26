#include "pdf.hpp"
using namespace arma;

/******************************************************
*  PDF (dimension = 1)
******************************************************/


pdf1d::pdf1d( double _mean, double _cov) {
	mean = _mean;
	covariance = _cov;
	stdDeviation = sqrt(covariance);

}

void pdf1d::setMean( double _mean ) {
	mean = _mean;
}

void pdf1d::setCovariance( double _cov) {
	covariance = _cov;
	stdDeviation = sqrt(covariance);
}

/******************************************************
*  Gaussian (dimension = 1)
******************************************************/

void Gaussian1d::setCovariance( double _cov ) {
	covariance = _cov;
	stdDeviation = std::sqrt(covariance);
	normConstant = 1.0 / std::sqrt( 2.0 * PI * _cov );
}


double Gaussian1d::sample() {
	//Generates two Gaussian random number, however in this case we only use one
	return this->mean + stdDeviation * randn();
}


Gaussian1d::Gaussian1d(double _mean, double _cov) :  pdf1d::pdf1d(_mean,_cov) {
	normConstant = 1.0 / sqrt( 2.0 * PI * _cov );
}

double Gaussian1d::getMean() {
	return this->mean;
}

double Gaussian1d::getCovariance() {
	return this->covariance;
}

double Gaussian1d::getMode() {
	return this->mean;
}

long double Gaussian1d::getLogDensity(double x) {
	double diff = x - mean;
	return log(normConstant) + -0.5*diff*diff/covariance;
}

/******************************************************
*  Log Normal (dimension = 1)
******************************************************/

double LogNormal1d::sample() {
	double z = randn();
	//Sample from a gaussian distribution
	return exp(location + scale * z);
}

//No check on input parameters here
/*
LogNormal1d::LogNormal1d(double _location, double _scale) :  pdf1d::pdf1d(std::exp(_location + _scale*_scale*0.5),(std::exp(_scale*_scale)-1.0)*std::exp(2.0*_location+_scale*_scale)) {

	location = _location;
	scale = _scale;
	normConstant = 1.0/(scale*std::sqrt(2.0*datum::pi));
}
*/
//dummy values for pdf1d
LogNormal1d::LogNormal1d(double median, double COV) : pdf1d(0.0, 1.0) {

	location = log(median);
	scale = std::sqrt(log(COV*COV+1.0));

	normConstant = 1.0/(scale*std::sqrt(2.0*datum::pi));
	mean = std::exp(location + scale*scale*0.5);
	//covariance = (std::exp(scale*scale)-1.0)*std::exp(2.0*location+scale*scale);

	stdDeviation = COV * mean;
	covariance = stdDeviation * stdDeviation;
	//stdDeviation = sqrt(covariance);
}

double LogNormal1d::getMean() {
	return this->mean;
}

double LogNormal1d::getCovariance() {
	return this->covariance;
}

double LogNormal1d::getMode() {
	return std::exp(location-scale*scale);
}

long double LogNormal1d::getLogDensity(double x) {
	//Limiting case
	if (x <= 0.0) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	double diff = log(x) - location;
	return log(normConstant/x) + -0.5*diff*diff/(scale*scale);
}

/******************************************************
*  Uniform (dimension = 1)
******************************************************/


//double Uniform1d::sample() {
	//Generates two Gaussian random number, however in this case we only use one
	//return this->mean + stdDeviation * randn();
//}


Uniform1d::Uniform1d(double _left, double _right) :  pdf1d::pdf1d( (_left+_right)/2.0 , ( _left - _right )*( _left - _right )/12.0 ) {
	left = _left;
	right = _right;
	normConstant = 1.0 / ( _right - _left );
}

double Uniform1d::getMean() {
	return this->mean;
}

double Uniform1d::getCovariance() {
	return this->covariance;
}

double Uniform1d::getMode() {
	return this->mean;
}

double Uniform1d::sample() {
	return left + (right - left) * randu();
}

long double Uniform1d::getLogDensity(double x) {
	if ( x <= right && x >= left ) {
		return log(normConstant);
	} else {
		return std::numeric_limits<double>::quiet_NaN();
	}
}

/******************************************************
*  Reciprocal (Improper prior)
*******************************************************/

reciprocal::reciprocal() : pdf1d(-1.0,-1.0) {
	//nothing to do here
}

double reciprocal::getCovariance() {
	throw std::runtime_error("Covariance undefined for reciprocal");
	return -1.0;
}

double reciprocal::getMean() {
	throw std::runtime_error("Mean undefined for reciprocal");
	return -1.0;
}

double reciprocal::getMode() {
	throw std::runtime_error("Mode undefined for reciprocal");
	return -1.0;
}

long double reciprocal::getLogDensity(const double x) {
	return -log(x);
}

double reciprocal::sample() {
	throw std::runtime_error("Sampling not implemented for reciprocal");
	return -1.0;
}

/******************************************************
*  PDF (dimension > 1)
******************************************************/

pdf::pdf() {
	//do nothing;
}
pdf::pdf( colvec _mean, mat _cov) {
	mean = _mean;
	covariance = _cov;

}

void pdf::setMean( colvec _mean ) {
	mean = _mean;
}

void pdf::setCovariance( mat _cov) {
	covariance = _cov;
}

int pdf::getDim() {
	return dim;
}

/******************************************************
*  Gaussian (dimension > 1)
******************************************************/


void Gaussian::setCovariance( mat _cov ) {
	covariance = _cov;
	//normConstant = 1.0 / sqrt( det(2.0 * PI * _cov) );
	//invCov = inv(_cov);
}
Gaussian::Gaussian() : pdf::pdf() {
	//default constructor, do nothing
}
Gaussian::Gaussian(const Gaussian & other) : pdf(other) {
	covariance = other.covariance;
	normConstant = other.normConstant;
	mean = other.mean;
	invCov = other.invCov;
	covChol = other.covChol;
	randomVec = colvec(mean.n_rows);
}

Gaussian::Gaussian(colvec _mean, mat _cov) :  pdf::pdf(_mean,_cov) {
	normConstant = 1.0 / sqrt( det(2.0 * PI * _cov) );
	invCov = inv(_cov);
	covChol = chol(_cov, "lower");
	randomVec = colvec(mean.n_rows);
}

colvec Gaussian::getMean() {
	return this->mean;
}

mat Gaussian::getCovariance() {
	return this->covariance;
}

colvec Gaussian::getMode() {
	return this->mean;
}

double Gaussian::evaluate(const colvec& x) {
	colvec diff = x - mean;
	return normConstant * exp( -0.5* as_scalar(trans(diff)*invCov*diff) );
}

long double Gaussian::getLogDensity(const colvec& x) {
	colvec diff = x - mean;
	return log(normConstant) + -0.5*as_scalar(trans(diff)*invCov*diff);
}


colvec Gaussian::sample() {
	//Generates two Gaussian random number, however in this case we only use one
	return mean + covChol * randomVec.randn();
}

long double Gaussian::getLogDensityKernel(const colvec& x) {
	colvec diff = x - mean;
	return -0.5*as_scalar(trans(diff)*invCov*diff);
}

/*
//Basic systematic resampling algorithm O(num of particles)
//Adapted from Arnaud Doucet and Nando de Freitas
void systematic_resampling2() {
	//round towards zero
	fesetround(FE_TOWARDZERO);
//	int * indices_in = new int(size);
//	int * indices_out = new int(size):
	int * label = new int(size);
	int * N_children = new int(size);
	//colvec indices_in = zeros<colvec>(size);
	//colvec indices_out = zeros<col
	//colvec N_children = zeros<colvec>(size);
	//colvec label = zeros<colvec>(size);
	int k,i,j;
	double s=1.0/double(size);
	double auxw=0.0;
	double auxl=0.0;
	int li=0;   // Label of the current point
	for (k = 0; k < size; k ++) {
		label[k] = k;
//		indices_in[k] = k;
	}
	double T = s * 	randu();
	j=0; //1.0 in original code
	double Q=0.0;
	//double i=0.0;
	std::cout << "*************" << std::endl;
	colvec u = randu<colvec>(size);
	while (T<1.0) {
   		if (Q>T) {
      		T+=s;
      		N_children[li]=N_children[li]+1;
      	} else {
      		//select i uniformly between j and N
      		//i=fix((N-j+1)*u(1,j))+j;
      		i = rint(  u[j] * double(size-j) + double(j));
      		//save the associate characteristic
      		auxw=weights[i];
      		li=label[i];
      		//update the cfd
      		Q+=auxw;
      		//swap
      		//wn(1,i)=wn(1,j);
      		weights[i] = weights[j];
      		label[i]=label[j];
      		//%wn(1,j)=auxw;
      		//%label(1,j)=li;
      		j++;
      	}
  	}
	std::cout << "*************" << std::endl;
	//Swap the particles
	int index=0;
	for (i=1; i < size ; i ++) {
		if (N_children[i] > 0) {
			for (j=index; j < index+N_children[i]-1; j++) {
      			//indices_out[j] = indicies_in[i];
      			samples.unsafe_col(j) = samples.unsafe_col(i);
      		}
		}
	index += N_children[i];
	}
	std::cout << "*************" << std::endl;

}
*/

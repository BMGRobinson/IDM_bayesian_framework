#ifndef PDF_HPP_
#define PDF_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <mpi.h>
#pragma GCC diagnostic pop

//const long double PI = 3.14159265359;
const long double PI=std::acos(-1.0L);

using namespace arma;

class pdf1d {
	protected:
		double mean;
		double covariance;
		double stdDeviation;
	public:
		pdf1d( double _mean, double _cov);
		virtual double getMean() = 0;
		virtual double getCovariance() = 0;
		virtual double getMode() = 0;
		virtual double sample() = 0;
		void setMean( double _mean );
		void setCovariance(double _covariance );
		virtual long double getLogDensity(const double x) = 0;
};


class Gaussian1d: public pdf1d {
public:
	Gaussian1d( double _mean, double _cov);
	double getMean();
	double sample();
	double getCovariance();
	double getMode();
	void setCovariance(double _cov);
	long double getLogDensity(const double x);

private:
	double normConstant;
};

class Uniform1d: public pdf1d {
public:
	Uniform1d(double _left, double _right);
	double getMean();
	double sample();
	double getCovariance();
	double getMode();
	long double getLogDensity(const double x);

private:
	double normConstant;
	double left;
	double right;
};
//implement log normal 1d pdf
class LogNormal1d: public pdf1d {
public:
	//LogNormal1d( double _loc, double _scale);
	LogNormal1d( double median, double _covariance);
	double getMean();
	double sample();
	double getCovariance();
	double getMode();
	long double getLogDensity(const double x);

private:
	double location;
	double scale;
	double normConstant;
};

class reciprocal: public pdf1d {
public:
	reciprocal();
	double getMean();
	double sample();
	double getCovariance();
	double getMode();
	long double getLogDensity(const double x);
};

//Uniform [0, infty]
/*
class Positive : public pdf1d {
	Positive();
	double getLogDensity(const double x);
	double getCovariance();
	double getMode();
	double getMode();
}
*/

class pdf {
protected:
	colvec mean;
	mat covariance;
	int dim;
public:
	pdf();
	pdf( colvec _mean, mat _cov);
	int getDim();
	virtual colvec getMean() = 0;
	virtual mat getCovariance() = 0;
	virtual colvec getMode() = 0;
	void setMean( colvec _mean );
	void setCovariance(mat _covariance );
};

class Gaussian: public pdf {
public:
	//Gaussian( double _mean, double _cov);
	Gaussian();
	Gaussian(const Gaussian & other);
	Gaussian( colvec _mean, mat _cov);
	colvec getMean();
	mat getCovariance();
	colvec getMode();
	void setCovariance(mat _cov);
	colvec sample();
	//double evaluate(const double x);
	double evaluate(const colvec& x);
	long double getLogDensity(const colvec& x);
	long double getLogDensityKernel(const colvec& x);
private:
	long double normConstant;
	mat invCov;
	mat covChol;
	colvec randomVec;
};



#endif

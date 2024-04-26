#ifndef STATESPACE_HPP_
#define STATESPACE_HPP_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <functional>
#pragma GCC diagnostic pop
using namespace arma;


//TODO
//need to implement adding loglikelihood function and fix it in filters
//measurement covariance is added in the statespace
//two possible likelihood funcs one that gives the measurement covariance, the other that doesn't
//next step is to clean all the constructors using initializer list
//make optional parameters


//Model functions
typedef std::function<colvec(const colvec&, const double, double, const colvec&)> modelOp;
typedef std::function<mat(const colvec&, const double, const colvec&)> modelJacobian;

//Measurement functions
typedef std::function<colvec(const colvec&, const mat&, const colvec&)> measOp;
typedef std::function<mat(const colvec&, const colvec&)> measJacobian;

//Loglikelihood function ( state, measurement, parameters, covariance matrix)
typedef std::function<double(const colvec&, const colvec&, const colvec&, const mat& )> loglikfunc;

class statespace {
private:
	modelOp f;
	modelJacobian dfdx;
	modelJacobian dfde;
	measOp h;
	measJacobian dhdx;
	measJacobian dhde;
	loglikfunc loglikelihoodfunc;
	mat measCov; //optional
	int m; //size of measurement vector
	int n; //size of state vector
	int ForecastStepsBetweenMeasurements;
	double dt;
	long double time;
public:
	statespace();
	statespace(modelOp _f, modelJacobian _dfdx, modelJacobian _dfde, measOp _h, measJacobian _dhdx, measJacobian _dhde, double _dt, int _n, int _m);
	statespace(modelOp _f, modelJacobian _dfdx, modelJacobian _dfde, measOp _h, measJacobian _dhdx, measJacobian _dhde, int _n, int _m);
	statespace(modelOp _f,  measOp _h, loglikfunc _lf, double _dt, int _n, int _m);
	statespace(modelOp _f,  measOp _h, loglikfunc _lf, int _n, int _m);
	statespace(modelOp _f,  measOp _h, double _dt, int _n, int _m);
	statespace(modelOp _f,  measOp _h, int _n, int _m);

	//statespace(modelOp _f,  measOp _h, measJacobian _measCov);
	double getDt();
	double getTime();
	int getMeasurementVectorSize();
	int getStateVectorSize();
	int getForecastStepsBetweenMeasurements();
	void setForecastStepsBetweenMeasurements(int _fStepsBetweenMeasurments);
	void setDt(double _dt);
	void setMeasCov( mat& _measCov  );

/*
* Wrapper functions for the statespace model.
*/
	colvec evaluatef( const colvec& x, double time, const colvec& parameters);
	colvec evaluatef( const colvec& x, const colvec& parameters);
	mat evaluatedfdx( const colvec& x, const colvec& parameters);
	mat evaluatedfde( const colvec& x, const colvec& parameters);
	colvec evaluateh( const colvec& x, const colvec& parameters);
	mat evaluatedhdx( const colvec& x, const colvec& parameters);
	mat evaluatedhde( const colvec& x, const colvec& parameters);
	//mat evaluateMeasurementCov( const colvec& x, const colvec& parameters );
	double evaluateloglikelihood( const colvec& d, const colvec& x, const colvec& parameters );

	/*
* Wrapper functions for the statespace model. Fixed parameters
*/
	colvec evaluatef( const colvec& x);
	colvec evaluatef( const colvec& x, double time);
	mat evaluatedfdx( const colvec& x);
	mat evaluatedfde( const colvec& x);
	colvec evaluateh( const colvec& x);
	mat evaluatedhdx( const colvec& x);
	mat evaluatedhde( const colvec& x);

	void resetTime();
	void timeIncrement();
};

#endif

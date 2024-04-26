#include "statespace.hpp"
  statespace::statespace() {
	f = 0;
	h = 0;
  loglikelihoodfunc = 0;
	dfdx = 0;
	dfde = 0;
	dhdx = 0;
	dhde = 0;
	dt = 0.0;
	time = 0.0;
  n = 0;
  m = 0;
}

statespace::statespace(modelOp _f, modelJacobian _dfdx, modelJacobian _dfde, measOp _h, measJacobian _dhdx, measJacobian _dhde, double _dt, int _n, int _m) {
	f = _f;
	h = _h;
	dfdx = _dfdx;
	dfde = _dfde;
	dhdx = _dhdx;
	dhde = _dhde;
	dt = _dt;
  loglikelihoodfunc = 0;
	time = 0.0;
	n = _n;
	m = _m;
}

statespace::statespace(modelOp _f, modelJacobian _dfdx, modelJacobian _dfde, measOp _h, measJacobian _dhdx, measJacobian _dhde, int _n, int _m) {
	f = _f;
	h = _h;
	dfdx = _dfdx;
	dfde = _dfde;
	dhdx = _dhdx;
	dhde = _dhde;
  loglikelihoodfunc = 0;
	dt = 0.0;
	time = 0.0;
	n = _n;
	m = _m;
}

statespace::statespace(modelOp _f,  measOp _h, double _dt, int _n, int _m) {
	f = _f;
	h = _h;
	dfdx = 0;
	dfde = 0;
	dhdx = 0;
	dhde = 0;
  loglikelihoodfunc = 0;
	n = _n;
  m = _m;
	dt = _dt;
	time = 0.0;
}

statespace::statespace(modelOp _f,  measOp _h, int _n, int _m) {

	f = _f;
	h = _h;
	dfdx = 0;
	dfde = 0;
	dhdx = 0;
	dhde = 0;
	dt = 0.0;
  loglikelihoodfunc = 0;
  n = _n;
  m = _m;
	time = 0.0;
}

statespace::statespace(modelOp _f,  measOp _h, loglikfunc _lf, double _dt, int _n, int _m) {
	f = _f;
	h = _h;
	dfdx = 0;
	dfde = 0;
	dhdx = 0;
	dhde = 0;
  loglikelihoodfunc = _lf;
  n = _n;
  m = _m;
	dt = _dt;
	time = 0.0;
}

statespace::statespace(modelOp _f,  measOp _h, loglikfunc _lf, int _n, int _m) {
	f = _f;
	h = _h;
	dfdx = 0;
	dfde = 0;
	dhdx = 0;
	dhde = 0;
	dt = 0.0;
  loglikelihoodfunc = _lf;
  n = _n;
  m = _m;
	//measCov = 0;
	time = 0.0;
}

double statespace::getDt() {
	return dt;
}

double statespace::getTime() {
  return time;
}

void statespace::setDt(double _dt) {
	dt = _dt;
}

void statespace::setMeasCov( mat& _measCov  ) {
	measCov = _measCov;
}

int statespace::getMeasurementVectorSize() {
  return m;
}

int statespace::getStateVectorSize() {
  return n;
}

int statespace::getForecastStepsBetweenMeasurements() {
	return ForecastStepsBetweenMeasurements;
}

void statespace::setForecastStepsBetweenMeasurements(int _fStepsBetweenMeasurments) {
	ForecastStepsBetweenMeasurements = _fStepsBetweenMeasurments;
}

/*

mat statespace::evaluateMeasurementCov( const colvec& x, const colvec& parameters ) {
	return measCov(x, parameters);
}
*/
colvec statespace::evaluatef( const colvec& x, double _time, const colvec& parameters) {
	return f(x, dt, _time, parameters);
}

mat statespace::evaluatedfdx( const colvec& x, const colvec& parameters) {
	return dfdx(x,dt, parameters);
}

mat statespace::evaluatedfde( const colvec& x, const colvec& parameters) {
	return dfde(x,dt, parameters);
}

colvec statespace::evaluateh( const colvec& x, const colvec& parameters) {
	return h(x, measCov, parameters);
}

mat statespace::evaluatedhdx( const colvec& x, const colvec& parameters) {
	return dhdx(x, parameters);
}
mat statespace::evaluatedhde( const colvec& x, const colvec& parameters) {
	return dhde(x, parameters);
}

colvec statespace::evaluatef( const colvec& x) {
	//pass empty parameters and time
	return f(x, dt, time, colvec() );
}

colvec statespace::evaluatef( const colvec& x, double _time) {
	return f(x, dt, _time, colvec () );
}

colvec statespace::evaluatef( const colvec& x, const colvec& parameters) {
	return f(x, dt, time, parameters);
}

double statespace::evaluateloglikelihood( const colvec& d, const colvec& x, const colvec& parameters ) {
  return loglikelihoodfunc(d, x, parameters , measCov);
}

mat statespace::evaluatedfdx( const colvec& x) {
	return dfdx(x,dt, colvec () );
}
mat statespace::evaluatedfde( const colvec& x) {
	return dfde(x,dt, colvec () );
}

colvec statespace::evaluateh( const colvec& x) {
	return h(x, measCov, colvec () );
}

mat statespace::evaluatedhdx( const colvec& x) {
	return dhdx(x, colvec () );
}
mat statespace::evaluatedhde( const colvec& x) {
	return dhde(x, colvec () );
}

void statespace::resetTime() {
	time = 0.0;
}

void statespace::timeIncrement() {
	time += dt;
}

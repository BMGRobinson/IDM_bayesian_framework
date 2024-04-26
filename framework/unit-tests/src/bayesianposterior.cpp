#include <gtest/gtest.h>
#include "bayesianPosterior.hpp"
#include "statespace.hpp"
#include <armadillo>
#include <cmath>
using namespace arma;

//This is more like an integration test
//At one point will need to mock objects
/*
mat::fixed<2,2> _dfdx;
colvec::fixed<1> _h;
colvec::fixed<2> statekp1;
mat::fixed<2,2> _dfde;
mat::fixed<1,2> _dhdx;
mat::fixed<1,1> _dhde;
const double c = 3.0;
const double k = 50.0;

colvec f(const colvec& state, double dt = 0.0, double time = 0.0, const colvec& param = 0) {
	statekp1[0] = state[0] + dt*state[1];
	statekp1[1] = -k*dt*state[0] + state[1]*(1.0 - c*dt);
	return statekp1;
}



colvec h(const colvec& state, const mat& cov = 0, const colvec& param = 0) {
	_h[0] = state[0];
	return _h;
}

mat dfdx(const colvec& state, double dt = 0.0, const colvec& param = 0) {
	_dfdx(0,0) = 1.0;
	_dfdx(0,1) = dt;
	_dfdx(1,0) = -k*dt;
	_dfdx(1,1) = 1.0-c*dt;
	return _dfdx;
}
mat dfde(const colvec& state, double dt = 0.0, const colvec& param = 0) {
	_dfde(0,0) = 0.0;
	_dfde(0,1) = 0.0;
	_dfde(1,0) = 0.0;
	_dfde(1,1) = 1.0;
	return _dfde;
}
mat dhdx(const colvec& state, const colvec& param = 0) {
	_dhdx(0,0) = 1.0;
	_dhdx(0,1) = 0.0;
	return _dhdx;
}
mat dhde(const colvec& state, const colvec& param = 0) {
	_dhde(0,0) = 1.0;
	return _dhde;
}


TEST(bayesianPosterior, bpEKFevaluate) {
	colvec temp;
	colvec dummyParameters;
	mat::fixed<1,1> obs = {1.0};

	// EKF filter
	double dt = 0.1;
	colvec::fixed<2> u = {2.0,3.0};
	mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
	Gaussian * state = new Gaussian(u,P);
	mat::fixed<2,2> Q = {{0.0,0.0},{0.0,dt}};
	mat::fixed<1,1> R = {0.5};

	// Statespace and filer
	statespace ss = statespace(f,dfdx,dfde,h,dhdx,dhde,dt,2,1);
	Ekf myekf = Ekf(state,ss,Q,R);

	//Creating bayesian Posterior object
	bayesianPosterior bp = bayesianPosterior(obs, &myekf);

	double actual = bp.evaluate( dummyParameters );

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );

	ASSERT_DOUBLE_EQ(actual, logpy);
}
*/

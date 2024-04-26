#include <gtest/gtest.h>
//#include "gtesthelper.hpp"
#include "filters.hpp"
#include "statespace.hpp"
#include <armadillo>
#include <cmath>
using namespace arma;

::testing::AssertionResult compareMatrix(const mat& A, const mat& B);

/* Definitions of function for all filters */
colvec::fixed<1> _h;
colvec::fixed<2> statekp1;
mat::fixed<2,2> _dfdx;
mat::fixed<2,2> _dfde;
mat::fixed<1,2> _dhdx;
mat::fixed<1,1> _dhde;

/* 1D */
mat::fixed<1,1> _dfdx1d;
colvec::fixed<1> _h1d;
colvec::fixed<1> xkp1;
mat::fixed<1,1> _dfde1d;
mat::fixed<1,1> _dhdx1d;
mat::fixed<1,1> _dhde1d;

const double c = 3.0;
const double k = 50.0;

colvec f(const colvec& state, double dt = 0.0, double time = 0.0, const colvec& param = 0) {
	statekp1[0] = state[0] + dt * state[1];
	statekp1[1] = -k * dt * state[0] + state[1] * (1.0 - c * dt);
	return statekp1;
}

colvec fs(const colvec& state, double dt = 0.0, double time = 0.0, const colvec& param = 0) {
	statekp1[0] = state[0] + dt * state[1];
	statekp1[1] = -k * dt * state[0] + state[1] * (1.0 - c * dt) + std::sqrt(dt) * randn();
	return statekp1;
}

colvec h(const colvec& state, const mat& cov = 0, const colvec& param = 0) {
	_h[0] = state[0];
	return _h;
}

colvec hs(const colvec& state, const mat& cov, const colvec& param = 0) {
	_h[0] = state[0] + std::sqrt(cov.at(0,0))*randn();
	return _h;
}

double loglik(const colvec& d, const colvec& x, const colvec& param = 0, const mat& measCov  = 0) {
	return Gaussian1d(d[0], measCov.at(0,0)).getLogDensity(x[0]);
}

colvec f1d(const colvec& state, double dt = 0.0, double time = 0.0, const colvec& param = 0) {
	xkp1[0] = state[0];
	return xkp1;
}

colvec h1d(const colvec& state, const mat& cov = 0, const colvec& param = 0) {
	return {state[0]};
}

mat dfdx1d(const colvec& state, double dt = 0.0, const colvec& param = 0) {
	_dfdx1d(0,0) = 1.0;
	return _dfdx1d;
}
mat dfde1d(const colvec& state, double dt = 0.0, const colvec& param = 0) {
	_dfde1d(0,0) = std::sqrt(dt);
	return _dfde1d;
}
mat dhdx1d(const colvec& state, const colvec& param = 0) {
	_dhdx1d(0,0) = 1.0;
	return _dhdx1d;
}
mat dhde1d(const colvec& state, const colvec& param = 0) {
	_dhde1d(0,0) = 1.0;
	return _dhde1d;
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

class ENKFMPITest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		int N = 2000000;//20000000;
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
		mat::fixed<1,1> R = {0.5};
		mat::fixed<1,2> H = {1.0,0.0};
		ss = statespace(fs,hs,loglik,dt,2,1);
		ss.setMeasCov(R);
		myenkf = EnkfMPI(u,P,N,ss, MPI::COMM_WORLD );
		MPI::COMM_WORLD.Barrier();
		abs_error = 1e-1; //Low absolute error due to sampling error.
	}

	EnkfMPI myenkf;
	statespace ss;
	double abs_error;
};

TEST_F(ENKFMPITest, resetFilter) {
	colvec temp;
	myenkf.forecast(temp);
	myenkf.reset();
	colvec actual = myenkf.getState().getMean();
	//myenkf.saveSamples("resetFilter.dat");

	ASSERT_NEAR(actual.at(0), 2.0, abs_error);
	ASSERT_NEAR(actual.at(1), 3.0, abs_error);
}


TEST_F(ENKFMPITest, ForecastMean) {
	colvec temp;
	myenkf.forecast(temp);
	colvec actual = myenkf.getState().getMean();

	ASSERT_NEAR(actual.at(0), 2.3, abs_error);
	ASSERT_NEAR(actual.at(1), -7.9, abs_error);
}

TEST_F(ENKFMPITest, ForecastCov) {
	colvec temp;
	myenkf.forecast(temp);
	mat actual = myenkf.getState().getCovariance();
	mat expected = {{2.22,-9.66},{-9.66,44.08}};
	//double err = std::max(std::max(std::abs(expected(0,0)-actual(0,0))/expected(0,0), std::abs(expected(0,1)-actual(0,1))/expected(0,1)),std::max(std::abs(expected(1,0)-actual(1,0))/expected(1,0),std::abs(expected(1,1)-actual(1,1))/expected(1,1) ));
	//double tol = 1.0e-3;
	//ASSERT_LT(err, tol);
	ASSERT_NEAR(actual.at(0,0), 2.22, abs_error);
	ASSERT_NEAR(actual.at(0,1), -9.66, abs_error);
	ASSERT_NEAR(actual.at(1,0), -9.66, abs_error);
	ASSERT_NEAR(actual.at(1,1), 44.08, abs_error);
}

TEST_F(ENKFMPITest, UpdateMean) {
	colvec temp;
	colvec obs = {1.0};
	myenkf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec expected = {1.2,2.6};
	colvec actual = myenkf.getState().getMean();
	//double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	//double tol = 1.0e-3;
	//ASSERT_LT(err, tol);
	ASSERT_NEAR(actual.at(0), 1.2, abs_error);
	ASSERT_NEAR(actual.at(1), 2.6, abs_error);
}

TEST_F(ENKFMPITest, UpdateCov) {
	colvec temp;
	colvec obs = {1.0};
	myenkf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat actual = myenkf.getState().getCovariance();
	ASSERT_NEAR(actual.at(0,0), 0.4, 1.0e-2);
	ASSERT_NEAR(actual.at(0,1), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,0), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,1), 1.6, 1.0e-2);
}

TEST_F(ENKFMPITest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = myenkf.logLikelihoodOfMeasurement(obs);
	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );
	ASSERT_NEAR(actual, logpy, 1.0e-2);
}

/*****************************************************************
******************************************************************
* PF-ENKF
******************************************************************
*****************************************************************/

class PFMPITest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		int N = 4000000;//20000000;
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
		mat::fixed<1,1> R = {0.5};
		mat::fixed<1,2> H = {1.0,0.0};
		ss = statespace(fs,hs,loglik,dt,2,1);
		ss.setMeasCov(R);
		mypf = PFMPI(u,P,N,ss, MPI::COMM_WORLD );
		MPI::COMM_WORLD.Barrier();
		abs_error = 1e-1; //Low absolute error due to sampling error.
	}

	PFMPI mypf;
	statespace ss;
	double abs_error;
};

TEST_F(PFMPITest, resetFilter) {
	colvec temp;
	mypf.forecast(temp);
	mypf.reset();
	colvec actual = mypf.getState().getMean();
	ASSERT_NEAR(actual.at(0), 2.0, abs_error);
	ASSERT_NEAR(actual.at(1), 3.0, abs_error);
}


TEST_F(PFMPITest, ForecastMean) {
	colvec temp;
	mypf.forecast(temp);
	colvec actual = mypf.getState().getMean();

	ASSERT_NEAR(actual.at(0), 2.3, abs_error);
	ASSERT_NEAR(actual.at(1), -7.9, abs_error);
}

TEST_F(PFMPITest, ForecastCov) {
	colvec temp;
	mypf.forecast(temp);
	mat actual = mypf.getState().getCovariance();

	mat expected = {{2.22,-9.66},{-9.66,44.08}};
	ASSERT_NEAR(actual.at(0,0), 2.22, abs_error);
	ASSERT_NEAR(actual.at(0,1), -9.66, abs_error);
	ASSERT_NEAR(actual.at(1,0), -9.66, abs_error);
	ASSERT_NEAR(actual.at(1,1), 44.08, abs_error);
}

TEST_F(PFMPITest, UpdateMean) {
	colvec temp;
	colvec obs = {1.0};
	mypf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec expected = {1.2,2.6};
	colvec actual = mypf.getState().getMean();
	ASSERT_NEAR(actual.at(0), 1.2, abs_error);
	ASSERT_NEAR(actual.at(1), 2.6, abs_error);
}

TEST_F(PFMPITest, UpdateCov) {
	colvec temp;
	colvec obs = {1.0};
	mypf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat actual = mypf.getState().getCovariance();
//	int S = 10;
//	for (int s = 1; s < S; s++) {
//		mypf.reset();
//		mypf.update(obs, temp);
//		actual += mypf.getState().getCovariance();
//	}
//	actual = actual / double(S); //Take the average to reduce the error
	ASSERT_NEAR(actual.at(0,0), 0.4, 1.0e-2);
	ASSERT_NEAR(actual.at(0,1), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,0), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,1), 1.6, 1.0e-2);
}

TEST_F(PFMPITest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = mypf.logLikelihoodOfMeasurement(obs);
	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );
	ASSERT_NEAR(actual, logpy, 1.0e-2);
}


/*****************************************************************
******************************************************************
* PF-ENKF
******************************************************************
*****************************************************************/

class PFENKFMPITest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		int N = 10000;//20000000;
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
		mat::fixed<1,1> R = {0.5};
		mat::fixed<1,2> H = {1.0,0.0};
		ss = statespace(fs,hs,loglik,dt,2,1);
		ss.setMeasCov(R);
		mypfenkf = PFEnkfMPI(u,P,N,ss, MPI::COMM_WORLD );
		MPI::COMM_WORLD.Barrier();
		abs_error = 1e-1; //Low absolute error due to sampling error.
	}

	PFEnkfMPI mypfenkf;
	statespace ss;
	double abs_error;
};

TEST_F(PFENKFMPITest, resetFilter) {
	colvec temp;
	mypfenkf.forecast(temp);
	mypfenkf.reset();
	colvec actual = mypfenkf.getState().getMean();
	ASSERT_NEAR(actual.at(0), 2.0, abs_error);
	ASSERT_NEAR(actual.at(1), 3.0, abs_error);
}


TEST_F(PFENKFMPITest, ForecastMean) {
	colvec temp;
	mypfenkf.forecast(temp);
	colvec actual = mypfenkf.getState().getMean();

	ASSERT_NEAR(actual.at(0), 2.3, abs_error);
	ASSERT_NEAR(actual.at(1), -7.9, abs_error);
}

TEST_F(PFENKFMPITest, ForecastCov) {
	colvec temp;
	mypfenkf.forecast(temp);
	mat actual = mypfenkf.getState().getCovariance();
	mat expected = {{2.22,-9.66},{-9.66,44.08}};
	ASSERT_NEAR(actual.at(0,0), 2.22, abs_error);
	ASSERT_NEAR(actual.at(0,1), -9.66, abs_error);
	ASSERT_NEAR(actual.at(1,0), -9.66, abs_error);
	ASSERT_NEAR(actual.at(1,1), 44.08, abs_error);
}

TEST_F(PFENKFMPITest, UpdateMean) {
	colvec temp;
	colvec obs = {1.0};
	mypfenkf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec expected = {1.2,2.6};
	colvec actual = mypfenkf.getState().getMean();
	ASSERT_NEAR(actual.at(0), 1.2, abs_error);
	ASSERT_NEAR(actual.at(1), 2.6, abs_error);
}

TEST_F(PFENKFMPITest, UpdateCov) {
	colvec temp;
	colvec obs = {1.0};
	mypfenkf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat actual = mypfenkf.getState().getCovariance();
	ASSERT_NEAR(actual.at(0,0), 0.4, 1.0e-2);
	ASSERT_NEAR(actual.at(0,1), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,0), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,1), 1.6, 1.0e-2);
}

TEST_F(PFENKFMPITest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = mypfenkf.logLikelihoodOfMeasurement(obs);
	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );
	ASSERT_NEAR(actual, logpy, 1.0e-2);
}

/*****************************************************************
******************************************************************
* filters
******************************************************************
*****************************************************************/


class EKFTest1d : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		double dt = 0.5;
		colvec::fixed<1> u = {0.0};
		mat::fixed<1,1> P = {1.0};
		Gaussian * state = new Gaussian(u,P);
		mat::fixed<1,1> Q = {1.0};
		mat::fixed<1,1> R = {1.0};
		ss = statespace(f1d,dfdx1d,dfde1d,h1d,dhdx1d,dhde1d,dt,1,1);
		ss.setForecastStepsBetweenMeasurements(1);
		myekf = Ekf(state,ss,Q,R);
	}
	Ekf myekf;
	statespace ss;
};


TEST_F(EKFTest1d, ForecastMean) {
	colvec temp;
	myekf.forecast(temp);
	colvec mean = myekf.getState()->getMean();
	ASSERT_DOUBLE_EQ(mean[0], 0.0);
}

TEST_F(EKFTest1d, ForecastCov) {
	colvec temp;
	myekf.forecast(temp);
	mat cov = myekf.getState()->getCovariance();
	ASSERT_DOUBLE_EQ(cov.at(0,0), 1.5);
}

TEST_F(EKFTest1d, UpdateMean) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec mean = myekf.getState()->getMean();
	ASSERT_DOUBLE_EQ(mean[0], 0.3);
}

TEST_F(EKFTest1d, UpdateCov) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat cov = myekf.getState()->getCovariance();
	ASSERT_DOUBLE_EQ(cov.at(0,0), 0.6);
}


TEST_F(EKFTest1d, ForecastMean2) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	myekf.forecast(temp);
	colvec mean = myekf.getState()->getMean();
	ASSERT_DOUBLE_EQ(mean[0], 0.3);
}

TEST_F(EKFTest1d, ForecastCov2) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	myekf.forecast(temp);
	mat cov = myekf.getState()->getCovariance();
	ASSERT_DOUBLE_EQ(cov.at(0,0), 1.1);
}

TEST_F(EKFTest1d, UpdateMean2) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	myekf.forecast(temp);
	colvec obs2 = {1.0};
	myekf.update(obs2,temp); //to evaluate kalman gain, these are not important for this case
	colvec mean = myekf.getState()->getMean();
	ASSERT_DOUBLE_EQ(mean[0], 2./3.);
}

TEST_F(EKFTest1d, UpdateCov2) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	myekf.forecast(temp);
	colvec obs2 = {1.0};
	myekf.update(obs2,temp); //to evaluate kalman gain, these are not important for this case
	mat cov = myekf.getState()->getCovariance();
	double realcov = 1.0*1.1 / (1.0 + 1.1);
	ASSERT_DOUBLE_EQ(cov.at(0,0), realcov );
}


TEST_F(EKFTest1d, LogLikelihood) {
	colvec temp;
	myekf.forecast(temp); //First forecast to 0.5 second
	colvec obs = {0.5};
	double actual = std::exp(myekf.logLikelihoodOfMeasurement(obs));
	double trueVal = 1.0/std::sqrt(2.0 * datum::pi * 2.5) * std::exp(-0.5 * 0.5 * 0.5 / 2.5); //0.24000778968602721
	ASSERT_DOUBLE_EQ(actual, trueVal);
}

TEST_F(EKFTest1d, LogLikelihood2) {
	colvec temp;
	colvec obs = {0.5};
	myekf.forecast(temp);
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	myekf.forecast(temp);
	colvec obs2 = {1.0};
	double actual = std::exp(myekf.logLikelihoodOfMeasurement(obs2));
	double trueVal = 1.0/std::sqrt(2.0 * datum::pi * (1.0+1.1)) * std::exp(-0.5 * (0.3 - 1.0) * (0.3 - 1.0) / (1.0+1.1));
	ASSERT_DOUBLE_EQ(actual, trueVal);
}

/* State estimation values (static problem)
* Forecast xkp1 = xk. Pkp1 = 0.5 + Pk
*  Time = 0											Update with d = 0.5
*	 u = 0.0, P = 1.0							K =  1.0/(1.0+1.0) = 0.5  u = 0.0 + 0.5 * 0.5 = 0.25  P = 0.5
*
*  Time = 0.5										Update with d = 1.0
*	 u = 0.25, P = 1.0							K =  1.0/(1.0+1.0) = 0.5  u = 0.25 + 0.5 * 0.75 = 0.625  P = 0.5*1.0 = 0.5
*
*  Time = 1.0										Update with d = -0.2
*	 u = 0.625, P = 1.0							u =
*
*/
TEST_F(EKFTest1d, LogLikelihoodDataSet) {
	colvec temp;
	mat obs = {0.5,1.0,-0.2}; //Measurements are taken at 0,0.5,1 seconds
	double actual = myekf.logLikelihood(obs, colvec());

	double ev1 = log(1.0/std::sqrt(2.0 * datum::pi * (1.0 + 1.0))) + (-0.5 * 0.5 * 0.5 / 2.0); //0.24000778968602721
	double ev2 = log(1.0/std::sqrt(2.0 * datum::pi * (1.0 + 1.0))) + (-0.5 * (0.25 - 1.0) * (0.25 - 1.0) / (1.0+1.0));
	double ev3 = log(1.0/std::sqrt(2.0 * datum::pi * (1.0+1.0))) + (-0.5 * (0.625 - -0.2) * (0.625 - -0.2) / (1.0+1.0));
	ASSERT_DOUBLE_EQ(actual, ev1+ev2+ev3);
}

/*
TEST_F(EKFTest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = myekf.logLikelihoodOfMeasurement(obs);

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );

	ASSERT_DOUBLE_EQ(actual, logpy);
}


*/



class EKFTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
		Gaussian * state = new Gaussian(u,P);
		mat::fixed<2,2> Q = {{0.0,0.0},{0.0,dt}};
		mat::fixed<1,1> R = {0.5};
		ss = statespace(f,dfdx,dfde,h,dhdx,dhde,dt,2,1);
		myekf = Ekf(state,ss,Q,R);
	}
	Ekf myekf;
	statespace ss;
};

TEST_F(EKFTest, checkMatrixSize) {
	bool actual = myekf.checkMatrixSize();
	ASSERT_TRUE(actual);
}

TEST_F(EKFTest, resetFilter) {
	colvec::fixed<2> expected = {2.0,3.0};
	colvec temp;
	myekf.forecast(temp);
	myekf.reset();
	colvec actual = myekf.getState()->getMean();
	ASSERT_TRUE(compareMatrix(actual,expected));
}


TEST_F(EKFTest, ForecastMean) {
	colvec temp;
	myekf.forecast(temp);
	colvec mean = myekf.getState()->getMean();
	colvec trueMean = colvec(2);
	trueMean[0] = 2.3;
	trueMean[1] = -7.9;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(EKFTest, ForecastCov) {
	colvec temp;
	myekf.forecast(temp);
	mat cov = myekf.getState()->getCovariance();
	mat trueCov = {{2.22,-9.66},{-9.66,44.08}};
	ASSERT_TRUE(compareMatrix(cov,trueCov));
}

TEST_F(EKFTest, UpdateKalman) {
	colvec temp;
	colvec obs = {1.0};
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat K = myekf.getKalmanGain();
	mat trueK = {0.8,0.4};
	trueK.reshape(2,1);
	ASSERT_TRUE(compareMatrix(K,trueK));
}

TEST_F(EKFTest, UpdateMean) {
	colvec temp;
	colvec obs = {1.0};
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec trueMean = {1.2,2.6};
	colvec mean = myekf.getState()->getMean();
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(EKFTest, UpdateCov) {
	colvec temp;
	colvec obs = {1.0};
	myekf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat cov = myekf.getState()->getCovariance();
	mat trueCov = {{0.4,0.2},{0.2,1.6}};
	ASSERT_TRUE(compareMatrix(cov,trueCov));
}

TEST_F(EKFTest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = myekf.logLikelihoodOfMeasurement(obs);

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );

	ASSERT_DOUBLE_EQ(actual, logpy);
}

/* It seems will never be NaN
TEST_F(EKFTest, LogLikelihoodNaN) {
	colvec temp;
	colvec obs = {std::numeric_limits<double>::max()};
	double actual = myekf.logLikelihood(obs);
	ASSERT_TRUE(isnan(actual));
}
*/

class ENKFTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		int N = 500000;
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
		mat::fixed<1,1> R = {0.5};
		mat::fixed<1,2> H = {1.0,0.0};
		ss = statespace(fs,hs,loglik,dt,2,1);
		ss.setMeasCov(R);
		myenkf = Enkf(u,P,N,ss);
	}
	Enkf myenkf;
	statespace ss;
};

TEST_F(ENKFTest, resetFilter) {
	colvec::fixed<2> expected = myenkf.getState().getMean();//{2.0,3.0};
	colvec temp;
	myenkf.forecast(temp);
	myenkf.reset();
	colvec actual = myenkf.getState().getMean();
	double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	double tol = 1.0e-2;
	ASSERT_LT(err, tol);
}


TEST_F(ENKFTest, ForecastMean) {
	colvec temp;
	myenkf.forecast(temp);
	colvec actual = myenkf.getState().getMean();
	colvec expected = colvec(2);
	expected[0] = 2.3;
	expected[1] = -7.9;
	double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	double tol = 1.0e-2;
	ASSERT_LT(err, tol);
}

TEST_F(ENKFTest, ForecastCov) {
	colvec temp;
	myenkf.forecast(temp);
	mat actual = myenkf.getState().getCovariance();
	mat expected = {{2.22,-9.66},{-9.66,44.08}};
	double err = std::max(std::max(std::abs(expected(0,0)-actual(0,0))/expected(0,0), std::abs(expected(0,1)-actual(0,1))/expected(0,1)),std::max(std::abs(expected(1,0)-actual(1,0))/expected(1,0),std::abs(expected(1,1)-actual(1,1))/expected(1,1) ));
	double tol = 1.0e-2;
	ASSERT_LT(err, tol);
}

TEST_F(ENKFTest, UpdateMean) {
	colvec temp;
	colvec obs = {1.0};
	myenkf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec expected = {1.2,2.6};
	colvec actual = myenkf.getState().getMean();
	double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	double tol = 1.0e-2;
	ASSERT_LT(err, tol);
}

TEST_F(ENKFTest, UpdateCov) {
	colvec temp;
	colvec obs = {1.0};
	myenkf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat actual = myenkf.getState().getCovariance();
	ASSERT_NEAR(actual.at(0,0), 0.4, 1.0e-2);
	ASSERT_NEAR(actual.at(0,1), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,0), 0.2, 1.0e-2);
	ASSERT_NEAR(actual.at(1,1), 1.6, 1.0e-2);
}

TEST_F(ENKFTest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = myenkf.logLikelihoodOfMeasurement(obs);

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );
	ASSERT_NEAR(actual, logpy, 1.0e-2);
}

/*
class PFTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		int N = 1000000;
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<2,2> P = {{2.0,1.0},{1.0,2.0}};
		Gaussian * state = new Gaussian(u,P);
		mat::fixed<2,2> Q = {{0.0,0.0},{0.0,dt}};
		mat::fixed<1,1> R = {0.5};
		ss = statespace(fs,h,dt);
		mypf = PF(u,P,N,ss,R);
	}
	PF mypf;
	statespace ss;
};

TEST_F(PFTest, resetFilter) {
	colvec::fixed<2> expected = mypf.getMean(); //{2.0,3.0};
	colvec temp;
	mypf.forecast(temp);
	mypf.reset();
	colvec actual = mypf.getMean();
	double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	double tol = 1.0e-3;
	ASSERT_LT(err, tol);
}


TEST_F(PFTest, ForecastMean) {
	colvec temp;
	mypf.forecast(temp);
	colvec actual = mypf.getMean();
	colvec expected = colvec(2);
	expected[0] = 2.3;
	expected[1] = -7.9;
	double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	double tol = 1.0e-3;
	ASSERT_LT(err, tol);
}

TEST_F(PFTest, ForecastCov) {
	colvec temp;
	mypf.forecast(temp);
	mat actual = mypf.getCovariance();
	mat expected = {{2.22,-9.66},{-9.66,44.08}};
	double err = std::max(std::max(std::abs(expected(0,0)-actual(0,0))/expected(0,0), std::abs(expected(0,1)-actual(0,1))/expected(0,1)),std::max(std::abs(expected(1,0)-actual(1,0))/expected(1,0),std::abs(expected(1,1)-actual(1,1))/expected(1,1) ));
	double tol = 1.0e-3;
	ASSERT_LT(err, tol);
}

TEST_F(PFTest, UpdateMean) {
	colvec temp;
	colvec obs = {1.0};
	mypf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	colvec expected = {1.2,2.6};
	colvec actual = mypf.getMean();
	double err = std::max(std::abs(expected[0]-actual[0])/expected[0], std::abs(expected[1]-actual[1])/expected[1]);
	double tol = 1.0e-3;
	ASSERT_LT(err, tol);
}

TEST_F(PFTest, UpdateCov) {
	colvec temp;
	colvec obs = {1.0};
	mypf.update(obs,temp); //to evaluate kalman gain, these are not important for this case
	mat actual = mypf.getCovariance();
	mat expected = {{0.4,0.2},{0.2,1.6}};
	double err = std::max(std::max(std::abs(expected(0,0)-actual(0,0))/expected(0,0), std::abs(expected(0,1)-actual(0,1))/expected(0,1)),std::max(std::abs(expected(1,0)-actual(1,0))/expected(1,0),std::abs(expected(1,1)-actual(1,1))/expected(1,1) ));
	double tol = 1.0e-2;
	ASSERT_LT(err, tol);
}

TEST_F(PFTest, LogLikelihood) {
	colvec temp;
	colvec obs = {1.0};
	double actual = mypf.logLikelihoodOfMeasurement(obs);

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );
	double err = std::abs(actual-logpy)/logpy;
	double tol = 1.0e-3;
	ASSERT_LT(err, tol);
	//ASSERT_DOUBLE_EQ(actual, logpy);
}
*/

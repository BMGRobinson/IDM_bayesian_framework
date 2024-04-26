#include <gtest/gtest.h>
#include "pdf.hpp"
#include <armadillo>
#include <cmath>
using namespace arma;

TEST(Gaussian1D, logDensity) {
	Gaussian1d gpdf = Gaussian1d(0.0, 1.0);
	double logdensity = gpdf.getLogDensity(0.5);
	ASSERT_DOUBLE_EQ(logdensity, log(1.0/sqrt(2.0*datum::pi)) - 0.5*0.5*0.5 );
}

TEST(PDF, logDensity2) {
	Gaussian1d * ypdf = new Gaussian1d(0.0, 4.0);
	double logpy = ypdf->getLogDensity( 0.0 );
	ASSERT_DOUBLE_EQ(logpy, log(1.0/(2.0*sqrt(2.0*PI))));
}

TEST(PDF, LogNormal1dMean) {
	double COV = 0.2;
	double median = 2.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	double mu = log(median);
	double scaleSqr = log(COV*COV+1.0);
	ASSERT_DOUBLE_EQ(ypdf->getMean() , exp(mu + scaleSqr/2.0) );
}

TEST(PDF, LogNormal1dCov) {
	double COV = 0.2;
	double median = 2.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	ASSERT_DOUBLE_EQ(ypdf->getCovariance(), 0.1664);
}

TEST(PDF, LogNormal1dMeanWolframAlpha) {
	double COV = 0.1;
	double median = 10.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	ASSERT_NEAR(ypdf->getMean() , 10.0498756211209, 1.0e-12);
}

TEST(PDF, LogNormal1dCovWolframAlpha) {
	double COV = 0.5;
	double median = 10.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	ASSERT_DOUBLE_EQ(ypdf->getCovariance(), 31.25);
}

TEST(PDF, LogNormal1d) {
	double COV = 0.5;
	double median = 0.2;
	LogNormal1d gpdf = LogNormal1d(median, COV);
	double val = 0.3;
	double logdensity = gpdf.getLogDensity(val);
	double scaleSqr = log(COV*COV+1.0);
	double mu = log(median);
	ASSERT_DOUBLE_EQ(logdensity, log(1.0/sqrt(val*val*scaleSqr*2.0*datum::pi)) - 0.5*(log(val)-mu)*(log(val)-mu)/scaleSqr );
}

TEST(PDF, LogNormal1d_2) {
	double COV = 4.0;
	double median = 10.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	double val = 0.5;
	double logdensity = ypdf -> getLogDensity(val);
	double scaleSqr = log(COV*COV+1.0);
	double mu = log(median);
	ASSERT_DOUBLE_EQ(logdensity, log(1.0/sqrt(val*val*scaleSqr*2.0*datum::pi)) - 0.5*(log(val)-mu)*(log(val)-mu)/scaleSqr );
}

TEST(PDF, LogNormal1dOutside) {
	double COV = 4.0;
	double median = 10.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	double val = -0.1;
	double logdensity = ypdf -> getLogDensity(val);
	ASSERT_DOUBLE_EQ(logdensity, 0.0 );
}

TEST(PDF, LogNormal1dZero) {
	double COV = 4.0;
	double median = 10.0;
	LogNormal1d * ypdf = new LogNormal1d(median, COV);
	double val = -0.1;
	double logdensity = ypdf -> getLogDensity(val);
	ASSERT_DOUBLE_EQ(logdensity, 0.0 );
}

TEST(PDF, SamplingFromGaussian1DMax) {
	Gaussian1d ypdf = Gaussian1d(10.0, 4.0);
	int i = 0;
	bool maxValue = false;
	for (i = 0; i < 10000; i++) {
		if ( std::abs(ypdf.sample()) > 1000) {
			maxValue = true;
		}
	}
	ASSERT_FALSE(maxValue);
}

TEST(PDF, Uniform1DDensityLeftBound) {
	Uniform1d * ypdf = new Uniform1d(0.0, 4.0);
	double logpy = ypdf->getLogDensity( 0.0 );
	ASSERT_DOUBLE_EQ(logpy, log(1.0/(4.0)));
}

TEST(PDF, Uniform1DDensityRightBound) {
	Uniform1d * ypdf = new Uniform1d(0.0, 4.0);
	double logpy = ypdf->getLogDensity( 4.0 );
	ASSERT_DOUBLE_EQ(logpy, log(1.0/(4.0)));
}

TEST(PDF, Uniform1DDensityInBounds) {
	Uniform1d * ypdf = new Uniform1d(0.0, 4.0);
	double logpy = ypdf->getLogDensity( 2.0 );
	ASSERT_DOUBLE_EQ(logpy, log(1.0/(4.0)));
}


TEST(PDF, Uniform1DDensityOutOfBounds) {
	Uniform1d * ypdf = new Uniform1d(0.0, 4.0);
	double logpy = ypdf->getLogDensity( 8.0 );

	ASSERT_TRUE(std::isnan(logpy));
}

TEST(PDF, Uniform1DDensityOutOfBoundsAddition) {
	Uniform1d * ypdf = new Uniform1d(0.0, 4.0);
	double logpy = ypdf->getLogDensity( 8.0 );
	logpy += 1.2;
	ASSERT_TRUE(std::isnan(logpy));
}

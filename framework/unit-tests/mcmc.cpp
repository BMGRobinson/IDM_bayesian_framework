#include <gtest/gtest.h>
#include "mcmc.hpp"
#include <armadillo>
#include <cmath>
using namespace arma;

double unNormalizedNormal( const colvec & sample ) {
	return exp(-0.5 * sample[0]*sample[0]);
}

/**********************************************
	* The following relationship holds true for
	* any sample of the MCMC chain
	*
	*   p(y'|d)   p(d|y') p(y')
	*   ------- = ------------
	*   p(y|d)    p(d|y)  p(y)
	**********************************************/

double getPosterior( const colvec & sample, const mat & D ) {
	//Gaussian1d meas_error = Gaussian1d(0.0, 2.0);
	Gaussian1d theta = Gaussian1d(10.0, 4.0);
	double diff = D(0,0)-sample[0];
	double loglikelihood = -0.5*diff*diff/2.0;
	double logprior = theta.getLogDensity(sample[0]); //1D sample
	return loglikelihood + logprior;
}

#include "samples.hpp"
double getPosteriorWrapper( const colvec & sample ) {
	//Generate measurement
	mat D = mat(1,1);
	//Gaussian1d meas_error = Gaussian1d(0.0, 2.0);
	D(0,0) = sample[0] +  sqrt(2.0)*randn();//meas_error.sample();
	return getPosterior(sample, D);
}

double gaussianPosterior( const colvec & sample ) {
	double diff = 0.0-sample[0];
	double loglikelihood = -0.5*diff*diff/4.0;
	return loglikelihood;
}
/*
* MCMC chain samples from a Gaussian Kernel. We test that the variance of the samples is the same as the target distribution with known variance
* Target distribution is N(0,4)
*/
/*
TEST(MCMC, GaussianSTDdeviation) {
	int L = 200000;
	mat proposal = mat(1,1);
	proposal(0,0) = 4.0;
	colvec sp = colvec(1);
	sp(0) = 0.5;
	mcmc chain = mcmc(L, proposal, gaussianPosterior);
	chain.disableScreenInfo();
	chain.setSartingPoint(sp);
	mat actual;
	chain.checkRejectionRatio(40.0,60.0);
	chain.checkMinIterations(10);

	if (chain.adaptiveProposal(1000)) {
		chain.metropolisHastings();
		actual = sqrt(chain.getChain().getCovariance());
	} else {
		actual = mat(1,1);
		actual(0,0) = -1;
	}
	double err = std::abs( (2.0-actual(0.0))/2.0);
	double tol = 0.02;

	ASSERT_LT(err, tol);
}
*/
/*
Do thesse tests really work?
 */
 /*
TEST(MCMC, gewekeAPplusMH) {
	int L = 10000;
	mat samples = mat(2, L);

	//Generate the first set of samples -> sampling from p(theta, D)
	for (int i = 0; i<L; i++) {
		samples(0,i) = 10.0 + 2.0 * randn();//
		//Generate the appropriate d sample assuming linear measurement	(here d = theta + N(0,2))
		samples(1,i) = samples(1,i) + sqrt(2.0)*randn(); // meas_error.sample();
	}

	samples.save("geweke_chain1.txt", raw_ascii);

	//Second method, using MCMC
	mat proposal = mat(1,1);
	proposal(0,0) = 5.0;
	colvec sp = colvec(1);
	sp(0) = samples(0,0);
	mcmc chain = mcmc(L, proposal,&getPosteriorWrapper);
	chain.disableScreenInfo();
	chain.setSartingPoint(sp);
	chain.checkRejectionRatio(60.0,90.0);
	chain.checkMinIterations(10);
	if (chain.adaptiveProposal(1000)) {;
		chain.metropolisHastings();
		chain.print(".","geweke_chain2");
	}
	//system("python geweke.py"); //run the python script
	std::cout << std::endl << "                ******************************************************************" << std::endl;
	std::cout << "                Check geweke.png, it should be a straight line from (0,0) to (1,1)" << std::endl;
	std::cout << "                ******************************************************************" << std::endl << std::endl;
}
*/
/*
TEST(MCMC, gewekeAM) {
	int L = 10000;
	mat samples = mat(2, L);

	//Generate the first set of samples -> sampling from p(theta, D)
	for (int i = 0; i<L; i++) {
		samples(0,i) = 10.0 + 2.0 * randn();//
		//Generate the appropriate d sample assuming linear measurement	(here d = theta + N(0,2))
		samples(1,i) = samples(1,i) + sqrt(2.0)*randn(); // meas_error.sample();
	}

	samples.save("geweke_chainB1.txt", raw_ascii);

	//Second method, using MCMC
	mat proposal = mat(1,1);
	proposal(0,0) = 5.0;
	colvec sp = colvec(1);
	sp(0) = samples(0,0);
	mcmc chain = mcmc(L, proposal,&getPosteriorWrapper);
	chain.disableScreenInfo();
	chain.setSartingPoint(sp);
	chain.checkMinMAPUnchangedIterations(5);
	chain.adaptiveProposal( 500, 5 ); //Pre-runs
	chain.adaptiveMetropolisburnin( 100 , 200 , 100000 );
	chain.adaptiveMetropolis( 200);

	chain.print(".","geweke_chainB2");

	system("python geweke.py"); //run the python script
	std::cout << std::endl << "                ******************************************************************" << std::endl;
	std::cout << "                Check gewekeB.png, it should be a straight line from (0,0) to (1,1)" << std::endl;
	std::cout << "                ******************************************************************" << std::endl << std::endl;
}
*/
/*
	TEST(MCMC, ratioTestEqualScalarSingleMeasurement) {
		Gaussian1d * ypdf = new Gaussian1d(0.5, 4.0);

		double y = 0.3;
		double yprime = 0.7;
		double d = 0.5;

		double logpy = ypdf->getLogDensity( y );
		double logpyprime = ypdf->getLogDensity( yprime );

		Gaussian1d * pDgivenY = new Gaussian1d(y, 4.0);
		Gaussian1d * pDgivenYprime = new Gaussian1d(yprime, 4.0);

		double logDgivenY = pDgivenY->getLogDensity(d);
		double logDgivenYprime = pDgivenYprime->getLogDensity(d);

		double logratio = logDgivenYprime + logpyprime - logpy - logDgivenY;
		mat proposal = mat(1,1);
		proposal.at(0,0) = 1.0;
		mcmc * mychain = new mcmc(100, proposal , getPosterior);
		mychain->run();
		mat sample = mychain->getChain();


	}
	*/

/*
TEST(MCMC, evidenceChibJeliazkovGaussian1D) {
	int L = 1000000;
	int Lj = 1000000;
	mat proposal = mat(1,1);
	proposal(0,0) = 1.0;
	colvec sp = colvec(1);
	colvec thetaStar = {0.0};
	sp(0) = 0.5;
	mcmc chain = mcmc(L, proposal, gaussianPosterior);	//sample from gaussian kernel with zero mean and variance 4
	chain.disableScreenInfo();
	chain.checkMinMAPUnchangedIterations(5);
	chain.setSartingPoint(sp);
	chain.adaptiveProposal( 500, 5 ); //Pre-runs
	//chain.adaptiveMetropolisburnin(100,1000, 100000);
	chain.DRAMburnin(100 , 200 , 100000 , 0.5, 0.1);
	chain.DRAM( 200, 0.2, 0.1);
	double actuallogevidence = chain.logEvidence(thetaStar,Lj, 1);
	double evidence = std::sqrt(2.0*datum::pi*4.0);
	double actual = std::exp(actuallogevidence);
	double err = std::abs( (evidence - actual)/actual );
	double tol = 1.0e-3;
	ASSERT_LT(err, tol);

	//ASSERT_DOUBLE_EQ(actuallogevidence,truelogevidence);

}
*/
TEST(MCMC, isAPowerOf2_TRUE) {
	ASSERT_TRUE(isAPowerOf2(8));
}

TEST(MCMC, isAPowerOf2_FALSE_EVEN) {
	ASSERT_FALSE(isAPowerOf2(10));
}

TEST(MCMC, isAPowerOf2_FALSE) {
	ASSERT_FALSE(isAPowerOf2(7));
}

TEST(MCMC, isAPowerOf2_ZERO) {
	ASSERT_FALSE(isAPowerOf2(0));
}

TEST(MCMC, twoExp_Zero) {
	ASSERT_EQ(twoExp(0) , 1);
}

TEST(MCMC, twoExp_1) {
	ASSERT_EQ(twoExp(1) , 2);
}

TEST(MCMC, twoExp_2) {
	ASSERT_EQ(twoExp(2) , 4);
}

TEST(MCMC, twoExp_6) {
	ASSERT_EQ(twoExp(6) , 64);
}

class ParallelCOVTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		lSamples = running_stat_vec<colvec>(true);
		gSamples = running_stat_vec<colvec>(true);
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
	}

	running_stat_vec<colvec> lSamples;
	running_stat_vec<colvec> gSamples;
	int id;
	int size;
};

TEST_F(ParallelCOVTest, 1D_COV) {
	colvec temp = zeros<colvec>(1);
	int N = 500;
	for (int i = 0; i < N*size; i ++) {
		temp[0] = 10.0*cos(2.0*double(i));
		gSamples(temp);

		if ( i >= id * N && i < (id+1)*N) {
		lSamples(temp);
		}
	}
	mat cov = getParallelCovariance(lSamples, MPI_COMM_WORLD );
	ASSERT_NEAR(gSamples.cov()(0,0),cov(0,0), 1.0e-8);
}

TEST_F(ParallelCOVTest, 2D_COV) {
	colvec temp = zeros<colvec>(2);
	int N = 500;
	for (int i = 0; i < N*size; i ++) {
		temp[0] = 10.0*cos(2.0*double(i));
		temp[1] = 5.0*sin(3.0*double(i));
		gSamples(temp);

		if ( i >= id * N && i < (id+1)*N) {
		lSamples(temp);
		}
	}
	mat cov = getParallelCovariance(lSamples, MPI_COMM_WORLD );
	ASSERT_NEAR(gSamples.cov()(0,0),cov(0,0), 1.0e-8);
	ASSERT_NEAR(gSamples.cov()(0,1),cov(0,1), 1.0e-8);
	ASSERT_NEAR(gSamples.cov()(1,0),cov(1,0), 1.0e-8);
	ASSERT_NEAR(gSamples.cov()(1,1),cov(1,1), 1.0e-8);
}

#include <gtest/gtest.h>
//#include "gtesthelper.hpp"
#include "filters.hpp"
#include "pdf.hpp"
#include "config.hpp"
#include "statespace.hpp"
#include <armadillo>
#include <cmath>
using namespace arma;

::testing::AssertionResult compareMatrix(const mat& A, const mat& B);

/*****************************************************************
******************************************************************
* Config file
******************************************************************
*****************************************************************/

TEST(ConfigFile, FilePresent) {
	Config cfg;
	bool flag = openConfigFile(cfg, "unit-tests.cfg");
	ASSERT_TRUE(flag);
}


TEST(ConfigFile, FileNotPresent) {
	Config cfg;
	bool flag = openConfigFile(cfg, "unit-tests2.cfg");
	ASSERT_FALSE(flag);
}


class ConfigFileFixture : public :: testing::Test {
protected:
	virtual void SetUp() {
	//Config cfg;
	openConfigFile(cfg, "unit-tests.cfg");
	}
	Config cfg;
};

TEST_F(ConfigFileFixture, ReadVector0Size) {
	colvec temp;
	readVector( temp, cfg, "vectorSize0" );
	int count = temp.size();
	ASSERT_EQ(0, count );
}


TEST_F(ConfigFileFixture, ReadVector1Size) {
	colvec temp;
	readVector( temp, cfg, "vectorSize1" );
	int count = temp.size();
	ASSERT_EQ(1, count );
}

TEST_F(ConfigFileFixture, ReadVector2Size) {
	colvec temp;
	readVector( temp, cfg, "vectorSize2" );
	int count = temp.size();
	ASSERT_EQ(2, count );
}


TEST_F(ConfigFileFixture, ReadVector1Value) {
	colvec temp;
	readVector( temp, cfg, "vectorSize1" );
	colvec trueValue = {0.0};
	ASSERT_TRUE(compareMatrix(temp,trueValue));
}

TEST_F(ConfigFileFixture, ReadVector3Value) {
	colvec temp;
	readVector( temp, cfg, "vectorSize3" );
	colvec trueValue = {0.0,1.0,2.0};
	ASSERT_TRUE(compareMatrix(temp,trueValue));
}


TEST_F(ConfigFileFixture, ReadMatrix2x3) {
	mat temp;
	readMatrix( temp, cfg, "Matrix2x3" );
	mat trueValue = {{0.0,1.0,2.0},{1.0,2.0,3.0}};
	ASSERT_TRUE(compareMatrix(temp,trueValue));
}

TEST_F(ConfigFileFixture, ReadGenModel) {
	std::vector<genModelParam> genModels;
	bool flag;
	flag = getGeneratingModelParameters( cfg, genModels );

	ASSERT_TRUE(flag);
}

TEST_F(ConfigFileFixture, ReadPropModel) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);

	ASSERT_TRUE(flag);
}

TEST_F(ConfigFileFixture, ReadPropModelDefaultProposal) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_TRUE(compareMatrix(propModels[0].mcmcParameters.initialProposal , eye<mat>(3,3)) );
}

TEST_F(ConfigFileFixture, ReadPropModelSamples) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_EQ( propModels[0].mcmcParameters.nsamples , 5000 );
}

TEST_F(ConfigFileFixture, ReadPropModelminIterations) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_EQ( propModels[0].mcmcParameters.minIterations , 10 );
}

TEST_F(ConfigFileFixture, ReadPropModelminMAPNotChangedIterations) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_EQ( propModels[0].mcmcParameters.minMAPNotChangedIterations , 5 );
}

TEST_F(ConfigFileFixture, ReadPropModelmaxSamplesUntilConvergence) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_EQ( propModels[0].mcmcParameters.maxSamplesUntilConvergence , 100000 );
}

TEST_F(ConfigFileFixture, ReadPropModeladaptationWindowLength) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_EQ( propModels[0].mcmcParameters.adaptationWindowLength , 500 );
}

TEST_F(ConfigFileFixture, ReadPropModelstartAdaptationIndex) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_EQ( propModels[0].mcmcParameters.startAdaptationIndex , 200 );
}

TEST_F(ConfigFileFixture, ReadPropModelsave_proposal_default) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_TRUE( propModels[0].mcmcParameters.save_proposal );
}

TEST_F(ConfigFileFixture, ReadPropModelDefaultStateVariance) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	ASSERT_TRUE(compareMatrix(propModels[0].initialStateVariance , eye<mat>(2,2) ));
}

TEST_F(ConfigFileFixture, ReadPropModelDefaultModelCovariance) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	mat trueValue = zeros<mat>(2,2);
	trueValue(1,1) = 1.0;
	ASSERT_TRUE(compareMatrix(propModels[0].modelCov , trueValue ));
}


TEST_F(ConfigFileFixture, TestPriors) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	mat trueValue = zeros<mat>(2,2);
	trueValue(1,1) = 1.0;

	double logPrior = 0.0;

	for(std::vector<pdf1d*>::iterator it = propModels[0].priors.begin(); it != propModels[0].priors.end(); ++it) {
    	logPrior += (*it)->getLogDensity(5.0);
	}

	Gaussian1d g1 = Gaussian1d(0.0,10.0);
	Gaussian1d g2 = Gaussian1d(5.0,10.0);
	Uniform1d u1 = Uniform1d(0.0,10.0);

	ASSERT_DOUBLE_EQ(logPrior , g1.getLogDensity(5.0)+g2.getLogDensity(5.0)+u1.getLogDensity(5.0) );
}

TEST_F(ConfigFileFixture, TestPriorsOutOfBounds) {
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);
	mat trueValue = zeros<mat>(2,2);
	trueValue(1,1) = 1.0;

	double logPrior = 0.0;

	for(std::vector<pdf1d*>::iterator it = propModels[0].priors.begin(); it != propModels[0].priors.end(); ++it) {
    	logPrior += (*it)->getLogDensity(-2.0);
	}

	ASSERT_TRUE(std::isnan(logPrior));
}

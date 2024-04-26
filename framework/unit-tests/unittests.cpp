#include <gtest/gtest.h>
#include "../SOURCE/mcmc.hpp"
#include "../SOURCE/pdf.hpp"
#include "../SOURCE/samples.hpp"
#include "../SOURCE/filters.hpp"
#include "../SOURCE/config.hpp"
#include "../SOURCE/bayesianPosterior.hpp"
#include <armadillo>
#include "../SOURCE/optimization.hpp"
#include <cmath>
using namespace arma;

/* For libconfig */
#include <libconfig.h++>
using namespace libconfig;

//Helper function to compare two matrices
::testing::AssertionResult compareMatrix(const mat& A, const mat& B) {
	int aCols, aRows, bCols, bRows;
	int i,j;

	aCols = A.n_cols;
	aRows = A.n_rows;
	bCols = B.n_cols;
	bRows = B.n_rows;

	if ((aCols != bCols) || (aRows != bRows) ) {
		return ::testing::AssertionFailure() << "Matrix size error. Actual (" << aRows << "," << aCols << ") != Expected (" << bRows << "," << bCols << ")";
	}

	double eps = 10e-8;
	for (i = 0; i < aRows; i++) {
		for (j = 0; j < aCols; j++) {
			if ( std::abs(A.at(i,j) - B.at(i,j)) > eps ) {
				return ::testing::AssertionFailure()
				<< "matrix[" << i << "," << j << "] (" << A.at(i,j) << ") != expected[" << i << "," << j << "] (" << B.at(i,j) << ")";
			}
		}
	}
	return ::testing::AssertionSuccess();
}

/*****************************************************************
******************************************************************
* PDF & Weighted Samples
******************************************************************
*****************************************************************/

/*****************************************************************
******************************************************************
* Samples & Weighted Samples
******************************************************************
*****************************************************************/


/*
	* Test the covariance

	TEST_F(WeightedSamplesTest, WeightedOneDimensionalCovariance) {
		mat actual = mysamples1d.getCovariance();
		mat expected = mat(1,1);
		expected.at(0,0) = 5.0/3.0;
		ASSERT_TRUE(compareMatrix(actual,expected));
	}

	TEST_F(WeightedSamplesTest, WeightedTwoDimensionalCovariance) {
		mat actual = mysamples2d.getCovariance();
		mat expected = mat(2,2);
		expected.at(0,0) = 5.0/3.0;
		expected.at(0,1) = -10.0/3.0;
		expected.at(1,0) = -10.0/3.0;
		expected.at(1,1) = 20.0/3.0;
		ASSERT_TRUE(compareMatrix(actual,expected));
	}

	TEST_F(WeightedSamplesTest, WeightedThreeDimensionalCovariance) {
		mat actual = mysamples3d.getCovariance();
		mat expected = mat(3,3);
		expected.at(0,0) = 5.0/3.0;
		expected.at(0,1) = 2.5/3.0;
		expected.at(0,2) = 5.0;
		expected.at(1,0) = 2.5/3.0;
		expected.at(1,1) = 1.25/3.0;
		expected.at(1,2) = 2.5;
		expected.at(2,0) = 5.0;
		expected.at(2,1) = 2.5;
		expected.at(2,2) = 49.0/3.0;
		ASSERT_TRUE(compareMatrix(actual,expected));
	}
	*/

/*****************************************************************
******************************************************************
* pdf
******************************************************************
*****************************************************************/




/*****************************************************************
******************************************************************
* MCMC
******************************************************************
*****************************************************************/






/*****************************************************************
******************************************************************
* Deterministic
******************************************************************
*****************************************************************/

/*
TEST_F(DeterministicTest, dataLogLikelihood) {
	colvec temp;
	//Three measurement points, with dty = 0.5 seconds
	mat data = {{1.0, 2.0, 0.0 },{0.0, 0.5, 1.0}};
	double actual = myse.loglikelihood(data, colvec() ); //No parameters
	double expected = 0.0;
	ASSERT_DOUBLE_EQ(expected, actual);
}
*/
/*****************************************************************
******************************************************************
* bayesianPosterior
******************************************************************
*****************************************************************/

/*
TEST(bayesianPosterior, quadrature1D) {
	colvec temp;
	colvec dummyParameters;
	Deterministic myse;
	statespace ss;
	mat::fixed<2,1> obs = {1.0,0.0};

	//State
	double dt = 0.1;/******/


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
	colvec::fixed<2> u = {2.0,3.0};
	ss = statespace(f,h,dt,2,1);
	myse = Deterministic(u,ss, {0.5});

	// Creating bayesian Posterior object
	bayesianPosterior bp = bayesianPosterior(obs, &myse);

	double actual = bp.evaluate( dummyParameters );

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );

	ASSERT_DOUBLE_EQ(actual, logpy);
}
*/
/*
TEST(bayesianPosterior, bpEKFevaluateFromConfigFile) {
	colvec temp;
	colvec dummyParameters;

	Config cfg;
	openConfigFile(cfg, "unit-tests.cfg");
	std::vector<proposedModels> propModels;
	bool flag;
	flag = getProposedModelParameters( cfg, propModels , MPI::COMM_WORLD);

	mat obs;
	obs.load(propModels[2].data);
	mat R;
	R.load(propModels[2].cov);

	// build the map
	std::map<std::string, statespace> func_map;

	statespace model1ss = statespace( f,dfdx,dfde,h,dhdx,dhde );
	func_map.insert(std::make_pair("func3",model1ss));

	// Statespace and filer
	statespace ss = func_map.find(propModels[2].function_handle)->second;

	ss.setDt( propModels[2].dt );

	Gaussian * state = new Gaussian(propModels[2].initialState, propModels[2].initialStateVariance  );

	Ekf mystateestimator = Ekf( state ,ss, propModels[2].modelCov , R);

	// Creating bayesian Posterior object
	bayesianPosterior bp = bayesianPosterior(obs, &mystateestimator);

	double actual = bp.evaluate( dummyParameters );

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );

	ASSERT_DOUBLE_EQ(actual, logpy);
}
*/

/*****************************************************************
******************************************************************
* Config file
******************************************************************
*****************************************************************/



/*
TEST_F(ConfigFileFixture, ReadMatrix) {
	colvec temp;
	colvec obs = {1.0};
	double actual = myekf.logLikelihoodOfMeasurement(obs);

	Gaussian1d * ypdf = new Gaussian1d(2.0, 2.5);
	double logpy = ypdf->getLogDensity( 1.0 );

	ASSERT_DOUBLE_EQ(actual, logpy);
}
*/

/*****************************************************************
******************************************************************
* Optimization
******************************************************************
*****************************************************************/
/*
double minfunc(const colvec& x) {
	return x[0]*x[0];
}

TEST(Optimization, Minx2) {
	int maxint = 100;
	nelderMead opt = nelderMead(minfunc);
	mat points = zeros<mat>(1,2);
	points(0,0) = -27.0;
	points(0,1) = 4.0;
	colvec point = opt.optimize( maxint, points );
	double actual = point[0];
	double err = std::abs( actual );
	double tol = 1.0e-7;
	ASSERT_LT(err, tol);
}
*/


int main(int argc, char **argv) {
	::testing::InitGoogleTest( &argc, argv );
	MPI::Init();
	bool result = RUN_ALL_TESTS();
	MPI::Finalize();
}

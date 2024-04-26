#include <gtest/gtest.h>
#include "gtesthelper.hpp"
#include "samples.hpp"
#include <armadillo>
#include <cmath>
using namespace arma;

class SamplesTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		int L = 4;
		mat samples1d = zeros<mat>(1,L);
		mat samples2d = zeros<mat>(2,L);
		mat samples3d = zeros<mat>(3,L);
		//Larger sample size
		mat samples3dL;
		samples3dL.load("samples3dL.txt");
		for (int i = 0; i < L; i ++) {
			samples1d(0,i) = double(i);

			samples2d(0,i) = double(i);
			samples2d(1,i) = - 2.0 * double(i);

			samples3d(0,i) = double(i);
			samples3d(1,i) = 0.5 * double(i);
			samples3d(2,i) = double(i) * double(i);
		}

		mysamples1d = Samples(samples1d);
		mysamples2d = Samples(samples2d);
		mysamples3d = Samples(samples3d);
		mysamples3dL = Samples(samples3dL);
	}


	Samples mysamples1d;
	Samples mysamples2d;
	Samples mysamples3d;
	Samples mysamples3dL;
};

TEST_F(SamplesTest, OneDimensionalSize) {
	int dim = mysamples1d.getDim();
	ASSERT_EQ(1,dim);
}

TEST_F(SamplesTest, TwoDimensionalSize) {
	int dim = mysamples2d.getDim();
	ASSERT_EQ(2,dim);
}

TEST_F(SamplesTest, ThreeDimensionalSize) {
	int dim = mysamples3d.getDim();
	ASSERT_EQ(3,dim);
}

TEST_F(SamplesTest, OneDimensionalMean) {
	colvec mean = mysamples1d.getMean();
	colvec trueMean = colvec(1);
	trueMean[0] = 1.5;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(SamplesTest, TwoDimensionalMean) {
	colvec mean = mysamples2d.getMean();
	colvec trueMean = colvec(2);
	trueMean[0] = 1.5;
	trueMean[1] = -3.0;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(SamplesTest, ThreeDimensionalMean) {
	colvec mean = mysamples3d.getMean();
	colvec trueMean = colvec(3);
	trueMean[0] = 1.5;
	trueMean[1] = 0.75;
	trueMean[2] = 3.5;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(SamplesTest, ThreeDimensionalMeanL) {
	colvec mean = mysamples3dL.getMean();
	colvec trueMean = colvec(3);
	trueMean[0] = 0.0071173487741673212986226459975114;
	trueMean[1] = 0.0071317707689530461523386151156956;
	trueMean[2] = -0.017272106841812847471961944734176;
	double abs_error = 0.001;
	ASSERT_NEAR(mean[0], trueMean[0],abs_error);
	ASSERT_NEAR(mean[1], trueMean[1],abs_error);
	ASSERT_NEAR(mean[2], trueMean[2],abs_error);
}

/*
	* Test the covariance
	*/
TEST_F(SamplesTest, OneDimensionalCovariance) {
	mat actual = mysamples1d.getCovariance();
	mat expected = mat(1,1);
	expected.at(0,0) = 5.0/3.0;
	ASSERT_TRUE(compareMatrix(actual,expected));
}

TEST_F(SamplesTest, TwoDimensionalCovariance) {
	mat actual = mysamples2d.getCovariance();
	mat expected = mat(2,2);
	expected.at(0,0) = 5.0/3.0;
	expected.at(0,1) = -10.0/3.0;
	expected.at(1,0) = -10.0/3.0;
	expected.at(1,1) = 20.0/3.0;
	ASSERT_TRUE(compareMatrix(actual,expected));
}

TEST_F(SamplesTest, ThreeDimensionalCovariance) {
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


TEST_F(SamplesTest, getAutocorrelationFunction) {
	mat actual = mysamples3d.getAutocorrelationFunction(4);
	mat expected = mat(4,4);
	double abs_error = 0.001;
	ASSERT_NEAR(actual.at(0,0), 1.0,abs_error);
	ASSERT_NEAR(actual.at(0,1), 0.25,abs_error);
	ASSERT_NEAR(actual.at(0,2), -0.3,abs_error);
	ASSERT_NEAR(actual.at(0,3), -0.45,abs_error);

	ASSERT_NEAR(actual.at(1,0), 1.0,abs_error);
	ASSERT_NEAR(actual.at(1,1), 0.25,abs_error);
	ASSERT_NEAR(actual.at(1,2), -0.3,abs_error);
	ASSERT_NEAR(actual.at(1,3), -0.45,abs_error);

	ASSERT_NEAR(actual.at(2,0), 1.0,abs_error);
	ASSERT_NEAR(actual.at(2,1), 0.209,abs_error); //see note
	ASSERT_NEAR(actual.at(2,2), -0.316,abs_error);
	ASSERT_NEAR(actual.at(2,3), -0.393,abs_error);

	ASSERT_NEAR(actual.at(3,0), 0.0,abs_error);
	ASSERT_NEAR(actual.at(3,1), 1.0,abs_error);
	ASSERT_NEAR(actual.at(3,2), 2.0,abs_error);
	ASSERT_NEAR(actual.at(3,3), 3.0,abs_error);

	//Note:Small difference here... with python I get 0.20899999...9
}

class WeightedSamplesTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		int L = 4;
		mat samples1d = zeros<mat>(1,L);
		rowvec weights = zeros<rowvec>(L);
		mat samples2d = zeros<mat>(2,L);
		mat samples3d = zeros<mat>(3,L);

		/* setting the weights */

		weights[0] = 0.01;
		weights[1] = 0.3;
		weights[2] = 0.5;
		weights[3] = 0.19;

		for (int i = 0; i < L; i ++) {
			samples1d(0,i) = double(i);

			samples2d(0,i) = double(i);
			samples2d(1,i) = - 2.0 * double(i);

			samples3d(0,i) = double(i);
			samples3d(1,i) = 0.5 * double(i);
			samples3d(2,i) = double(i) * double(i);
		}

		mysamples1d = WeightedSamples(samples1d);
		mysamples2d = WeightedSamples(samples2d);
		mysamples3d = WeightedSamples(samples3d);
		mywsamples1d = WeightedSamples(samples1d);
		mywsamples2d = WeightedSamples(samples2d);
		mywsamples3d = WeightedSamples(samples3d);
		mywsamples1d.setWeights( weights );
		mywsamples2d.setWeights( weights );
		mywsamples3d.setWeights( weights );
	}


	WeightedSamples mysamples1d;
	WeightedSamples mywsamples1d;
	WeightedSamples mysamples2d;
	WeightedSamples mywsamples2d;
	WeightedSamples mysamples3d;
	WeightedSamples mywsamples3d;
};

TEST_F(WeightedSamplesTest, OneDimensionalMean) {
	colvec mean = mysamples1d.getMean();
	colvec trueMean = colvec(1);
	trueMean[0] = 1.5;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(WeightedSamplesTest, TwoDimensionalMean) {
	colvec mean = mysamples2d.getMean();
	colvec trueMean = colvec(2);
	trueMean[0] = 1.5;
	trueMean[1] = -3.0;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(WeightedSamplesTest, ThreeDimensionalMean) {
	colvec mean = mysamples3d.getMean();
	colvec trueMean = colvec(3);
	trueMean[0] = 1.5;
	trueMean[1] = 0.75;
	trueMean[2] = 3.5;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}
/*
	* Test the covariance
	*/
TEST_F(WeightedSamplesTest, OneDimensionalCovariance) {
	mat actual = mysamples1d.getCovariance();
	mat expected = mat(1,1);
	expected.at(0,0) = 5.0/3.0;
	ASSERT_TRUE(compareMatrix(actual,expected));
}

TEST_F(WeightedSamplesTest, TwoDimensionalCovariance) {
	mat actual = mysamples2d.getCovariance();
	mat expected = mat(2,2);
	expected.at(0,0) = 5.0/3.0;
	expected.at(0,1) = -10.0/3.0;
	expected.at(1,0) = -10.0/3.0;
	expected.at(1,1) = 20.0/3.0;
	ASSERT_TRUE(compareMatrix(actual,expected));
}

TEST_F(WeightedSamplesTest, ThreeDimensionalCovariance) {
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

TEST_F(WeightedSamplesTest, WeightedOneDimensionalMean) {
	colvec mean = mywsamples1d.getMean();
	colvec trueMean = colvec(1);
	trueMean[0] = 1.87;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(WeightedSamplesTest, WeightedTwoDimensionalMean) {
	colvec mean = mywsamples2d.getMean();
	colvec trueMean = colvec(2);
	trueMean[0] = 1.87;
	trueMean[1] = -3.74;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

TEST_F(WeightedSamplesTest, WeightedThreeDimensionalMean) {
	colvec mean = mywsamples3d.getMean();
	colvec trueMean = colvec(3);
	trueMean[0] = 1.87;
	trueMean[1] = 0.935;
	trueMean[2] = 4.01;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}

/*
*
*	Parallel implementation
*
*/


class SamplesTestMPI : public :: testing::Test {
protected:
	virtual void SetUp() {

		MPI_Comm_size(MPI_COMM_WORLD, &size);
		L = 50*size;
		v1d = running_stat_vec<colvec>(true);
		v2d = running_stat_vec<colvec>(true);
		v3d = running_stat_vec<colvec>(true);
		mat samples1d = zeros<mat>(1, L);
		mat samples2d = zeros<mat>(2, L);
		mat samples3d = zeros<mat>(3, L);

		vec1d = zeros<colvec>(1);
		vec2d = zeros<colvec>(2);
		vec3d = zeros<colvec>(3);
		for (int i = 0; i < L; i ++) {
			samples1d(0, i) = 0.1*double(i);

			samples2d(0, i) = 0.1*double(i);
			samples2d(1, i) = - 0.4 * double(i);

			samples3d(0, i) = 0.1*double(i);
			samples3d(1, i) = 0.2 * double(i);
			samples3d(2, i) = std::sqrt(double(i));

			vec1d[0] = samples1d(0, i);

			vec2d[0] = samples2d(0, i);
			vec2d[1] = samples2d(1, i);

			vec3d[0] = samples3d(0, i);
			vec3d[1] = samples3d(1, i);
			vec3d[2] = samples3d(2, i);

			v1d(vec1d);
			v2d(vec2d);
			v3d(vec3d);

		}
		mysamples1d = SamplesMPI(samples1d, MPI::COMM_WORLD );
		mysamples2d = SamplesMPI(samples2d, MPI::COMM_WORLD );
		mysamples3d = SamplesMPI(samples3d, MPI::COMM_WORLD );
	}
	colvec vec1d,vec2d,vec3d;
	running_stat_vec<colvec> v1d,v2d,v3d;
	int size,L;
	SamplesMPI mysamples1d,mysamples2d,mysamples3d;
};

TEST_F(SamplesTestMPI, OneDimensionalSize) {
	int dim = mysamples1d.getDim();
	ASSERT_EQ(1, dim);
}

TEST_F(SamplesTestMPI, TwoDimensionalSize) {
	int dim = mysamples2d.getDim();
	ASSERT_EQ(2, dim);
}

TEST_F(SamplesTestMPI, ThreeDimensionalSize) {
	int dim = mysamples3d.getDim();
	ASSERT_EQ(3, dim);
}

TEST_F(SamplesTestMPI, OneDimensionalMean) {
	colvec mean = mysamples1d.getMean();
	//ASSERT_NEAR(mean[0], v1d.mean()[0], 1.0e-8 );
	ASSERT_TRUE(compareMatrix(mean, v1d.mean() ));
}

TEST_F(SamplesTestMPI, TwoDimensionalMean) {
	colvec mean = mysamples2d.getMean();
	ASSERT_TRUE(compareMatrix(mean, v2d.mean() ));
	//ASSERT_NEAR(mean[0], v2d.mean()[0], 1.0e-8 );
	//ASSERT_NEAR(mean[1], v2d.mean()[1], 1.0e-8 );
}

TEST_F(SamplesTestMPI, ThreeDimensionalMean) {
	colvec mean = mysamples3d.getMean();
	ASSERT_TRUE(compareMatrix(mean, v3d.mean() ));
}
/*
	* Test the covariance
	*/
TEST_F(SamplesTestMPI, OneDimensionalCovariance) {
	mat actual = mysamples1d.getCovariance();
	//ASSERT_TRUE(compareMatrix(actual, expected));
	ASSERT_NEAR(actual.at(0,0), v1d.cov().at(0,0), 1e-8);
}

TEST_F(SamplesTestMPI, TwoDimensionalCovariance) {
	mat actual = mysamples2d.getCovariance();
	ASSERT_TRUE(compareMatrix(actual, v2d.cov()));
}

TEST_F(SamplesTestMPI, ThreeDimensionalCovariance) {
	mat actual = mysamples3d.getCovariance();
	ASSERT_TRUE(compareMatrix(actual, v3d.cov()));
}

/*
TEST_F(SamplesTestMPI, getAutocorrelationFunction) {
	mat actual = mysamples3d.getAutocorrelationFunction(4);
	mat expected = mat(4, 4);
	double abs_error = 0.001;
	ASSERT_NEAR(actual.at(0, 0), 1.0, abs_error);
	ASSERT_NEAR(actual.at(0, 1), 0.25, abs_error);
	ASSERT_NEAR(actual.at(0, 2), -0.3, abs_error);
	ASSERT_NEAR(actual.at(0, 3), -0.45, abs_error);

	ASSERT_NEAR(actual.at(1, 0), 1.0, abs_error);
	ASSERT_NEAR(actual.at(1, 1), 0.25, abs_error);
	ASSERT_NEAR(actual.at(1, 2), -0.3, abs_error);
	ASSERT_NEAR(actual.at(1, 3), -0.45, abs_error);

	ASSERT_NEAR(actual.at(2, 0), 1.0, abs_error);
	ASSERT_NEAR(actual.at(2, 1), 0.209, abs_error); //see note
	ASSERT_NEAR(actual.at(2, 2), -0.316, abs_error);
	ASSERT_NEAR(actual.at(2, 3), -0.393, abs_error);

	ASSERT_NEAR(actual.at(3, 0), 0.0, abs_error);
	ASSERT_NEAR(actual.at(3, 1), 1.0, abs_error);
	ASSERT_NEAR(actual.at(3, 2), 2.0, abs_error);
	ASSERT_NEAR(actual.at(3, 3), 3.0, abs_error);

	//Note:Small difference here... with python I get 0.20899999...9
}
*/
//We are not chaning the weights here
class WeightedSamplesTestMPI : public :: testing::Test {
protected:
	virtual void SetUp() {

		MPI_Comm_size(MPI_COMM_WORLD, &size);
		L = 50*size;
		v1d = running_stat_vec<colvec>(true);
		v2d = running_stat_vec<colvec>(true);
		v3d = running_stat_vec<colvec>(true);

		vec1d = zeros<colvec>(1);
		vec2d = zeros<colvec>(2);
		vec3d = zeros<colvec>(3);

		mat samples1d = zeros<mat>(1, L);
		mat samples2d = zeros<mat>(2, L);
		mat samples3d = zeros<mat>(3, L);
		for (int i = 0; i < L; i ++) {
			samples1d(0, i) = 0.1*double(i);

			samples2d(0, i) = 0.1*double(i);
			samples2d(1, i) = - 0.4 * double(i);

			samples3d(0, i) = 0.1*double(i);
			samples3d(1, i) = 0.2 * double(i);
			samples3d(2, i) = std::sqrt(double(i));

			vec1d[0] = samples1d(0, i);

			vec2d[0] = samples2d(0, i);
			vec2d[1] = samples2d(1, i);

			vec3d[0] = samples3d(0, i);
			vec3d[1] = samples3d(1, i);
			vec3d[2] = samples3d(2, i);

			v1d(vec1d);
			v2d(vec2d);
			v3d(vec3d);
		}

		mysamples1d = WeightedSamplesMPI(samples1d, MPI::COMM_WORLD );
		mysamples2d = WeightedSamplesMPI(samples2d, MPI::COMM_WORLD );
		mysamples3d = WeightedSamplesMPI(samples3d, MPI::COMM_WORLD );
	}

	colvec vec1d,vec2d,vec3d;
	running_stat_vec<colvec> v1d,v2d,v3d;
	int size,L;
	WeightedSamplesMPI mysamples1d;
	WeightedSamplesMPI mysamples2d;
	WeightedSamplesMPI mysamples3d;
};

TEST_F(WeightedSamplesTestMPI, OneDimensionalSize) {
	int dim = mysamples1d.getDim();
	ASSERT_EQ(1, dim);
}

TEST_F(WeightedSamplesTestMPI, TwoDimensionalSize) {
	int dim = mysamples2d.getDim();
	ASSERT_EQ(2, dim);
}

TEST_F(WeightedSamplesTestMPI, ThreeDimensionalSize) {
	int dim = mysamples3d.getDim();
	ASSERT_EQ(3, dim);
}

TEST_F(WeightedSamplesTestMPI, OneDimensionalMean) {
	colvec mean = mysamples1d.getMean();
	ASSERT_TRUE(compareMatrix(mean, v1d.mean()));
}

TEST_F(WeightedSamplesTestMPI, TwoDimensionalMean) {
	colvec mean = mysamples2d.getMean();
	ASSERT_TRUE(compareMatrix(mean, v2d.mean()));
}

TEST_F(WeightedSamplesTestMPI, ThreeDimensionalMean) {
	colvec mean = mysamples3d.getMean();
	ASSERT_TRUE(compareMatrix(mean, v3d.mean()));
}
/*
	* Test the covariance
	*/
TEST_F(WeightedSamplesTestMPI, OneDimensionalCovariance) {
	mat actual = mysamples1d.getCovariance();
	//ASSERT_TRUE(compareMatrix(actual, expected));
	ASSERT_NEAR(actual.at(0,0), v1d.cov().at(0,0), 1e-3);
}

TEST_F(WeightedSamplesTestMPI, TwoDimensionalCovariance) {
	mat actual = mysamples2d.getCovariance();

	ASSERT_TRUE(compareMatrix(actual, v2d.cov()));
}

TEST_F(WeightedSamplesTestMPI, ThreeDimensionalCovariance) {
	mat actual = mysamples3d.getCovariance();
	ASSERT_TRUE(compareMatrix(actual, v3d.cov()));
}

/*
TEST_F(WeightedSamplesTestMPI, getAutocorrelationFunction) {
	mat actual = mysamples3d.getAutocorrelationFunction(4);
	mat expected = mat(4, 4);
	double abs_error = 0.001;
	ASSERT_NEAR(actual.at(0, 0), 1.0, abs_error);
	ASSERT_NEAR(actual.at(0, 1), 0.25, abs_error);
	ASSERT_NEAR(actual.at(0, 2), -0.3, abs_error);
	ASSERT_NEAR(actual.at(0, 3), -0.45, abs_error);

	ASSERT_NEAR(actual.at(1, 0), 1.0, abs_error);
	ASSERT_NEAR(actual.at(1, 1), 0.25, abs_error);
	ASSERT_NEAR(actual.at(1, 2), -0.3, abs_error);
	ASSERT_NEAR(actual.at(1, 3), -0.45, abs_error);

	ASSERT_NEAR(actual.at(2, 0), 1.0, abs_error);
	ASSERT_NEAR(actual.at(2, 1), 0.209, abs_error); //see note
	ASSERT_NEAR(actual.at(2, 2), -0.316, abs_error);
	ASSERT_NEAR(actual.at(2, 3), -0.393, abs_error);

	ASSERT_NEAR(actual.at(3, 0), 0.0, abs_error);
	ASSERT_NEAR(actual.at(3, 1), 1.0, abs_error);
	ASSERT_NEAR(actual.at(3, 2), 2.0, abs_error);
	ASSERT_NEAR(actual.at(3, 3), 3.0, abs_error);

	//Note:Small difference here... with python I get 0.20899999...9
}
*/

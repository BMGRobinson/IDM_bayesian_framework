#include <gtest/gtest.h>
#include "optimization.hpp"
#include <cmath>
using namespace arma;

TEST(GaussHermite, QuadraturePointsN2) {
	GaussHermite gh = GaussHermite(2, 1);

	double point = -0.7071067811865475; //1.0/std::sqrt(2.0);

	vec points = gh.getQuadraturePoints();

	ASSERT_NEAR( points[0] , point, 1.0e-15);
	ASSERT_NEAR( points[1], -1.0*point, 1.0e-15);
}

TEST(GaussHermite, QuadratureWeightsN2) {
	GaussHermite gh = GaussHermite(2, 1);
	double weight = 0.886226925452758;
	vec w = gh.getQuadratureWeights();

	ASSERT_NEAR( w[0] , weight, 1.0e-15);
	ASSERT_NEAR( w[1], weight, 1.0e-15);
}

TEST(GaussHermite, QuadratureWeightsN3) {
	GaussHermite gh = GaussHermite(3, 1);
	double weight1 = 0.295408975150919;
	double weight2 = 1.181635900603677;
	vec w = gh.getQuadratureWeights();

	ASSERT_NEAR( w[0] , weight1, 1.0e-15);
	ASSERT_NEAR( w[1], weight2, 1.0e-15);
	ASSERT_NEAR( w[2], weight1, 1.0e-15);
}

TEST(GaussHermite, QuadratureWeightsN4) {
	GaussHermite gh = GaussHermite(4, 1);
	double weight1 = 0.081312835447245;
	double weight2 = 0.804914090005513; //last digit 2
	vec w = gh.getQuadratureWeights();

	ASSERT_NEAR( w[0] , weight1, 1.0e-15);
	ASSERT_NEAR( w[1], weight2, 1.0e-15);
	ASSERT_NEAR( w[2], weight2, 1.0e-15);
	ASSERT_NEAR( w[3], weight1, 1.0e-15);
}

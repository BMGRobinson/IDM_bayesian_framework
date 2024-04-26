#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;
//const double c  = 0.02;
/** Generating model **/
//Model x' = W(t)
extern "C" colvec linear(const colvec & parameters , const colvec & state, const double time, const double dt ) {

	/* Record the noise */
	const double force = std::sqrt(dt)*randn();

	colvec temp = state;
	const double x = state[0];
	temp[0] = x + force;
	temp[1] = force;

	return temp;
}

/*************************************************************
************************** Model 1 ********************
*************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double x = state[0];
	colvec temp = state;

	temp[0] = x;
	return temp;
}

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

// Declare all the jacobians for EKF
mat dfdx (const colvec & state, const double dt, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	const double sigma = parameters[0];
	return {std::sqrt(dt)*sigma};
}

mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}


/*************************************************************
************************************ Build all the statespaces
*************************************************************/

extern "C" statespace model1SS = statespace(model1, dfdx, dfde, _h, dhdx, dhde, 1, 1);

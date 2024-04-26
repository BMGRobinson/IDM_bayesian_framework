rowvec H = {1.0,0.0};
mat modelMeasVariance;


colvec ekf_f( const colvec & state, const double dt, double time, const colvec & parameters ) {

	double c = 3.7699;
	double k = 8882.644;

	colvec temp = state;
	double x1 = state[0];
	double x2 = state[1];

	/* Explicit */
	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt * (- k * x1 - c * x2);

	return temp;
}

// jacobians
//

// Declare all the jacobians for EKF

mat dhdx (const colvec& , const colvec & parameters ) {
	return {1.0,0.0};
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}

mat dfdx (const colvec&, const double dt, const colvec & parameters ) {
	double m = 1.0;
	double c = 3.7699;
	double k = 8882.644;
	return {{1.0,dt},{-dt*k/m, 1.0-dt*c/m}};
}

mat dfde (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = 100.0;
	return {{0.0,0.0},{0.0,std::sqrt(dt)*sigma}};
}


/* Model for PF */
colvec pf_f(const colvec & state, const double dt, double time, const colvec & parameters ) {
	double c = 3.7699;
	double k = 8882.644;
	double sigmag = 100.0;

	double x1 = state[0];
	double x2 = state[1];

	colvec temp = state;

	temp[0] = x1 + dt * x2;
	temp[1] = x2 + dt*(-k*x1 - c*x2) + sigmag*std::sqrt(dt)*randn();
	return temp;
}

//
//	Measurement operators
//
//
colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

colvec hs = zeros<colvec>(1);
colvec _hs( const colvec & state, const mat & covariance, const colvec & parameters ) {
	hs[0] = state[0] + std::sqrt(covariance.at(0,0))*randn();
	return hs;
}

//Log function
double logfunc( const colvec& d, const colvec&  state, const colvec& parameters, const mat& cov) {
		double diff = d[0]-state[0];
		return -0.5*log(2.0*datum::pi*cov.at(0,0)) -0.5*diff*diff/cov.at(0,0);
}


/*Statespace */
statespace ss_ekf = statespace(ekf_f, dfdx, dfde, _h, dhdx, dhde, 2,1);
statespace ss_enkf = statespace( pf_f, _hs, logfunc,  2, 1);
statespace ss_pf = statespace( pf_f, _h, logfunc,  2, 1);

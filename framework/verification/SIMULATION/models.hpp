
//Quasi-steady model with white noise process
colvec model1q( const colvec & state, const double dt, double time, const colvec & parameters ) {

	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 	= parameters[1];
	double e3   = parameters[2];
	double e4		= parameters[3];
	double e4		= parameters[3];

	double theta 	= state[0];
	double thetadot = state[1];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);

	double cm = e1*theta + e2*thetadot + e3*thetacube + e4*thetasquare*thetadot;
	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;

	colvec temp = state;

	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot;
	return temp;
}

//Quasi-steady model with colored noise process
colvec model1qc( const colvec & state, const double dt, double time, const colvec & parameters ) {

	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 	= parameters[1];
	double e3   = parameters[2];
	double e4		= parameters[3];
	double e4		= parameters[3];

	double theta 	= state[0];
	double thetadot = state[1];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);

	double cm = e1*theta + e2*thetadot + e3*thetacube + e4*thetasquare*thetadot;
	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;

	colvec temp = state;

	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot;
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

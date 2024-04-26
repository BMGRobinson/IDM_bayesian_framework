#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

static bool dryfriction = false;

/* Constant parameters valid for all models */

const double rho =  1.1839000e+00;//  --> rho
const double I =   1.4000000e-03;//  --> I
const double ah = -6.4000000e-01;//  --> ah
const double c = 1.5600000e-01;//  --> c
const double s = 6.1000000e-01;//  --> s
//  -8.1640900e-01  --> struct1
//  -2.4205200e+02  --> struct2
//  -7.9280240e-01  --> struct3
//   2.1660790e+03  --> struct4
const double U =  8.5000000e+00 ;//  --> U (wind velocity)

const double tauTot = 2.0*U/c;
const double A = 0.5;
const double ebyc = 0.5*(0.5+ah);

const double c1		= -8.1640900e-01 / (tauTot * tauTot);
const double c2		= -2.4205200e+02 / (tauTot * tauTot);
const double c3		= -7.9280240e-01 / (tauTot);
const double c4		= 0.5*rho*U*U*c*c*s/(I * tauTot * tauTot);
const double c5		= 2.1660790e+03  / (tauTot * tauTot);
const double c6		= 2.0*datum::pi * ebyc * (0.5-ah)*(1.0-A);//parameters[5];

/*************************************************************
************************** Generating models ************************
**************************************************************/

extern "C" colvec white(const colvec & parameters , const colvec & state, const double time, const double dt ) {

std::cout << "C1 = " << c1 << std::endl;
std::cout << "C2 = " << c2 << std::endl;
std::cout << "C3 = " << c3 << std::endl;
std::cout << "C4 = " << c4 << std::endl;
std::cout << "C5 = " << c5 << std::endl;
std::cout << "C6 = " << c6 << std::endl;


//Change the timescale
double dtau = tauTot*dt;

double e1			= parameters[0];
double e2 		= parameters[1];
double e3   	= parameters[2];
double e4			= parameters[3];
double B			= parameters[4];
double sigma 	= parameters[5];

double theta 		= state[0];
double thetadot = state[1];
double cm 			= state[2];

double signThetaDot;

if (std::abs(theta) > 0.295*datum::pi/180.0) {
	dryfriction = true;
}

if (thetadot > 0.0) {
	signThetaDot = 1.0;
} else {
	signThetaDot = -1.0;
}

double thetasquare = pow(theta,2.0);
double thetacube	= pow(theta,3.0);
double force = std::sqrt(dtau)*sigma*randn();
double thetadotdot = c2*theta + c3*thetadot + c4*cm + c5*thetacube;

if (dryfriction) {
	thetadotdot += c1*signThetaDot;
}

colvec temp = state;

temp[0] = theta + dtau * thetadot;
temp[1] = thetadot + dtau*thetadotdot;
temp[2] = (1.0 - dtau*B)*cm + dtau*B*(e1*theta +e2*thetadot+e3*thetacube+e4*thetasquare*thetadot) + dtau*c6*thetadotdot + B*force;
temp[3] = force;
return temp;
}

/*************************************************************
************************** All models ************************
**************************************************************/

colvec h = zeros<colvec>(1);
colvec _h( const colvec & state, const mat & covariance, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

mat dhde (const colvec&, const colvec & parameters ) {
	return eye<mat>( 1, 1 );
}


#include "quasisteadyfunctions.hpp"
#include "unsteadyfunctions.hpp"


/*************************************************************
************************************ Stochastic models
*************************************************************/

colvec model1qs( const colvec & state, const double dt, double time, const colvec & parameters ) {

	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 	= parameters[1];
	double e3   = parameters[2];
	double e4		= parameters[3];
	double sigma = parameters[4];

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
	temp[1] = thetadot + dtau*thetadotdot + c4*std::sqrt(dtau)*sigma*randn();
	return temp;
}

colvec model1qcs( const colvec & state, const double dt, double time, const colvec & parameters ) {

	//Change the timescale
	double dtau = tauTot*dt;

	double e1			= parameters[0];
	double e2 		= parameters[1];
	double e3   	= parameters[2];
	double e4			= parameters[3];
	double tau		= parameters[4];
	double sigma  = parameters[5];

	double theta 				= state[0];
	double thetadot     = state[1];
	double qk						= state[2];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);

	double cm = e1*theta + e2*thetadot + e3*thetacube + e4*thetasquare*thetadot + qk;
	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube ;

	colvec temp = state;
	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot;
	temp[2] = std::exp(-dtau/tau) * qk + std::sqrt(sigma*sigma*tau*0.5*(1.0-std::exp(-2.0*dtau/tau)))*randn();
	return temp;
}

colvec model2qs( const colvec & state, const double dt, double time, const colvec & parameters ) {
	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 		= parameters[1];
	double e3   	= parameters[2];
	double e4		= parameters[3];
	double e5		= parameters[4];
	double sigma = parameters[5];


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
	double thetafifth = pow(theta,5.0);

	double cm = e1*theta + e2*thetadot + e3*thetacube + e4*thetasquare*thetadot + e5*thetafifth;

	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;

	colvec temp = state;

	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot + c4*std::sqrt(dtau)*sigma*randn();
	return temp;
}

colvec model2qcs( const colvec & state, const double dt, double time, const colvec & parameters ) {
	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 	= parameters[1];
	double e3   = parameters[2];
	double e4		= parameters[3];
	double e5		= parameters[4];
	double tau	= parameters[5];
	double sigma = parameters[6];

	double theta 	= state[0];
	double thetadot = state[1];
	double qk = state[2];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);
	double thetafifth = pow(theta,5.0);

	double cm = e1*theta + e2*thetadot + e3*thetacube + e4*thetasquare*thetadot + e5*thetafifth + qk;

	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;

	colvec temp = state;

	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot;
	temp[2] = std::exp(-dtau/tau) * qk + std::sqrt(sigma*sigma*tau*0.5*(1.0-std::exp(-2.0*dtau/tau)))*randn();
	return temp;
}

colvec model1us( const colvec & state, const double dt, double time, const colvec & parameters ) {

	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 	= parameters[1];
	double e3   = parameters[2];
	double e4		= parameters[3];
	double B		= parameters[4];
	double sigma = parameters[5];

	double theta 		= state[0];
	double thetadot = state[1];
	double cm 			= state[2];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);

	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;

	colvec temp = state;
	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot;
	temp[2] = (1.0 - dtau*B)*cm + dtau*B*(e1*theta +e2*thetadot+e3*thetacube+e4*thetasquare*thetadot) + dtau*c6*thetadotdot + std::sqrt(dtau)*B*sigma*randn();
	return temp;
}

colvec model1usPC( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double alpha = 0.5;
	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 	= parameters[1];
	double e3   = parameters[2];
	double e4		= parameters[3];
	double B		= parameters[4];
	double sigma = parameters[5];


	//Predictor

	double theta 		= state[0];
	double thetadot = state[1];
	double cm 			= state[2];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);

	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;
	double cmdot = -B*cm + B*(e1*theta +e2*thetadot+e3*thetacube+e4*thetasquare*thetadot) + c6*thetadotdot;
	//Predictor
	colvec corrector = state;
	double noise = randn();

	double predtheta = theta + dtau * thetadot;
	double predthetadot = thetadot + dtau*thetadotdot;
	double predcm = cm + dtau*cmdot + std::sqrt(dtau)*B*sigma*noise;

	//Corrector. Evaluate a() at the new position


	double signThetaDot_pred;
	if (predthetadot > 0.0) {
		signThetaDot_pred = 1.0;
	} else {
		signThetaDot_pred = -1.0;
	}

	double thetasquare_pred = pow(predtheta,2.0);
	double thetacube_pred	= pow(predtheta,3.0);


	double thetadot_corr = predthetadot;
	double thetadotdot_corr = c1*signThetaDot_pred + c2*predtheta + c3*predthetadot + c4*predcm + c5*thetacube_pred;
	double cmdot_corr = -B*cm + B*(e1*predtheta +e2*predthetadot+e3*thetacube_pred+e4*thetasquare_pred*predthetadot) + c6*thetadotdot_corr;

	//Corrector
	corrector[0] = theta + dtau*(alpha*thetadot_corr + (1.0 - alpha)*thetadot);
	corrector[1] = thetadot + dtau*(alpha*thetadotdot_corr + (1.0 - alpha)*thetadotdot);
 	corrector[2] = cm + dtau*(alpha*cmdot_corr + (1.0 - alpha)*cmdot) + std::sqrt(dtau)*B*sigma*noise;
	return corrector;
}

colvec model1ucs( const colvec & state, const double dt, double time, const colvec & parameters ) {
	//Change the timescale
	double dtau = tauTot*dt;

	double e1		= parameters[0];
	double e2 		= parameters[1];
	double e3   	= parameters[2];
	double e4		= parameters[3];
	double B		= parameters[4];
	double tau		= parameters[5];
	double sigma = parameters[6];

	double theta 	= state[0];
	double thetadot = state[1];
	double cm 		= state[2];
	double qk		= state[3];

	double signThetaDot;
	if (thetadot > 0.0) {
		signThetaDot = 1.0;
	} else {
		signThetaDot = -1.0;
	}

	double thetasquare = pow(theta,2.0);
	double thetacube	= pow(theta,3.0);

	double thetadotdot = c1*signThetaDot + c2*theta + c3*thetadot + c4*cm + c5*thetacube;

	colvec temp = state;

	temp[0] = theta + dtau * thetadot;
	temp[1] = thetadot + dtau*thetadotdot;
	temp[2] = (1.0 - dtau*B)*cm + dtau*B*(e1*theta +e2*thetadot+e3*thetacube+e4*thetasquare*thetadot) + dtau*c6*thetadotdot + dtau*B*qk;
	temp[3] = std::exp(-dtau/tau) * qk + std::sqrt(sigma*sigma*tau*0.5*(1.0-std::exp(-2.0*dtau/tau)))*randn();
	return temp;
}




//Log function
double logfunc( const colvec& d, const colvec&  state, const colvec& parameters, const mat& cov) {
		double diff = d[0]-state[0];
		//std::cout << "data point " << d[0] << " and theta " << state[0] << std::endl;
		//std::cout << "Cov at 0 is " << cov.at(0,0) << std::endl;
		//std::cout << "Particle lik: " << -0.5*log(2.0*datum::pi*cov.at(0,0)) -0.5*diff*diff/cov.at(0,0) << std::endl;
		return -0.5*log(2.0*datum::pi*cov.at(0,0)) -0.5*diff*diff/cov.at(0,0);
}



/*************************************************************
************************************ Build all the statespaces
*************************************************************/

//White noise variant
extern "C" statespace model1qSS = statespace(model1q, dfdx1q, dfde1q, _h, dhdx1q, dhde, 2, 1);
extern "C" statespace model2qSS = statespace(model2q, dfdx2q, dfde2q, _h, dhdx2q, dhde, 2, 1);

extern "C" statespace model1qSSs = statespace(model1qs, _h, logfunc, 2, 1);
extern "C" statespace model2qSSs = statespace(model2qs, _h, logfunc, 2, 1);

//colored noise variant, first order
extern "C" statespace model1qSSc = statespace(model1qc, dfdx1qc, dfde1qc, _h, dhdx1qc, dhde, 3, 1);
extern "C" statespace model1qSScb = statespace(model1qcb, dfdx1qc, dfde1qc, _h, dhdx1qc, dhde, 3, 1);
extern "C" statespace model2qSSc = statespace(model2qc, dfdx2qc, dfde2qc, _h, dhdx2qc, dhde, 3, 1);

//Colored noise quasi-steay stochastic
extern "C" statespace model1qSScs = statespace(model1qcs, _h, logfunc, 3, 1);
extern "C" statespace model2qSScs = statespace(model2qcs, _h, logfunc, 3, 1);


//Unsteady white noise
extern "C" statespace model1uSS = statespace(model1u, dfdx1u, dfde1u, _h, dhdx1u, dhde, 3, 1);
extern "C" statespace model2uSS = statespace(model2u, dfdx2u, dfde2u, _h, dhdx2u, dhde, 3, 1);

//Unsteady colored noise
extern "C" statespace model1uSSc = statespace(model1uc, dfdx1uc, dfde1uc, _h, dhdx1uc, dhde, 4, 1);
extern "C" statespace model2uSSc = statespace(model2uc, dfdx2uc, dfde2uc, _h, dhdx2uc, dhde, 4, 1);



extern "C" statespace model1uSSs = statespace(model1us, _h, logfunc, 3, 1);
extern "C" statespace model1uSSsPC = statespace(model1usPC, _h, logfunc, 3, 1);
extern "C" statespace model1uSScs = statespace(model1ucs, _h, logfunc, 4, 1);

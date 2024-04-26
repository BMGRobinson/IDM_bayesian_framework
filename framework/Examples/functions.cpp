#include <armadillo>
#include "statespace.hpp"
#include "pdf.hpp"
using namespace arma;

/** Generating model for example 1 **/
extern "C" colvec sirbeta(const colvec & parameters , const colvec & state, const double time, const double dt ) {
	const double gamma= 1.0/14.0;
	const double q = 1.0;
	
	double rho = parameters[0];

	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	temp[0] = sk + dt*(-betak*sk*ik);	 
	temp[1] = ik + dt*(betak*sk*ik - gamma*ik);
	temp[2] = rk + dt*(gamma*ik);
	temp[3] = betak;                                                                                                              	//
	return temp;
}

/*************************************************************
**********************SIR Model M1************************
*************************************************************/

colvec model1( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double rho = parameters[0];
	double beta = parameters[2];
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];

	colvec temp = state;

	if (time == 0 ) {
		ik = parameters[3];
		sk = 1.0-ik;
	}
	
	temp[0] = sk + dt*(-beta*sk*ik);	 
	temp[1] = ik + dt*(beta*sk*ik - gamma*ik);
	temp[2] = rk + dt*(gamma*ik);                                                                                                              	//                                                                              	//
	return temp;
}

colvec h1 = zeros<colvec>(1);
colvec _h1(const colvec & state, const mat & covariance, const colvec & parameters ) {
	double rho = parameters[0];

	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[3];
	}
	
	h1[0] = rho*ik;
	return h1;
}

// Declare all the jacobians for EKF
mat dfdx1(const colvec & state, const double dt, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double rho = parameters[0];
	double beta = parameters[2];
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];

	colvec temp = state;

	if (time == 0 ) {
		ik = parameters[3];
		sk = 1.0-ik;
	}

	return {{1.0 - dt*beta*ik, -dt*beta*sk, 0.0},
		{dt*beta*ik, 1.0 + dt*(beta*sk - gamma), 0.0, },
		{0.0, dt*gamma, 1.0}};
}

mat dfde1(const colvec & state, const double dt, const colvec & parameters ) {
	double rho = parameters[0];
	double q = parameters[1];
		
	double sk = state[0];
	double ik = state[1];

	if (time == 0 ) {
		ik = parameters[3];
		sk = 1.0-ik;
	}
	
	return {{-q*std::sqrt(dt)*sk*ik,0.0, 0.0},
		{q*std::sqrt(dt)*sk*ik, 0.0, 0.0},
		{0.0, 0.0, 0.0}};
}	

mat dhdx1(const colvec& , const colvec & parameters ) {
	double rho = parameters[0];
	
	return {0.0,rho,0.0};

}

mat dhde1(const colvec & state, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[3];
	}
	
	return {rho*ik + 1.0e-10};
}

static double staticTime;


/*************************************************************
**********************SIR Model M2************************
*************************************************************/
colvec model2( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double q = parameters[1];
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[1];
		ik = parameters[2];
		sk = 1.0 - ik;
	}

	temp[0] = sk + dt*(-betak*sk*ik);	 
	temp[1] = ik + dt*(betak*sk*ik - gamma*ik);
	temp[2] = rk + dt*(gamma*ik);
	temp[3] = betak;                                                                                                              	//                                                                              	//
	return temp;
}

colvec h2 = zeros<colvec>(1);
colvec _h2( const colvec & state, const mat & covariance, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[2];
	}
	
	h2[0] = rho*ik;
	return h2;
}

// Declare all the jacobians for EKF
mat dfdx2 (const colvec & state, const double dt, const colvec & parameters ) {
	const double gamma= 1.0/14.0;

	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[1];
		ik = parameters[2];
		sk = 1.0 - ik;
	}

	return {{1.0 - dt*betak*ik, -dt*betak*sk, 0.0, -dt*sk*ik},
		{dt*betak*ik, 1.0 + dt*(betak*sk - gamma), 0.0, dt*sk*ik},
		{0.0, dt*gamma, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0}};
}

mat dfde2 (const colvec & state, const double dt, const colvec & parameters ) {
	double q = 0.001;
	
	return {{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{q*std::sqrt(dt), 0.0, 0.0, 0.0}};
}	

mat dhdx2 (const colvec& , const colvec & parameters ) {
	double rho = parameters[0];
	
	return {0.0,rho,0.0,0.0};

}

mat dhde2 (const colvec & state, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[2];
	}
	
	return {rho*ik + 1.0e-10};
}

/*************************************************************
**********************SIR Model M3************************
*************************************************************/

colvec model3( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[1];
		ik = parameters[2];
		sk = 1.0 - ik;
	}
	
	temp[0] = sk + dt*(-betak*sk*ik);	 
	temp[1] = ik + dt*(betak*sk*ik - gamma*ik);
	temp[2] = rk + dt*(gamma*ik);
	temp[3] = betak;                                                                                                              	//                                                                              	//
	return temp;
}

colvec h3 = zeros<colvec>(1);
colvec _h3( const colvec & state, const mat & covariance, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[2];
	}
	
	h3[0] = rho*ik;
	return h3;
}

// Declare all the jacobians for EKF
mat dfdx3 (const colvec & state, const double dt, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[1];
		ik = parameters[2];
		sk = 1.0 - ik;
	}

	return {{1.0 - dt*betak*ik, -dt*betak*sk, 0.0, -dt*sk*ik},
		{dt*betak*ik, 1.0 + dt*(betak*sk - gamma), 0.0, dt*sk*ik},
		{0.0, dt*gamma, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0}};
}

mat dfde3 (const colvec & state, const double dt, const colvec & parameters ) {
	double q = 0.01;
	
	return {{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{q*std::sqrt(dt), 0.0, 0.0, 0.0}};
}	

mat dhdx3 (const colvec& , const colvec & parameters ) {
	double rho = parameters[0];
	
	return {0.0,rho,0.0,0.0};

}

mat dhde3 (const colvec & state, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[2];
	}
	
	return {rho*ik + 1.0e-10};
}

/*************************************************************
**********************SIR Model M4************************
*************************************************************/

colvec model4( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[2];
		ik = parameters[3];
		sk = 1.0 - ik;
	}
	
	temp[0] = sk + dt*(-betak*sk*ik);	 
	temp[1] = ik + dt*(betak*sk*ik - gamma*ik);
	temp[2] = rk + dt*(gamma*ik);
	temp[3] = betak;                                                                                                              	//                                                                              	//
	return temp;
}

colvec h4 = zeros<colvec>(1);
colvec _h4( const colvec & state, const mat & covariance, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[3];
	}
	
	h4[0] = rho*ik;
	return h4;
}

// Declare all the jacobians for EKF
mat dfdx4 (const colvec & state, const double dt, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[2];
		ik = parameters[3];
		sk = 1.0 - ik;
	}

	return {{1.0 - dt*betak*ik, -dt*betak*sk, 0.0, -dt*sk*ik},
		{dt*betak*ik, 1.0 + dt*(betak*sk - gamma), 0.0, dt*sk*ik},
		{0.0, dt*gamma, 1.0, 0.0},
		{0.0, 0.0, 0.0, 1.0}};
}

mat dfde4 (const colvec & state, const double dt, const colvec & parameters ) {
	double q = parameters[1];
	
	return {{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{q*std::sqrt(dt), 0.0, 0.0, 0.0}};
}	

mat dhdx4 (const colvec& , const colvec & parameters ) {
	double rho = parameters[0];
	
	return {0.0,rho,0.0,0.0};

}

mat dhde4 (const colvec & state, const colvec & parameters ) {
	double rho = parameters[0];
	
	double ik = state[1];
	
	if (time == 0 ) {
		ik = parameters[3];
	}	
	
	return {rho*ik + 1.0e-10};
}


/*************************************************************
*********************** ENFK SETUP *********************** INCOMPLETE
*************************************************************/
colvec model4s( const colvec & state, const double dt, double time, const colvec & parameters ) {
	const double gamma= 1.0/14.0;
	double q = parameters[1];
		
	double sk = state[0];
	double ik = state[1];
	double rk = state[2];
	double betak = state[3];

	colvec temp = state;

	if (time == 0 ) {
		betak = parameters[2];
		ik = parameters[3];
		sk = 1.0 - ik;
	}
	
	temp[0] = sk + dt*(-betak*sk*ik);	 
	temp[1] = ik + dt*(betak*sk*ik - gamma*ik);
	temp[2] = rk + dt*(gamma*ik);
	temp[3] = betak + q*std::sqrt(dt)*randn();;                                                                                                              	//                                                                              	//
	return temp;
}

colvec hs = zeros<colvec>(1);
colvec _hs( const colvec & state, const mat & covariance, const colvec & parameters ) {
	hs[0] = parameters[0]*state[1]*(1.0 + std::sqrt(covariance.at(0,0))*randn());
	return hs;
}

//Log function
double logfunc( const colvec& d, const colvec&  state, const colvec& parameters, const mat& cov) {
		double diff = (d[0]/(parameters[0]*state[1]+1.0e-10) - 1.0);
		return -0.5*log(2.0*datum::pi*cov.at(0,0)) - parameters[0]*state[1] -0.5*diff*diff/cov.at(0,0);
}	
	

/*************************************************************
************************************ Build all the statespaces
*************************************************************/

extern "C" statespace m1 = statespace(model1, dfdx1, dfde1, _h1, dhdx1, dhde1, 3, 1);

extern "C" statespace m2 = statespace(model2, dfdx2, dfde2, _h2, dhdx2, dhde2, 4, 1);

extern "C" statespace m3 = statespace(model3, dfdx3, dfde3, _h3, dhdx3, dhde3, 4, 1);

extern "C" statespace m4 = statespace(model4, dfdx4, dfde4, _h4, dhdx4, dhde4, 4, 1);

extern "C" statespace m4enkf = statespace(model4s, _hs, logfunc, 4, 1);

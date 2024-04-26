#include "solver.hpp"
using namespace arma;

solver::solver(const double _dt,const statespace & _ss) {
	dt = _dt;
	sqrtDt = std::sqrt(dt);
	ss = _ss;
}

basic::basic(const double _dt,const statespace & _ss) : solver( _dt, _ss) {}

//Used for KF, EKF, and filters of the sort
colvec 	basic::stateTimeUpdate(const colvec & state, const colvec&  parameters) {
	colvec Xdot = ss.evaluatef( state , parameters );
	return state + Xdot * dt;
}

mat basic::covTimeUpdate(const colvec & state, const mat & cov, const mat & Q, const colvec & parameters) {
	mat _dfdx = ss.evaluatedfdx( state, parameters );
	mat _dfde = ss.evaluatedfde( state, parameters );
	return  _dfdx * cov * trans(_dfdx) + _dfde * Q * trans(_dfde);
}


eulerMaruyama::eulerMaruyama(const double _dt,const statespace & _ss) : solver( _dt, _ss) {}

//	dX = F(X)dt + G(X)dW(t) using Euler–Maruyama method
//	Xkp1 = Xk + F(X) dt + sqrt(dt)G(x) * dW
//	Currently only implemented for a 1 dimensional dW
colvec eulerMaruyama::stateTimeUpdate(const colvec & state, const colvec &  parameters) {
	colvec Xdot = ss.evaluatef(state, parameters);
	colvec G		= ss.evaluateg(state, parameters);
	return state + Xdot*dt+sqrtDt*G*randn();
}

class predictorCorrector : public solver {
	colvec stateTimeUpdate();
};

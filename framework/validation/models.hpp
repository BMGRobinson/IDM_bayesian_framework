
mat data;
rowvec H = {1.0,0.0};
colvec h = zeros<colvec>(1);
mat dhde = zeros<mat>(1,1);
mat modelMeasVariance;

colvec _h( const colvec & state, const colvec & parameters ) {
	h[0] = state[0];
	return h;
}

mat _dhde (const colvec&, const colvec & parameters ) {
	dhde(0,0) = 1.0;
	return dhde;
}

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

mat _dfdx2 (const colvec&, const double dt, const colvec & parameters ) {
	double m = 1.0;
	double c = 3.7699;
	double k = 8882.644;
	return {{1.0,dt},{-dt*k/m, 1.0-dt*c/m}};
}

mat _dfde2 (const colvec&, const double dt, const colvec & parameters ) {
	double sigma = 100.0;
	return {{0.0,0.0},{0.0,std::sqrt(dt)*sigma}};
}

mat _dhdx2 (const colvec& , const colvec & parameters ) {
	//mat::fixed<2,1> I = {1.0,0.0};
	return {1.0,0.0};
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


/*Statespace */
statespace ss_ekf = statespace(ekf_f, _dfdx2, _dfde2, _h, _dhdx2, _dhde, 0.0001);
statespace ss_pf = statespace( pf_f, _h, 0.0001);
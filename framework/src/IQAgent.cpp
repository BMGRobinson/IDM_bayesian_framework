
#include "IQAgent.hpp"


/**
IQAgent class.
Based on
	-Numerical Recipes The Art of Scientific Computing 3rd Edition
	-Chambers, J.M., James, D.A., Lambert, D., and Vander Wiel, S. 2006, “Monitoring Networked
		Applications with Incremental Quantiles,” Statistical Science, vol. 21.

This is a modified version so it can run in parallel.
*/

IQAgent::IQAgent() {
	nq = 251;
	nt = 0;
	nd = 0;
	batch = 1000;
	q0 = 1.0e99;
	qm = -1.0e99;

	//Creating the vectors
	buffer = std::vector<double>(batch);
	pvalues = std::vector<double>(nq);
	qile  = std::vector<double>(nq, 0.0);
	tempqile = std::vector<double>(nq, 0.0);

	//P values of 10 to 90 %
	for (int j = 85; j <= 165; j ++) {
			pvalues[j] = (static_cast<double>(j) - 75.0)/100.0;
	}
	//The tails
	for (int j = 84; j>= 0 ; j--) {
		pvalues[j] = 0.87191909*pvalues[j+1];
		pvalues[250-j] = 1.0-pvalues[j];
	}

}

void IQAgent::add( double datum) {
	buffer[nd] = datum;
	nd ++;

	if (datum < q0) {
		q0 = datum;
	}

	if (datum > qm) {
		qm = datum;
	}

	if (nd == batch) {
		update();
	}
}

void IQAgent::update() {
	int jd,jq,iq;
	jd = 0;
	jq = 1;
	double target, tnew, told, qold, qnew;
	told = 0.0;
	tnew = 0.0;

	//Sort the values
	std::sort( buffer.begin(), buffer.end() );

	//Set lowest and highest to min and max seen
	qold = q0;
	qnew = q0;
	qile[0] = q0;
	tempqile[0] = q0;

	qile[nq-1] = qm;
	tempqile[nq-1] = qm;

	//Update the first and last value
	pvalues[0] = std::min(0.5/double(nt+nd), 0.5*pvalues[1]);
	pvalues[nq-1] = std::max(1.0-0.5/double(nt+nd), 0.5*(1.0+pvalues[nq-2]));
	//Do the remaining values
	for (iq = 1; iq < nq - 1; iq++) {
			target = double(nt+nd)*pvalues[iq];

			if (tnew < target) for(;;) {
					if (jq < nq && (jd >= nd || qile[jq] < buffer[jd])) {
						qnew = qile[jq];
						tnew = double(jd) + double(nt)*pvalues[jq];
						jq ++;
						if (tnew >= target) break;
					} else {
						qnew = buffer[jd];
						tnew = told;
						if (qile[jq] > qile[jq-1]) {
							tnew += double(nt)*(pvalues[jq]-pvalues[jq-1])*(qnew-qold)/(qile[jq]-qile[jq-1]);
						}
						jd ++;
						if (tnew >= target) break;
						told = tnew++;
						qold = qnew;
						if (tnew >= target) break;
					}
					told = tnew;
					qold = qnew;
				}
		if (tnew == told) tempqile[iq] = 0.5*(qold+qnew);
		else	tempqile[iq] = qold + (qnew-qold)*(target-told)/(tnew-told);
		told = tnew;
		qold = qnew;
	}

	//Copy values over
	for (iq = 0; iq < nq ; iq++) {
		qile[iq] = tempqile[iq];
		//std::cout << "Updating p " << pvalues[iq] << " -> " << qile[iq] << std::endl;
	}
	nt += nd;
	nd = 0;
}



double IQAgent::report(double p) {
	double q;

	//if (nd > 0) update();

	int jl, jh, j;
	jl = 0;
	jh = nq-1;

	while (jh-jl > 1) {
		j = (jh + jl)>>1;
		if (p > pvalues[j]) jl = j;
		else jh = j;
	}
	j = jl;
	q = qile[j] + (qile[j+1]-qile[j])*(p-pvalues[j])/(pvalues[j+1]-pvalues[j]);
	return std::max(qile[0], std::min(qile[nq-1], q));
}

//Set up parallel state estimation or parallel mcmc
void IQAgent::setCom( const MPI::Intracomm& _com) {
	//get id
	com = _com;
}

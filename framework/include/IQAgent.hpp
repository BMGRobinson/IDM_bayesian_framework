#ifndef IQAGENT_HPP_
#define IQAGENT_HPP_

//Following is used to disable warning from outside libraries
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <iomanip> //std::setw
#include <mpi.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>
#include <functional>
#pragma GCC diagnostic pop

using namespace arma;

/**
IQAgent class.
Based on
	-Numerical Recipes The Art of Scientific Computing 3rd Edition
	-Chambers, J.M., James, D.A., Lambert, D., and Vander Wiel, S. 2006, “Monitoring Networked
		Applications with Incremental Quantiles,” Statistical Science, vol. 21.

This is a modified version so it can run in parallel.
*/
class IQAgent {
public:
	/*
	*  Constructors
	*/
	IQAgent(); //default constructor required for gtest
	void add( double datum);
	double report(double p);

	//Set up parallel state estimation or parallel mcmc
	void setCom( const MPI::Intracomm& _com);

private:
	void update();

	double q0, qm;
	int nq, nt, nd, batch; //nd : new data points
	std::vector<double>  buffer, pvalues, qile, tempqile;
	//Communicator used when parallel
	bool parallel;
	MPI::Intracomm com;
	int id;
};
#endif

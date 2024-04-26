#ifndef OPTIMIZATION_HPP_
#define OPTIMIZATION_HPP_

//Following is used to disable warning from outside libraries
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <mpi.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>
#include "bayesianPosterior.hpp"
#pragma GCC diagnostic pop

using namespace arma;


typedef std::function<long double (const colvec&)> optfunc;
typedef std::vector<int> V;

class GaussHermite {
	public:
		double quadrature(double tol, int MAXIT);
		GaussHermite(int order, int dim);
		GaussHermite(bayesianPosterior& _bp , int order, int dim);
		GaussHermite(optfunc _f , int order, int dim);
		double getPolynomial(int n, double x);
		double getWeight(int n, double x);
		int factorial(int n) ;
		void calculatePoints(int n);
		vec getQuadraturePoints();
		void calculateWeights( int n );
		vec getQuadratureWeights();
		void setMultivariateQuadPoints(int order, int dim, int ncols);
		long double fwrapper( colvec x);
		void setInitialSigma( mat sigma );
		void setInitialMean( colvec mu );
		//colvec mean;
		//mat sigma;
	private:
		/* function to optimize */
		optfunc f;
		bayesianPosterior bp;
		int dim;
		int order;
		int ncols;
		colvec mean;
		mat sigma;
		vec quadraturePoints;
		vec quadratureWeights;
		mat multivariateQuadPoints;
		vec multivariateQuadWeights;
		//variables & functions to get permutation matrix
		int col;
		Mat<int> getPermutationMatrix(int dim, int order);
		void permute(Mat<int>& permMatrix, Col<int>& permVec, Col<int>& temp, int start, int end, int index, int r , int R);
};

/* Find the minimum of a function */
class nelderMead {
	public:
		/* flags */
		bool print;
		nelderMead(optfunc _f );
		nelderMead(double _alpha, double _gamma, double _rho, double _sigma, optfunc _f);
		nelderMead( bayesianPosterior& bp );
		colvec optimize(const int maxIt, const mat start);
		void setStateEstimatorCom( const MPI::Intracomm& _com);
	private:
		/* Map where key is the function evaluation and the value is the trial points */
		std::vector< std::pair<double, colvec> > values;

		/* maximum number of iterations */
		int maxIt;
		int n; //size of point
		int id; //id of local communicator
		/* points and their associated function evaluations */
		colvec reflectedPoint;
		double fr;
		colvec expandedPoint;
		double fe;
		colvec contractedPoint;
		double fc;
		colvec centroid;

		//Communicator used when state estimation is parallel
		bool parallelStateEstimation;
		MPI::Intracomm statecom;

		/* function to optimize */
		optfunc f;
		bayesianPosterior bp;

		void order();
		void reflection();
		void expansion();
		void contraction();
		void reduction();

		/* Coefficients */
		double alpha;
		double gamma;
		double rho;
		double sigma;

		long double fwrapper( colvec x);
};

#endif

#include "optimization.hpp"
using namespace arma;


/*
*
*	Gauss-Hermite
*
*/

/*
*	Helper functions to get permutation matrix
*
*/

void GaussHermite::permute(Mat<int>& permMatrix, Col<int>& permVec, Col<int>& temp, int start, int end, int index, int r , int R) {
	if (r == 0) {
        for (int j=0; j<R; j++) {
			permMatrix(j,col) = temp[j];
		}
		col ++;
        return;
	}

	for (int i=0; i<end; i++)
    {
        temp[index] = permVec[i];
        permute(permMatrix, permVec, temp, start+1, end, index+1, r-1, R);
    }

}

Mat<int> GaussHermite::getPermutationMatrix(int dim, int order) {

	int ncols = pow(order,dim);
	Mat<int> permMatrix = Mat<int>(dim, ncols); //Initialize matrix

	col = 0;

	Col<int>  indices = Col<int>(order);

	for (int i = 0; i < order; i++) {
		indices[i] = i;
	}

	Col<int> tempIndices = Col<int>(dim);

	permute(permMatrix, indices , tempIndices, 0, order, 0, dim ,dim);

	return permMatrix;
}

double GaussHermite::quadrature(double tol, int MAXIT) {
	double previdence = 0.0;
	double err;
	double	evidence = 0.0;
	double logevidence, prelogevidence;
	prelogevidence = 0.0;
	double norm_constant;
	int it = 0;
	int Nq = multivariateQuadPoints.n_cols;
	mat L;
	double sqrtDetSigma;

	double alpha = 1.0; //relaxation parameter

	colvec point;
	colvec theta;
	colvec previousMean;

	long double * fEvaluations = new long double[Nq];
	long double * logEvals = new long double[Nq];
	std::cout << "Starting iterative procedure." << std::endl;
	double overflowProtect;
	double a1,a2;
	//sigma = sigma*0.01;
	while (it < MAXIT) {
		//Evidence
		sqrtDetSigma = std::sqrt(det(sigma));
		previousMean = mean;
		L = trans(chol(sigma));

		//Reset values
		mean.zeros();
		sigma.zeros();
		evidence = 0.0;

		//Sometimes the logEvidence is too l
		overflowProtect = 0.0;
		for (int i = 0; i < Nq; i ++) {
			theta = multivariateQuadPoints.unsafe_col(i);
			point = L*theta + previousMean;
			a1 = fwrapper( point );
			a2 = 0.5*as_scalar(trans(theta)*theta);
			logEvals[i] = log(multivariateQuadWeights(i)) + a1 + a2;
			if (!std::isnan(a1)) {
				overflowProtect += logEvals[i];
			}
		}
		overflowProtect = overflowProtect / double(Nq);
		for (int i = 0; i < Nq; i ++) {
			if (std::isnan(logEvals[i]))  {
				fEvaluations[i] = std::exp(- overflowProtect);
			} else {
				fEvaluations[i] = std::exp(logEvals[i] - overflowProtect);
			}
			evidence += fEvaluations[i] ;
		}
		norm_constant = evidence;

		logevidence = log(evidence) + log(std::sqrt(pow(2.0*datum::pi, dim)))  + log(sqrtDetSigma) + overflowProtect;
		err = std::abs(logevidence - prelogevidence);
		std::cout << "Iteration: " << it << " logevidence is " << logevidence << " log error is " << err << " log tolerance is " << -1.0*log(1.0-tol) << std::endl;
		if (err < -1.0*log(1.0-tol)) {
			std::cout << "\t \t Completed! logEvidence is " << logevidence << std::endl;
			return logevidence;
		}
		prelogevidence = logevidence;

		//Update the mean
		for (int i = 0; i < Nq; i ++) {
			theta = multivariateQuadPoints.unsafe_col(i);
			if (!std::isnan(fEvaluations[i])) {
				mean += fEvaluations[i] * (L*theta + previousMean);
			}
		}
		mean = mean / norm_constant;
		mean.print("Mean:");

		//Update the covariance

		for (int i = 0; i < Nq; i ++) {
			theta = multivariateQuadPoints.unsafe_col(i);
			if (!std::isnan(fEvaluations[i])) {
				sigma += fEvaluations[i] * ((L*theta)*trans(L*theta));
			}
		}
		sigma = sigma / norm_constant;
		sigma.print("Sigma:");

		previousMean = mean;

		it++;
	}
}

GaussHermite::GaussHermite(int order, int _dim) {
	dim = _dim;
	mean = zeros<colvec>(dim);
	sigma = eye<mat>(dim,dim);
	calculatePoints( order );
	//quadraturePoints.print("Quadrature points used: ");
	quadratureWeights = zeros<vec>(order);
	calculateWeights( order );
	ncols = 1;

	for (int i = 0; i < dim; i ++) {
		ncols *= order;
	}

	multivariateQuadPoints = zeros<mat>(dim, ncols );
	multivariateQuadWeights = zeros<vec>(ncols);
	setMultivariateQuadPoints(order, dim, ncols);

	//multivariateQuadPoints.print("Points");
	//multivariateQuadWeights.print("Associated Weights");
}

GaussHermite::GaussHermite(bayesianPosterior& _bp, int order, int dim) : GaussHermite(order,dim) {
	//std::cout << "In Gauss-Hermite constructor." << std::endl << "\t The dimension of the problem is " << dim << std::endl << "\t The number of quadrature points is " << order << std::endl;
	bp = _bp;
	f = 0;

}

GaussHermite::GaussHermite(optfunc _f , int order, int dim) : GaussHermite(order,dim) {
	f = _f;
}

void GaussHermite::setInitialSigma( mat _sigma ) {
	sigma = _sigma;
}

void GaussHermite::setInitialMean( colvec mu ) {
	mean = mu;
}

//Recrusive method to compute the polynomial
double GaussHermite::getPolynomial(int n, double x) {
	if (n == 0) {
		return 1.0;
	} else if (n == 1) {
		return 2.0*x;
	} else {
		return 2.0*x*getPolynomial(n-1, x) - 2.0*double(n-1)*getPolynomial(n-2, x);
	}
}


long double GaussHermite::fwrapper( colvec x) {
	long double result;
	//std::cout << "f is " << f << std::endl;
	if (f == 0) {
		result =  bp.evaluate( x );
	} else {
		result = f(x);
	}
	return result;
}

void GaussHermite::setMultivariateQuadPoints(int order, int dim, int ncols) {
	int i,j;
	double w;
	Mat<int> permMatrix;

	permMatrix = getPermutationMatrix(dim, order);
	//Mat<int> permMatrix = {{0,0,0,1,1,1,2,2,2},{0,1,2,0,1,2,0,1,2}};
	for (j = 0; j < ncols; j ++) {
		w = 1.0;
		for (i = 0; i < dim; i ++) {
			multivariateQuadPoints(i,j) = quadraturePoints[permMatrix(i,j)] * std::sqrt(2.0);
			w *= quadratureWeights[permMatrix(i,j)] / std::sqrt(datum::pi);
		}
		multivariateQuadWeights(j) = w;
	}

	/*
	for (j = 0; j < ncols; j ++) {
		multivariateQuadPoints(0,j) = quadraturePoints[j] * std::sqrt(2.0);
	}
	multivariateQuadWeights = quadratureWeights / std::sqrt(datum::pi);
	*/
}

void GaussHermite::calculateWeights( int n ) {
	for (int i = 0; i < n; i ++) {
		quadratureWeights[i] = getWeight(n, quadraturePoints[i] );
		//std::cout << "value of i is " << i << std::endl;
	}
}

vec GaussHermite::getQuadratureWeights() {
	return quadratureWeights;
}

vec GaussHermite::getQuadraturePoints() {
	return quadraturePoints;
}

void GaussHermite::calculatePoints(int n) {
	//build the symetric matrix
	mat J = zeros<mat>(n,n);
	double val;
	//for each row
	for (int i = 0; i < n-1; i ++) {
		val = std::sqrt(double(i+1) / 2.0);
		J(i,i+1) = val;
		J(i+1,i) = val;
	}
	//Get the Eigenvalues
	eig_sym( quadraturePoints, J );
}

double GaussHermite::getWeight(int n, double x) {
	//std::cout << "in get weight" << std::endl;
	double hermitePolynomial = getPolynomial(n-1, x);
	//std::cout << "The hermite polynomial is " << hermitePolynomial << std::endl;
	double weight = pow(2.0, n-1)*double( factorial(n) ) * std::sqrt(datum::pi) / (double(n*n) * pow(hermitePolynomial, 2.0));
	//std::cout << "Weight :" << weight << std::endl;
	//std::cout << pow(2.0, n-1) << ", " << double( factorial(n) ) << " , " << double(n*n) << " , " << pow(hermitePolynomial, 2.0) << std::endl;
	return weight;
}

int GaussHermite::factorial(int n) {
	//std::cout << "In factorial and n is " << n << std::endl;
	if (n == 0 || n == 1) {
		//std::cout << " n is " << n << " and returning 1" << std::endl;
		return 1;
	} else {
		return n*factorial(n - 1);
	}
}

nelderMead::nelderMead(optfunc _f ) {
	print = false;
	alpha = 1.0;
	gamma = 2.0;
	rho = -0.5;
	sigma = 0.5;
	f = _f;
	parallelStateEstimation = false;
	id = 0;
}

nelderMead::nelderMead( bayesianPosterior& _bp ) {
	print = false;
	alpha = 1.0;
	gamma = 2.0;
	rho = -0.5;
	sigma = 0.5;
	bp = _bp;
	f = 0;
	id = 0;
	parallelStateEstimation = false;
}

void nelderMead::setStateEstimatorCom( const MPI::Intracomm& _com) {
	statecom = _com;
	id = statecom.Get_rank();
	parallelStateEstimation = true;
}

nelderMead::nelderMead(double _alpha, double _gamma, double _rho, double _sigma, optfunc _f) {
	print = false;
	alpha = _alpha;
	gamma = _gamma;
	rho = _rho;
	sigma = _sigma;
	f = _f;
}


long double nelderMead::fwrapper( colvec x) {
	long double result;

	if (parallelStateEstimation) {
		statecom.Bcast(x.memptr(), x.n_rows , MPI::LONG_DOUBLE, 0);
	}

	if (f == 0) {
			result =  -1.0 * bp.evaluate( x ); //we want the max value
		} else {
			result = f(x);
		}
	if (parallelStateEstimation) {
		statecom.Barrier();
	}
		return result;
	}

colvec nelderMead::optimize( const int _maxIt, const mat start ) {
	n = start.n_rows;
	maxIt = _maxIt;
	int i = 0;

	/* Initialization */
	for (unsigned int j = 0; j < start.n_cols; j++) {
		values.push_back( std::make_pair(fwrapper(start.unsafe_col(j)), start.unsafe_col(j) ) );
	}

	while (i < maxIt ) {
		order();
		reflection();
		//reflection is either the best, or better than second last

		if (fr >= values.front().first && fr < values.rbegin()[1].first) {
			values.pop_back(); //delete the last element
			values.push_back( std::make_pair(fr, reflectedPoint) ); //add the reflectedPoint
		} else if ( fr < values.front().first ) { //reflection gave the best point
			expansion();
			//if the expansion is better than the reflection
			if (fe < fr) {
				//Swap the expansion with the last element
				values.pop_back(); //delete the last element
				values.push_back( std::make_pair(fe, expandedPoint) ); //add the expansionPoint
			} else {
				//Swap the reflection with the last element
				values.pop_back(); //delete the last element
				values.push_back( std::make_pair(fr, reflectedPoint) ); //add the reflectedPoint
			}
		} else {
			contraction();
			if ( fc < values.back().first ) { //contraction point is better than last point
				values.pop_back(); //delete the last element
				values.push_back( std::make_pair(fc, contractedPoint) ); //add the contractedPoint
			} else {
				reduction(); //No proposed point was good enough, we keep the best one and start over
			}
		}
		if (print && id == 0) {
			std::cout << "Iteration " << i << " best function evaluation is " << values.front().first << std::endl;
		}
		i++;
	}
	return values.front().second;
}

void nelderMead::order() {
	/* Sort the vector */
	std::sort( values.begin(), values.end(), [](const std::pair<double,colvec> &left, const std::pair<double,colvec> &right)
	{return left.first < right.first;} );

	//Get the centroid except for the max value
	centroid = zeros<colvec>(n);
	for (auto it = values.begin() ; it != values.end()-1; ++it) {
		centroid += it->second;
	}
	//remove last element,
	centroid = centroid / static_cast<double>(n);
}

void nelderMead::reflection() {
	reflectedPoint = centroid + alpha*(centroid - values.back().second);
	fr = fwrapper(reflectedPoint);
}
void nelderMead::expansion() {
	expandedPoint = centroid + gamma*(centroid - values.back().second);
	fe = fwrapper(expandedPoint);
}
void nelderMead::contraction() {
	contractedPoint = centroid + rho*(centroid - values.back().second);
	fc = fwrapper(contractedPoint);
}

void nelderMead::reduction() {
	for (auto it = values.begin()+1 ; it != values.end(); ++it) {
		it->first = values.front().first;
		it->second = values.front().second + sigma * (it->second - values.front().second);
	}
}

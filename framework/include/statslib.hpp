#ifndef STATSLIB_HPP_
#define STATSLIB_HPP_
#include "armadillo"
#include "myexception.hpp"
#include "mt19937p.h"
#include <unistd.h>
using namespace arma;
const double PI = 3.141592;
const double factorPI = 1.0 / sqrt((long double) 2.0*PI);
static double seedflag = 0;

inline double evaluate_gaussian_prob(const colvec& value, const colvec& mean, const mat& cov);
inline double evaluate_gaussian_prob(const colvec& value, const colvec& mean, const mat& covariance_inv , const double covariance_det );

//code adapted from http://www.johndcook.com/normal_cdf_inverse.html
inline double rational_approximation( double t ) {
	const double c[] = {2.515517, 0.802853, 0.010328};
    const double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

inline double norminv( double prob, double mean, double variance ) {
//Returns x for a N(0,1) Gaussian having a probability of prob, then transform it to x' ~ N(mean, variance) having the same probability
	double x;

	if (prob < 0.5) {
		x = -rational_approximation( sqrt(-2.0*log(prob)) );
	} else {
		x = rational_approximation( sqrt(-2.0*log(1.0-prob)));
	}

	//Transform x to x'
	return x*sqrt(variance) + mean;
}

inline double evaluate_gaussian_prob(const colvec& value, const colvec& mean, const mat& cov) {
	return evaluate_gaussian_prob(value, mean, inv(cov), det(cov) );
}


inline double evaluate_gaussian_prob(const colvec& value, const colvec& mean, const mat& covariance_inv , const double covariance_det ) {
	colvec t = value - mean;
	int k = t.size();
	long double coef = -0.5 * as_scalar(trans(t) * covariance_inv * t);
	return exp( coef ) / sqrt( pow(2.0 * PI, (double) k ) * covariance_det );
}

inline double evaluate_gaussian_kernel(const colvec& value, const colvec& mean, const mat& covariance_inv ) {
	colvec t = value - mean;
	long double coef = -0.5 * as_scalar(trans(t) * covariance_inv * t);
	return exp( coef );
}

inline colvec generate_gaussian( const colvec& mean, const mat& covariance) {
	mat L = trans(chol(covariance)); //we want the lower cholesky decomposition
	return mean + L * randn<vec>(mean.n_rows);
	}

inline colvec generate_gaussian_cholesky( const colvec& mean, const mat& L) {
	return mean + L * randn<vec>(mean.n_rows);
	}
//L : lower matrix from cholesky decomposition of Covariance, 0 mean
inline colvec fast_generate_gaussian( const mat&L) {
	return L * randn<vec>(L.n_rows);
}


//See https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
inline bool isAPowerOf2( const int x) {
		return (x != 0) && ((x & (x - 1)) == 0);
}
//return 2^x
inline twoExp(const int x) {
	assert(x >= 0);
	if (x == 0) {
		return 1.0;
	}
	return 2.0*twoExp(x - 1);

}

//Only avaiable for proc in power of 2: 2,4,8,16,32, etc...
inline mat getParallelCovariance(const running_stat_vec<colvec> & samples, const & MPI_Intracomm com ) {
	const int N = samples.count();
	colvec T = double(N)*samples.mean();
	colvec diff;
	mat S = (double(N)-1.0)*samples.cov();
	int dim = T.size();
	colvec Tother = zeros<colvec>(dim);
	mat Sother = zeros<mat>(dim,dim);
	MPI_Status * status;
	const int size = Get_size(com);
	assert(isAPowerOf2(size));

	const int level = log(double(size))/log(2.0);
	const int id = Get_rank(com);

	for (int l = 0; l < level; l ++) {
		//This processor will be active
		if ((id % twoExp(l)) == 0) {
				//Process that receives
				if ((id % twoExp(l+1)) == 0) {
					MPI_Recv(Tother.memptr(), dim, MPI_DOUBLE, id + twoExp(l), 0, com, status);
					MPI_Recv(Sother.memptr(), dim*dim, MPI_DOUBLE, id + twoExp(l), 1, com, status);
					diff = T - Tother;
					S = S + Sother + diff * diff.t() / double(twoExp(l)*double(N));
					T = T + Tother;
					//Process that sends
				} else {
					MPI_Send(T.memptr(), dim, MPI_DOUBLE, id - twoExp(l) , 0, com);
					MPI_Send(S.memptr(), dim*dim, MPI_DOUBLE, id - twoExp(l), 1, com);
				}
		}
	}
	return S;
}


inline mat fast_generate_gaussian(unsigned int length, const mat&L) {
	//std::cout << "In gerenate noise" << std::endl;
	double u;
	double v;
	double s;
	double f;
	unsigned int n = L.n_rows;
	static struct mt19937p mt;
	if (seedflag == 0) {
		/* If using threads, make sure they are a different seed!
		long seed = abs((long int) (((time(0)*181)*((getpid()-83)*359))%104729)^omp_get_thread_num());
		*/
		long seed = abs((long int) (((time(0)*181)*((getpid()-83)*359))%104729));
		sgenrand(seed, &mt);
		seedflag = 1;
	}

	mat temp(L.n_rows, length);
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < length; j = j + 2) {
			//box muller polar form u and v are uniform from -1 to 1
			s = 0.0;
			while (s==0.0 || s >= 1.0 ) {
				u = 2.0 * genrand(&mt) - 1.0;
				v = 2.0 * genrand(&mt) - 1.0;
				s = u*u + v*v;
			}
			f = sqrt( -2.0*log(s) / s );

			temp.at(i,j) = u * f;
			if ((j+1) < length) {
				temp.at(i,j+1) = v * f;
			}
		}
	}
	return L * temp;
}

//Randomly permutes each column of matrix P with numbers from 1 to N
inline void permute_matrix(mat& P) {
	//initialize the elements
	unsigned int i,j;
	for (i=0; i < P.n_cols; i++) {
		for (j=0; j< P.n_rows; j++) {
			P(j,i) = i+1;
		}
	}
	//shuffle all the columns
	P = shuffle(P);
}

inline void fast_generate_gaussian(mat& noise, unsigned int length, const mat&L, const int LHS) {
	unsigned int i,j; //used for LHS
	//normal
	if (LHS == 0) {
		noise = L * randn<mat>(L.n_rows,length);
	} else {
		//latin hypercube sampling
		mat P(noise.n_rows, length);
		permute_matrix( P );

		for (j=0;j<noise.n_rows;j++) {
			for (i=0;i<length; i++) {
				noise(j,i) = norminv((P(j,i)-1.0 + (rand() / double(RAND_MAX))) / double(length) , 0.0 , L(j,j)*L(j,j) );
			}
		}
	}
}
//Unbiased weighted covariance. Each col represent one observation, each row represents a variable
inline mat getWeightedCovariance(const mat& X, const rowvec& weights) {
	const int n = X.n_cols;
	mat X_bar = repmat(X * trans(weights), 1, n);
	mat X_prime = X - X_bar;
	double ratio = 1.0 / (1.0 - as_scalar(sum(square(weights),1)));

    return  ratio * X_prime * diagmat(weights) * trans(X_prime);
}
/*
inline void evaluate_kde_at(rowvec& density, const mat& ensemble_x, const mat& samples, const unsigned int ensemble_size) {
//using the Eucledean distance between the samples and ensemble_x
rowvec distance = zeros<rowvec>(ensemble_size);
rowvec temp = zeros<rowvec>(ensemble_x.n_cols );
int i,j;
	for (i=0; i < ensemble_size; i++) {
		temp = samples.col(i) - ensemble_x.col(i);
		distance[i] = sqrt( dot(temp,temp) );

		for (j=0; j < ensemble_size; j++) {
			density
		}

		for (i=0; i < ensemble_size; i++) {
		x = (value - ensemble_x[i]) / bandwidth;
		likelihood += ensemble_w[i] * exp((long double) (-0.5 * x * x ));
		}
		likelihood = likelihood * factorPI / bandwidth ;


	}


}
*/

inline void evaluate_kde(rowvec& likelihood, const mat& ensemble_x, const rowvec& ensemble_w, const mat& samples, const unsigned int chunk) {
 	unsigned int d = ensemble_x.n_rows;
 	unsigned int ensemble_size = ensemble_x.n_cols;
    mat R;
    mat covariance = getWeightedCovariance(ensemble_x, ensemble_w);

	try
	{
		R = sqrt( diagmat(covariance) );
	}
	catch (...) {
		throw FilterUnstable();
	}
	double n_eff = 1.0 / as_scalar(sum(square(ensemble_w),1));
	mat H,xt,yt,Hinv;
	double detH;
   	H = pow(n_eff, -1.0/(double(d)+4.0))*R;
		//H = pow(double(ensemble_size), -1.0/(double(d)+4.0))*trans(R);
		//H = pow(4.0/(double(d)+2.0),1.0/(double(d)+4.0) )*pow(double(ensemble_size), -1.0/(double(d)+4.0))*R;
	detH = det(H);
	Hinv = inv(H);
	xt = Hinv*ensemble_x;
	yt = Hinv*samples;

	//mat xii(xt.n_rows, ensemble_size);
	//vec exp_sum_square(1, ensemble_size);
	double exp_sum_square;
	double temp_sum;
	const double constant = pow(2*PI,(-double(d)/2.0))*1.0/detH;
	for (unsigned int l = 0; l < ensemble_size; l++) {
		for (unsigned int j=0; j < chunk; j++) {
			temp_sum = 0.0;
			for (unsigned int s= 0; s < xt.n_rows; s++) {
				temp_sum = temp_sum + (xt.at(s,l) - yt.at(s,j))*(xt.at(s,l) - yt.at(s,j));
			}
			exp_sum_square = ensemble_w[l] * constant * exp( -0.5 * temp_sum );
			likelihood[j] = likelihood[j] + exp_sum_square;
		}
			//xii = repmat( xt.unsafe_col(l), 1, ensemble_size ) - yt;
			//likelihood = likelihood + pow(2*PI,(-double(d)/2.0))*(1.0/double(ensemble_size))*(1.0/detH)*exp(-0.5 * sum( square(xii) ));
			//likelihood = likelihood + ensemble_w[l]*pow(2*PI,(-double(d)/2.0))*1.0/detH*exp(-0.5 * sum( square(xii) ));
	}
}


inline void evaluate_kde(rowvec& likelihood, const mat& ensemble_x, const mat& samples, const unsigned int chunk) {
 	unsigned int d = ensemble_x.n_rows;
 	unsigned int ensemble_size = ensemble_x.n_cols;
    mat R;
    mat covariance = cov(trans(ensemble_x) );
	//Get the covariance of the ensemble
	///*
	try
	{
		R = sqrt( diagmat(covariance) );
	}
	catch (...) {
		//mat temp = trans(ensemble_x);
		//temp.print("Ensemble:");
		//temp.save("ensemble_negative_cov.dat", raw_ascii);
		//D.print("Covariance matrix");
		//ensemble_x.print("Ensemble x is ");
		std::cout << "Impossible to take KDE in proposal" << std::endl;
		throw FilterUnstable();
	}

	mat H,xt,yt,Hinv;
	double detH;
 	H = pow(double(ensemble_size), -1.0/(double(d)+4.0))*R;
		//H = pow(4.0/(double(d)+2.0),1.0/(double(d)+4.0) )*pow(double(ensemble_size), -1.0/(double(d)+4.0))*R;
	detH = det(H);
	Hinv = inv(H);
	xt = Hinv*ensemble_x;
	yt = Hinv*samples;
	//mat xii(xt.n_rows, ensemble_size);
	//vec exp_sum_square(1, ensemble_size);
	double exp_sum_square;
	double temp_sum;
	const double constant = 1.0/double(ensemble_size)*pow(2*PI,(-double(d)/2.0))*1.0/detH;
	//for each particle we want to construct the kde with
	for (unsigned int l = 0; l < ensemble_size; l++) {
		//for each particle delegated to that core (where we want to evaluate the density)
		for (unsigned int j=0; j < chunk; j++) {
			temp_sum = 0.0;
			for (unsigned int s= 0; s < xt.n_rows; s++) {
				//xii.unsafe_col[j] = xt.unsafe_col(l) - yt.unsafe_col[j];
				temp_sum = temp_sum + (xt.at(s,l) - yt.at(s,j))*(xt.at(s,l) - yt.at(s,j));
			}
			//exp_sum_square[j] = constant * exp( -0.5 * temp_sum );
			exp_sum_square = constant * exp( -0.5 * temp_sum );
			likelihood[j] = likelihood[j] + exp_sum_square;
		}
		//xii = repmat( xt.unsafe_col(l), 1, ensemble_size ) - yt;
		//likelihood = likelihood + pow(2*PI,(-double(d)/2.0))*(1.0/double(ensemble_size))*(1.0/detH)*exp(-0.5 * sum( square(xii) ));
		//likelihood = likelihood + ensemble_w[l]*pow(2*PI,(-double(d)/2.0))*1.0/detH*exp(-0.5 * sum( square(xii) ));
	}
  /*
    mat H = pow(double(ensemble_size), -1.0/(double(d)+4.0))*trans(R);
	double detH = det(H);
    mat Hinv = inv(H);


	mat xt = Hinv*ensemble_x;
	mat yt = Hinv*samples;
	mat xii;
    //likelihood.clear();
	for (int l = 0; l < ensemble_size; l++) {
    	xii = repmat( xt.col(l), 1, ensemble_size ) - yt;
    	//likelihood = likelihood + pow(2*PI,(-double(d)/2.0))*(1.0/double(ensemble_size))*(1.0/detH)*exp(-0.5 * sum( square(xii) ));
    	likelihood = likelihood + 1.0/double(ensemble_size)*pow(2*PI,(-double(d)/2.0))*1.0/detH*exp(-0.5 * sum( square(xii) ));
	}
	*/
}

//likelihood should be set to all zeros beforehand
inline double evaluate_density_at(const mat& ensemble, const rowvec& ensemble_w, const colvec& point) {
 	int d = ensemble.n_rows;
 	const int ensemble_size = ensemble.n_cols;
    mat R;
    mat covariance = getWeightedCovariance(ensemble, ensemble_w);



	try
	{
		R = sqrt( diagmat(covariance) );
		//covariance.print("Cov:");
		//R = sqrt( covariance.at(0,0));
	}
	catch (...) {
		//mat temp = trans(ensemble_x);
		//temp.print("Ensemble:");
		//temp.save("ensemble_negative_cov.dat", raw_ascii);
		//D.print("Covariance matrix");
		//std::cout << "Impossible to take KDE in prior" << std::endl;
		throw FilterUnstable();
	}
	mat H,xt,yt,Hinv;
	double detH;


    H = pow(4.0/(double(d)+2.0),1.0/(double(d)+4.0) )*pow(double(ensemble_size), -1.0/(double(d)+4.0))*R;
    //mat H = pow(double(ensemble_size), -1.0/(double(d)+4.0))*trans(R);
	detH = det(H);
    Hinv = inv(H);


	xt = Hinv*ensemble;
	yt = Hinv*point;
	mat xii;
	double likelihood = 0.0;
    //likelihood.clear();
	for (int l = 0; l < ensemble_size; l++) {
    	xii = xt.col(l) - yt;
    	//likelihood = likelihood + pow(2*PI,(-double(d)/2.0))*(1.0/double(ensemble_size))*(1.0/detH)*exp(-0.5 * sum( square(xii) ));
    	likelihood = likelihood + ensemble_w[l] * 1.0/double(ensemble_size)*pow(2*PI,(-double(d)/2.0))*1.0/detH*as_scalar(exp(-0.5 * sum( square(xii) )));
	}
	return likelihood;
}
#endif

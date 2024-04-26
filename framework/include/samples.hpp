#ifndef SAMPLES_HPP_
#define SAMPLES_HPP_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <mpi.h>
#include <vector>
#pragma GCC diagnostic pop
#include <assert.h>     /* assert */
//const double PI = 3.14159265359;

using namespace arma;

class Samples {
protected:
	colvec mean;
	mat covariance;
	mat crosscov;
	mat samples;
	int size;
	int currentIndex;
	int dim;
	int effectiveSize;
	bool dirty_mean; //Flag variable that skips the evaluation if the samples haven't been updated
	bool dirty_cov;
	std::vector<int> shuffledIndices;
public:
	Samples();
	Samples( int _dim,int _size) ;
	Samples( colvec _mean, mat _cov, int _size);
	mat getAutocorrelationFunction( int maxlag);
	Samples( mat _samples );
//	Samples( Gaussian* gpdf, int _size);
	colvec& getMean();
	mat& getCovariance();
	mat& getCrossCovariance( Samples& b );
	//mat getCovariance(const mat& _samples);
	colvec getSampleAt(int index);
	int getSize();
	int getDim();
	void setSampleAt(int index, colvec _sample);
	mat getSamples();
	colvec drawWithoutReplacement();
};
//
class SamplesMPI : public Samples {
public:
	SamplesMPI();
	SamplesMPI(int _dim, int _size, MPI::Intracomm _com);
	SamplesMPI(colvec _mean, mat _cov, int _size, MPI::Intracomm _com);
	SamplesMPI(mat _samples, MPI::Intracomm _com);
	colvec getSampleAt(int index);
	void setSampleAt(int index, colvec _sample);
	mat& getCovarianceAtRoot();
	mat& getCovariance();
	colvec& getMeanAtRoot();
	colvec& getMean();
	mat getSamplesAtRoot();
	mat getAutocorrelationFunction( int maxlag );
	mat& getCrossCovarianceAtRoot( SamplesMPI& b );
	void redistribute(std::vector<unsigned int>& localIndex);
	int getFirst() { return first;};
	int getLast() { return last;};
protected:
	/*MPI related variables BEGIN*/
	int globalSize;
	colvec globalMean;
	mat globalCovariance;
	mat crosscov;
	int nproc;
	int id;
	int first; 	//Index of the first element belonging to the current process
	int last;	//Index of the last element belonging to the current process
	MPI::Intracomm com;	//Communicator
	/*MPI related variables END*/
};

/*
* Sub class of samples for weighted samples
*/

class WeightedSamples : public Samples {
	public:
		WeightedSamples();
		WeightedSamples( colvec _mean, mat _cov, int _size);
		WeightedSamples( mat _samples );
//		WeightedSamples( Gaussian* gpdf, int _size);
		colvec getMean();
		mat getCovariance();
		void setWeights(rowvec _weights);
		double getWeightAt(int index);
		double getEffectiveSize();
		void setWeightAt(int index, long double _weight);
		void normalizeWeightAt(int index, long double factor);
		void systematic_resampling();
		colvec evaluate_kde(const mat& _samples, bool same_weights);
	private:
		rowvec weights;

};


class WeightedSamplesMPI : public SamplesMPI {
	public:
		WeightedSamplesMPI();
		WeightedSamplesMPI(int _dim, int _size, MPI::Intracomm _com);
		WeightedSamplesMPI(colvec _mean, mat _cov, int _size, MPI::Intracomm _com);
		WeightedSamplesMPI( mat _samples, MPI::Intracomm _com);
		mat& getCovarianceAtRoot();
		mat& getCovariance();
		colvec& getMeanAtRoot();
		colvec& getModeAtRoot();
		colvec& getMean();
		//void setWeights(rowvec _weights);
		double getWeightAt(int index);
		double getEffectiveSize();
		void setWeightAt(int index, long double _weight);
		void normalizeWeightAt(int index, long double factor);
		std::vector<unsigned int> systematic_resampling();
		std::vector<unsigned int> systematic_resampling_index();
		mat& getCrossCovarianceAtRoot( WeightedSamplesMPI& b ); //Use the weights of this
		colvec evaluate_kde( const mat & candidatePoints, bool use_weights);
		double getCovFactorAtRoot();
		double getCovFactor();
		void shareSamples();
	private:
		mat allSamples;
		mat allWeights;
		rowvec weights;
};




#endif

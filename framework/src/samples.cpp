#include "samples.hpp"

/******************************************************
*  Samples
******************************************************/

Samples::Samples() {
	dirty_mean = true;
	dirty_cov = true;
	currentIndex = 0;
}

Samples::Samples(int _dim, int _size) : Samples() {
	size = _size;
	dim = _dim;
	samples = zeros<mat>(dim, size);
	covariance = zeros<mat>(dim,dim);

	//std::vector<int> shuffledIndices(_size);
	shuffledIndices.reserve(_size);
	for (int s = 0; s < size; s ++) {
		shuffledIndices.push_back(s);
	}
	//std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);//Fill with 0 to ... length -1
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::mt19937{std::random_device{}()});
}

Samples::Samples( colvec _mean, mat _cov, int _size) : Samples() {
	size = _size;
	dim = _mean.n_rows;
	//create the initial ensemble
	mat L = trans(chol(_cov)); //we want the lower cholesky decomposition
	mean = _mean;
	covariance = _cov;
	this->samples = repmat(_mean, 1, size) + L * randn<mat>(dim, size);

	//std::vector<int> shuffledIndices(size);
	shuffledIndices.reserve(size);

	for (int s = 0; s < size; s ++) {
		shuffledIndices.push_back(s);
	}

	//std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);//Fill with 0 to ... length -1
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::mt19937{std::random_device{}()});
}

Samples::Samples( mat _samples ) :Samples() {
	size = _samples.n_cols;
	dim = _samples.n_rows;
	covariance = zeros<mat>(dim,dim);
	samples = _samples;

	//std::vector<int> shuffledIndices(size);
	shuffledIndices.reserve(size);
	for (int s = 0; s < size; s ++) {
		shuffledIndices.push_back(s);
	}
	//std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);//Fill with 0 to ... length -1
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::mt19937{std::random_device{}()});
}
/*
Samples::Samples(Gaussian* gpdf, int _size) {
	dirty_mean = true;
	dirty_cov = true;
	size = _size;
	mean = gpdf->getMean();
	covariance = gpdf->getCovariance();
	dim = mean.n_rows;
	//create the initial ensemble
	mat L = trans(chol(covariance)); //we want the lower cholesky decomposition
	mat temp = repmat(mean, 1, size);
	this->samples = repmat(mean, 1, size) + L * randn<mat>(dim, size);
	//this->weights = rowvec( size );
	//this->weights.fill( 1.0 / (double) size );
}
*/
colvec& Samples::getMean() {
	if (dirty_mean) {
		mean = sum(samples,1) / double(size);
		dirty_mean = false;
	}
	return mean;
}

mat Samples::getAutocorrelationFunction( int maxlag ) {
	colvec xbar = getMean();
	mat acf = mat(dim+1,maxlag); //Contains the autocorrelation for each dimension of the chain. Last row corresponds to the lag
	double temp, tempCov;
	int i,j,k;
	for (k = 0; k < maxlag; k++) { //For each lag
		for (j = 0; j < dim; j++) { //For each dimension of the MCMC chain
			temp = 0.0;
			tempCov = 0.0;
			for (i = 0; i < size - k; i ++) { //For each sample
				temp += (samples(j,i) - xbar(j))*(samples(j,i+k) - xbar(j));
			}
			//Get the denominator
			for (i = 0; i < size ; i ++) {
				tempCov += (samples(j,i) - xbar(j))*(samples(j,i) - xbar(j));
			}
			//Store the result
			acf(j,k) = temp / tempCov;
		}
		acf(dim,k) = int(k); //Last row contains lag k
	}
	return acf;
}

int Samples::getDim() {
	return dim;
}

mat& Samples::getCovariance() {
	if (dirty_cov) {
		mat X_bar = repmat( getMean() , 1, size);
		mat X_prime = samples - X_bar;
		double ratio = 1.0 / (double(size) - 1.0) ; /// (1.0 - as_scalar(sum(square(weights),1)));
		covariance = ratio * X_prime * trans(X_prime);
		dirty_cov = false;
	}
	return covariance;
}

colvec Samples::getSampleAt(int index) {
	return samples.unsafe_col(index);
}

colvec Samples::drawWithoutReplacement() {
	currentIndex ++;
	return getSampleAt(shuffledIndices[currentIndex-1]);
}


int Samples::getSize() {
	return size;
}

void Samples::setSampleAt(int index, colvec _sample) {
	dirty_mean = true;
	dirty_cov = true;
	samples.unsafe_col(index) = _sample;
}

mat Samples::getSamples() {
	return samples;
}

mat& Samples::getCrossCovariance( Samples& b) {
		if (dirty_mean) {
			getMean();
		}
		colvec xprime,dprime;
		crosscov = zeros<mat>(dim, b.getDim());
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			dprime = b.samples.unsafe_col(i) - b.getMean();
			crosscov += xprime*trans(dprime);
		}
		crosscov *=  1.0 / (double(size) - 1.0);
		return crosscov;
	}


/*
colvec Samples::evaluate_kde(const mat& _samples, bool use_weights) {
	//int d = ensemble_x.n_rows;
 	//int ensemble_size = ensemble_x.n_cols;

    mat R = sqrt( diagmat(getCovariance() ) );
    colvec likelihood = zeros<colvec>(getSize() );
	try
	{
		R = sqrt( diagmat(covariance) );
	}
	catch (...) {
		std::cout << "Error in KDE" << std::endl;
	}

	double n_eff;
	if (use_weights) {
		n_eff = 1.0 / as_scalar(sum(square(weights),1));
	} else {
		n_eff = double( getSize() );
	}

	mat H,xt,yt,Hinv;
	double detH, exp_sum_square, temp_sum;
   	H = pow(n_eff, -1.0/(double( getDim() )+4.0))*R;
	double w = 1.0/double(getSize());
	detH = det(H);
	Hinv = inv(H);

	xt = Hinv*samples;
	yt = Hinv*_samples;

	const double constant = pow(2.0*3.14159265359,(-double(getDim() )/2.0))*1.0/detH;
	for (int l = 0; l < getSize(); l++) {
		for (int j=0; j < getSize(); j++) {
			temp_sum = 0.0;
			for (int s= 0; s < xt.n_rows; s++) {
				temp_sum += (xt.at(s,l) - yt.at(s,j))*(xt.at(s,l) - yt.at(s,j));
			}
			if (use_weights) {
				w = getWeightAt(l);
			}
			exp_sum_square = w * constant * exp( -0.5 * temp_sum );
			likelihood[j] += exp_sum_square;
		}
	}
	return likelihood;
}
*/

/* See http://prod.sandia.gov/techlib/access-control.cgi/1998/980210.pdf
	A User's Guide to LHS, Gegory D. Wyss and Kelly H. Jorgensen
*/
/* So far only support Gaussian distribution */
//LatinHypercubeSamples::LatinHypercubeSamples( colvec _mean, mat _cov, int _size, int divisions) {
//}


/******************************************************
*  Samples_MPI
******************************************************/


SamplesMPI::SamplesMPI() : Samples() {}

SamplesMPI::SamplesMPI(colvec _mean, mat _cov, int _size, MPI::Intracomm _com) {
	com = _com;	//Communicator
	nproc = com.Get_size();
	id = com.Get_rank();

	assert((_size % nproc) == 0); //Each process have the same amount of particles

	size = _size / nproc; //number of particles for each process
	globalSize = _size;
	first = id*size; 			//Index of the first element belonging to the current process
	last = first + size - 1;		//Index of the last element belonging to the current process

	dirty_mean = true;
	dirty_cov = true;
	dim = _mean.n_rows;
	//create the initial ensemble
	mat L = trans(chol(_cov)); //we want the lower cholesky decomposition
	mean = _mean;
	covariance = _cov;
	this->samples = repmat(_mean, 1, size) + L * randn<mat>(dim, size); //each process get a set of particles
/*
	if (id == 0) {
		samples.save("Samples0.dat", raw_ascii);
	} else {
		samples.save("Samples1.dat", raw_ascii);
	}
	*/
}

SamplesMPI::SamplesMPI(int _dim, int _size, MPI::Intracomm _com) {
	com = _com;	//Communicator
	nproc = com.Get_size();
	id = com.Get_rank();

	assert((_size % nproc) == 0); //Each process have the same amount of particles

	size = _size / nproc; //number of particles for each process
	globalSize = _size;
	first = id*size; 			//Index of the first element belonging to the current process
	last = first + size - 1;		//Index of the last element belonging to the current process

	dirty_mean = true;
	dirty_cov = true;
	dim = _dim;
	//create the initial ensemble
	this->samples = zeros<mat>(dim,size);
	//Initialize covariance
	covariance = zeros<mat>(dim,dim);
}

SamplesMPI::SamplesMPI(mat _samples, MPI::Intracomm _com) {
	com = _com;	//Communicator
	nproc = com.Get_size();
	id = com.Get_rank();
	globalSize = _samples.n_cols;
	assert((globalSize % nproc) == 0); //Each process have the same amount of particles

	size = globalSize / nproc;
	first = id*size; 			//Index of the first element belonging to the current process
	last = first + size - 1;		//Index of the last element belonging to the current process
	//std::cout << " In Samples MPI constructor : " << id << " first :" << first << " last :" << last << std::endl;
	dim = _samples.n_rows;
	samples = zeros<mat>(dim , size);

	for (int i = first; i <= last; i++) {
		//std::cout << "(" << id << ") and i " << i << " goes into " << i-first << std::endl;
		samples.unsafe_col(i-first) = _samples.unsafe_col(i);
	}

	dirty_mean = true;
	dirty_cov = true;
	covariance = zeros<mat>(dim,dim);

}

//Parallel version of getMean.
colvec& SamplesMPI::getMean() {
	//All process should have the flags the same value
	if (dirty_mean) {
		mean = sum( samples.cols(0,size-1),1 ) / double(globalSize);
		dirty_mean = false;
		//ALL REDUCE OPERATION (All the process will have the mean)
		com.Allreduce(MPI::IN_PLACE, mean.memptr(), dim, MPI::DOUBLE, MPI::SUM);
		globalMean = mean;
	}
	return mean;
}

colvec& SamplesMPI::getMeanAtRoot() {
	//All process should have the flags the same value
	if (dirty_mean) {
		mean = sum( samples.cols(0,size-1),1 ) / double(globalSize);
		dirty_mean = false;
		//ALL REDUCE OPERATION (All the process will have the mean)
		if (id == 0) {
			com.Reduce(MPI::IN_PLACE, mean.memptr(), dim, MPI::DOUBLE, MPI::SUM, 0);
		} else {
			com.Reduce(mean.memptr(), mean.memptr(), dim, MPI::DOUBLE, MPI::SUM, 0);
		}
		globalMean = mean;
	}
	return mean;
}

//Redistribute samples based on the local index vector
//Each procs has its own index vector
void SamplesMPI::redistribute(std::vector<unsigned int>& localIndex) {
	//Not the most efficient way, but everything is collected at the root level
	//Everything is swapped accordingly, then everything is scattered
	mat allSamples = getSamplesAtRoot();
	mat oldSamples = allSamples;
	std::vector<unsigned int> globalIndex;
	globalIndex.reserve(globalSize);

	//Gather all indeces
	MPI_Gather(&localIndex.front(), size, MPI_UNSIGNED, &globalIndex.front(), size, MPI_UNSIGNED, 0, com);
	if (id == 0) {
		for (int i = 0; i < globalSize; i++) {
			allSamples.unsafe_col(i) = oldSamples.unsafe_col( globalIndex[i] );
		}
	}

	MPI_Scatter(allSamples.memptr(), dim*size, MPI_DOUBLE,  samples.memptr(), dim*size, MPI_DOUBLE, 0, com );
}

mat& SamplesMPI::getCovariance() {
	if (dirty_cov) {
		//If the mean was not previously computed, compute it
		if (dirty_mean) {
			getMean();
		}
		colvec xprime;
		covariance.zeros(); //reset covariance
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			covariance += xprime*trans(xprime);
		}
		covariance *=  1.0 / (double(globalSize) - 1.0);
		com.Allreduce(MPI::IN_PLACE, covariance.memptr(), dim*dim, MPI::DOUBLE, MPI::SUM);
		globalCovariance = covariance;
		dirty_cov = false;
	}
	return covariance;
}

mat& SamplesMPI::getCovarianceAtRoot() {
	if (dirty_cov) {
		//If the mean was not previously computed, compute it
		if (dirty_mean) {
			getMean();
		}
		colvec xprime;

		covariance.zeros(); //reset covariance
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			covariance += xprime*trans(xprime);
		}
		covariance *=  1.0 / (double(globalSize) - 1.0);

		if (id == 0) {
			com.Reduce(MPI::IN_PLACE, covariance.memptr(), dim*dim, MPI::DOUBLE, MPI::SUM,0);
		} else {
			com.Reduce(covariance.memptr(), covariance.memptr(), dim*dim, MPI::DOUBLE, MPI::SUM,0);
		}
		globalCovariance = covariance;
		dirty_cov = false;
	}
	return covariance;
}

mat& SamplesMPI::getCrossCovarianceAtRoot( SamplesMPI& b) {
		if (dirty_mean) {
			getMean();
		}
		colvec xprime,dprime;
		crosscov = zeros<mat>(dim, b.getDim());
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			dprime = b.samples.unsafe_col(i) - b.getMean();
			crosscov += xprime*trans(dprime);
		}
		crosscov *=  1.0 / (double(globalSize) - 1.0);

		if (id == 0) {
			com.Reduce(MPI::IN_PLACE, crosscov.memptr(), dim*b.getDim(), MPI::DOUBLE, MPI::SUM,0);
		} else {
			com.Reduce(crosscov.memptr(), crosscov.memptr(), dim*b.getDim(), MPI::DOUBLE, MPI::SUM,0);
		}

		return crosscov;
	}

//index goes from first to last (example 1000 to 1200)
//but I need to shift it back from 0 to local size
colvec SamplesMPI::getSampleAt(int index) {
	//make sure that the index number is between first and last

	//std::cout << first << " - " << index << " - " << last << std::endl;
	//std::cout << "(getSampleAt) Sample i " << index << " becomes " << index-first << std::endl;

	assert(index >= first && index <= last);
	//shift back index
	return samples.unsafe_col(index - first);
}


void SamplesMPI::setSampleAt(int index, colvec _sample) {
	assert(index >= first && index <= last);
	dirty_mean = true;
	dirty_cov = true;
	//std::cout << "(getSampleAt) Sample i " << index << " becomes " << index-first << std::endl;
	samples.unsafe_col(index - first) = _sample;
}

mat SamplesMPI::getAutocorrelationFunction( int maxlag ) {
	colvec xbar = getMeanAtRoot();

	mat acf = mat(dim+1,maxlag); //Contains the autocorrelation for each dimension of the chain. Last row corresponds to the lag
	double temp, tempCov;
	int i,j,k;

	//Gather all the samples at the root
	mat globalSamples(dim, globalSize);
	com.Gather(samples.memptr(), dim*size, MPI::DOUBLE, globalSamples.memptr(), dim*size, MPI::DOUBLE, 0 );

	if (id == 0) {
	for (k = 0; k < maxlag; k++) { //For each lag
		for (j = 0; j < dim; j++) { //For each dimension of the MCMC chain
			temp = 0.0;
			tempCov = 0.0;
			for (i = 0; i < (globalSize - k); i ++) { //For each sample
				temp += (globalSamples(j,i) - xbar(j))*(globalSamples(j,i+k) - xbar(j));
			}
			//Get the denominator
			for (i = 0; i < globalSize ; i ++) {
				tempCov += (globalSamples(j,i) - xbar(j))*(globalSamples(j,i) - xbar(j));
			}
			//Store the result
			acf(j,k) = temp / tempCov;
		}
		acf(dim,k) = int(k); //Last row contains lag k
	}
}
return acf;
}

	mat SamplesMPI::getSamplesAtRoot() {
		//std::cout << "In get Samples at Root" << std::endl;
		//std::cout << "Dim is " << dim << " and globalSize is " << globalSize << std::endl;
		mat globalSamples(dim, globalSize);
		com.Gather(samples.memptr(), dim*size, MPI::DOUBLE, globalSamples.memptr(), dim*size, MPI::DOUBLE, 0 );
		return globalSamples;
	}

/******************************************************
*  Weighted Samples
******************************************************/


WeightedSamples::WeightedSamples() : Samples::Samples() {
	dirty_mean = true;
	dirty_cov = true;
}


WeightedSamples::WeightedSamples( colvec _mean, mat _cov, int _size) : Samples(_mean, _cov, _size) {
	dirty_mean = true;
	dirty_cov = true;
	this->weights = rowvec( size );
	this->weights.fill( 1.0 / static_cast<double>(size) );
}

WeightedSamples::WeightedSamples( mat _samples ) : Samples(_samples) {
	dirty_mean = true;
	dirty_cov = true;
	this->weights = rowvec( size );
	this->weights.fill( 1.0 / static_cast<double>(size) );
}

colvec WeightedSamples::getMean() {
	if (dirty_mean) {
		mean = samples * trans(weights);
		dirty_mean = false;
	}
	return mean;
}


mat WeightedSamples::getCovariance() {
	if (dirty_cov) {
		mat X_bar = repmat( getMean() , 1, size);
		mat X_prime = samples - X_bar;
		double ratio = 1.0 / (1.0 - as_scalar(sum(square(weights),1)));
		covariance = ratio * X_prime * diagmat(weights) * trans(X_prime);
		dirty_cov = false;
	}
	return covariance;
}

double WeightedSamples::getEffectiveSize() {
	effectiveSize = 1.0 / sum(square(weights));
	return effectiveSize;
}

void WeightedSamples::setWeightAt(int index, long double _weight) {
	dirty_mean = true;
	dirty_cov = true;
	weights[index] = _weight;
}


//Serial version. This version disregard off-diagonal terms in covariance
colvec WeightedSamples::evaluate_kde(const mat& _samples, bool use_weights) {
	//int d = ensemble_x.n_rows;
 	//int ensemble_size = ensemble_x.n_cols;

 mat R = sqrt( diagmat(getCovariance() ) );
 colvec likelihood = zeros<colvec>(_samples.n_cols );
	try
	{
		R = sqrt( diagmat(covariance) );
	}
	catch (...) {
		std::cout << "Error in KDE" << std::endl;
	}

	double n_eff;
	if (use_weights) {
		n_eff = 1.0 / as_scalar(sum(square(weights),1));
	} else {
		n_eff = double( getSize() );
	}

	mat H,xt,yt,Hinv;
	double detH, exp_sum_square, temp_sum;
  H = pow(n_eff, -1.0/(double( getDim() )+4.0))*R;
	double w = 1.0/double(getSize());

	detH = det(H);
	Hinv = inv(H);

	//xt = Hinv*samples;
	//yt = Hinv*_samples;
	xt = samples;
	yt = _samples;

	const double constant = pow(2.0*3.14159265359,(-double(getDim() )/2.0))*1.0/sqrt(detH);

for (int j = 0; j < yt.n_cols; j++) {
	for (int l = 0; l < xt.n_cols; l++) {
			temp_sum = 0.0;
			for (unsigned int s= 0; s < xt.n_rows; s++) {
				temp_sum += Hinv(s,s)*(xt.at(s,l) - yt.at(s,j))*(xt.at(s,l) - yt.at(s,j));
			}
			if (use_weights) {
				w = getWeightAt(l);
			}
			exp_sum_square = exp( -0.5 * temp_sum );
			likelihood[j] += exp_sum_square;
		}
		likelihood[j]*= w*constant;
	}
	return likelihood;
}

void WeightedSamples::setWeights(rowvec _weights) {
	dirty_mean = true;
	dirty_cov = true;
	weights = _weights;
}

double WeightedSamples::getWeightAt(int index) {
	return weights[index];
}


void WeightedSamples::normalizeWeightAt(int index, long double factor) {
	dirty_mean = true;
	dirty_cov = true;
	weights[index] = std::exp(log(weights[index])-log(factor));
	//weights[index] /= factor;
}

void WeightedSamples::systematic_resampling() {
	dirty_mean = true;
	dirty_cov = true;
	mat oldsamples = samples;
	double ns = double(size);
	rowvec Q = cumsum(weights,1);
	colvec T = zeros<colvec>(size+1);
	int s;
	double u = randu() / ns;
	for (s = 0; s < size; s ++) {
		T[s] = double(s) * 1.0/ns + u;
	}
	T[size] = 1.0;

	int i=0;
	int j=0;
	while ( i<size && j < size ) {
    	if (T[i]<Q[j]) {
  //  		std::cout << " i , j of N " << i << " , " << j << " of " << size << std::endl;
        	samples.unsafe_col(i) = oldsamples.unsafe_col(j);
        	i++;
    	} else {
        	j++;
    	}
	}

	//Reset all the weights
	weights.fill( 1.0 / static_cast<double>(size) );
}


WeightedSamplesMPI::WeightedSamplesMPI(int _dim, int _size, MPI::Intracomm _com) : SamplesMPI(_dim, _size, _com) {
	this->weights = zeros<rowvec>(size);
	weights.fill(1.0/double(globalSize));

	allSamples = zeros<mat>( dim, globalSize );
	allWeights = zeros<rowvec>( globalSize );

}


WeightedSamplesMPI::WeightedSamplesMPI() : SamplesMPI() {}

WeightedSamplesMPI::WeightedSamplesMPI(colvec _mean, mat _cov, int _size, MPI::Intracomm _com) :SamplesMPI(_mean, _cov, _size, _com) {
	this->weights = zeros<rowvec>(size);
	weights.fill(1.0/double(globalSize));

	allSamples = zeros<mat>( dim, globalSize );
	allWeights = zeros<rowvec>( globalSize );

}

WeightedSamplesMPI::WeightedSamplesMPI(mat _samples, MPI::Intracomm _com) : SamplesMPI(_samples, _com) {
	this->weights = zeros<rowvec>(size);
	weights.fill(1.0/double(globalSize));

	allSamples = zeros<mat>( dim, globalSize );
	allWeights = zeros<rowvec>( globalSize );

}

//Parallel version of getMean.
colvec& WeightedSamplesMPI::getMean() {
	//All process should have the flags the same value
	if (dirty_mean) {
		mean = samples.col(0) * weights[0];
		for (int i = 1; i < size; i++) {
			mean += samples.col(i) * weights[i];
		}
		dirty_mean = false;
		//ALL REDUCE OPERATION (All the process will have the mean)
		com.Allreduce(MPI::IN_PLACE, mean.memptr(), dim, MPI::DOUBLE, MPI::SUM);
		globalMean = mean;
	}
	return mean;
}

colvec& WeightedSamplesMPI::getMeanAtRoot() {
	//All process should have the flags the same value
	if (dirty_mean) {
		mean = samples.col(0) * weights[0];
		for (int i = 1; i < size; i++) {
			mean += samples.col(i) * weights[i];
		}
		dirty_mean = false;
		//ALL REDUCE OPERATION (All the process will have the mean)
		if (id == 0) {
			com.Reduce(MPI::IN_PLACE, mean.memptr(), dim, MPI::DOUBLE, MPI::SUM, 0);
		} else {
			com.Reduce(mean.memptr(), mean.memptr(), dim, MPI::DOUBLE, MPI::SUM, 0);
		}
		globalMean = mean;
	}
	return mean;
}
//Return the particle with the highest weight
colvec& WeightedSamplesMPI::getModeAtRoot() {
	struct key{
		double val;
		int rank;
	} mykey, maxkey;

	double maxW = 0.0;
	double index;
	colvec modeParticle;
	//Each process find it's maximum
	for (int i = 0; i < size; i++) {
		if (weights[i] > maxW) {
			maxW = weights[i];
			index = i;
		}
	}

	modeParticle = samples.unsafe_col(index);
	mykey.val = maxW;
	mykey.rank = id;

	//Compare at root
	MPI_Allreduce(&mykey, &maxkey , 1, MPI_DOUBLE_INT, MPI_MAXLOC, com);
	//Broadcast to other procs
	MPI_Bcast(modeParticle.memptr(), dim,  MPI_DOUBLE, maxkey.rank, com);
	return modeParticle;
}

mat& WeightedSamplesMPI::getCovariance() {
	if (dirty_cov) {
		//If the mean was not previously computed, compute it
		if (dirty_mean) {
			getMean();
		}

		colvec xprime;
		covariance.zeros(); //reset covariance
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			covariance += xprime*trans(xprime)*weights[i];
		}
		covariance *= getCovFactor();
		com.Allreduce(MPI::IN_PLACE, covariance.memptr(), dim*dim, MPI::DOUBLE, MPI::SUM);
		globalCovariance = covariance;
		dirty_cov = false;
	}
	return covariance;
}

mat& WeightedSamplesMPI::getCovarianceAtRoot() {
	if (dirty_cov) {
		//If the mean was not previously computed, compute it
		if (dirty_mean) {
			getMean();
		}
		colvec xprime;

		covariance.zeros(); //reset covariance
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			covariance += xprime*trans(xprime)*weights[i];
		}

		if (id == 0) {
			com.Reduce(MPI::IN_PLACE, covariance.memptr(), dim*dim, MPI::DOUBLE, MPI::SUM,0);
		} else {
			com.Reduce(covariance.memptr(), covariance.memptr(), dim*dim, MPI::DOUBLE, MPI::SUM,0);
		}
		covariance *= getCovFactorAtRoot();
		globalCovariance = covariance;
		dirty_cov = false;
	}
	return covariance;
}

mat& WeightedSamplesMPI::getCrossCovarianceAtRoot( WeightedSamplesMPI& b) {
		if (dirty_mean) {
			getMean();
		}
		colvec xprime,dprime;
		crosscov = zeros<mat>(dim, b.getDim());
		for (int i = 0; i < size; i++) {
			xprime = samples.unsafe_col(i) - mean;
			dprime = b.samples.unsafe_col(i) - b.getMean();
			crosscov += xprime*trans(dprime)*b.weights[i];
		}
		crosscov *= getCovFactorAtRoot();

		if (id == 0) {
			com.Reduce(MPI::IN_PLACE, crosscov.memptr(), dim*b.getDim(), MPI::DOUBLE, MPI::SUM,0);
		} else {
			com.Reduce(crosscov.memptr(), crosscov.memptr(), dim*b.getDim(), MPI::DOUBLE, MPI::SUM,0);
		}
		crosscov *= getCovFactorAtRoot();
		return crosscov;
}


//void WeightedSamplesMPI::setWeights(rowvec _weights);
double WeightedSamplesMPI::getWeightAt(int index) {
	assert(index >= first && index <= last);
	return weights[index - first];
}

double WeightedSamplesMPI::getEffectiveSize() {
	double squareWeight = 0.0;
	for (int i = 0; i < size; i++) {
		squareWeight += weights[i]*weights[i];
	}

 	com.Allreduce(MPI::IN_PLACE, &squareWeight, 1, MPI::DOUBLE, MPI::SUM);
	effectiveSize = 1.0 / squareWeight;
	return effectiveSize;
}

void WeightedSamplesMPI::setWeightAt(int index, long double _weight) {
	dirty_cov = true;
	dirty_mean = true;
	assert(index >= first && index <= last);
	weights[index - first] = _weight;
}

void WeightedSamplesMPI::normalizeWeightAt(int index, long double factor) {
		dirty_cov = true;
		dirty_mean = true;
		assert(index >= first && index <= last);
		weights[index - first] = std::exp(log(weights[index - first]) - log(factor) );
}

double WeightedSamplesMPI::getCovFactor() {
	double squareWeight = 0.0;
	for (int i = 0; i < size; i++) {
		squareWeight += weights[i]*weights[i];
	}

 	com.Allreduce(MPI::IN_PLACE, &squareWeight, 1, MPI::DOUBLE, MPI::SUM);

	return 1.0 / (1.0 - squareWeight);
}

double WeightedSamplesMPI::getCovFactorAtRoot() {
	double squareWeight = 0.0;
	for (int i = 0; i < size; i++) {
		squareWeight += weights[i]*weights[i];
	}

	if (id == 0) {
		com.Reduce(MPI::IN_PLACE, &squareWeight, 1, MPI::DOUBLE, MPI::SUM,0);
	} else {
		com.Reduce(&squareWeight, &squareWeight, 1, MPI::DOUBLE, MPI::SUM,0);
	}

	return 1.0 / (1.0 - squareWeight);
}

void WeightedSamplesMPI::shareSamples() {
	com.Allgather(samples.memptr(), dim*size , MPI::DOUBLE, allSamples.memptr(), dim*size, MPI::DOUBLE);
	com.Allgather(weights.memptr(), size , MPI::DOUBLE, allWeights.memptr(), size, MPI::DOUBLE);
}

//Current formulation is only valid if we only look at the diagonal element of the covariance
//Need to check the current formulation. Should compare with matlab
colvec WeightedSamplesMPI::evaluate_kde( const mat & candidatePoints, bool use_weights) {

	int n = candidatePoints.n_rows;
	//Gather all the points
	//Send samples to each other process


  mat R;
  colvec likelihood = zeros<colvec>( candidatePoints.n_cols ); //Local likelihood
	try
	{
		R = sqrt( diagmat(getCovariance() ) );
	}
	catch (...) {
		std::cout << "Error in KDE" << std::endl;
	}

	double n_eff;
	if (use_weights) {
		n_eff = getEffectiveSize();
	} else {
		n_eff = double( globalSize );
	}

	mat H,xt,yt,Hinv;
	double detH, exp_sum_square, temp_sum;
  H = pow(n_eff, -1.0/(double( getDim() )+4.0))*R;
	double w = 1.0/double(globalSize);

	detH = det(H);
	Hinv = inv(H);

	//if (dirty_mean or dirty_cov) {
	//shareSamples();
	//}

	//xt = Hinv * allSamples; //Complete xt (samples)
	//yt = Hinv * candidatePoints; //Partial yt  (sub samples)
	xt = allSamples;
	yt = candidatePoints;
	const double constant = pow(2.0*3.14159265359,(-double(getDim() )/2.0))*1.0/sqrt(detH);
	colvec diff;
		for (int j = 0; j < yt.n_cols; j++) {
			for (int l = 0; l < xt.n_cols; l++) {

			temp_sum = 0.0;
			//for (unsigned int s= 0; s < xt.n_rows; s++) {
			//	temp_sum += (xt.at(s,l) - yt.at(s,j))*(xt.at(s,l) - yt.at(s,j));
			//}
			//diff = xt.unsafe_col(l) - yt.unsafe_col(j);
			for (unsigned int s= 0; s < xt.n_rows; s++) {
				temp_sum += Hinv(s,s)*(xt.at(s,l) - yt.at(s,j))*(xt.at(s,l) - yt.at(s,j));
			}
			if (use_weights) {
				w = allWeights[l];
			}
			exp_sum_square = exp( -0.5 * temp_sum );
			//exp_sum_square = exp(-0.5 * as_scalar(trans(diff)*Hinv*diff ));
			likelihood[j] += exp_sum_square;
		}
		likelihood[j] *= w * constant;
	}
	return likelihood;
}



std::vector<unsigned int> WeightedSamplesMPI::systematic_resampling() {

	shareSamples();

	dirty_mean = true;
	dirty_cov = true;

	std::vector<unsigned int> localIndex = std::vector<unsigned int>(size);
	std::vector<unsigned int> globalIndex = std::vector<unsigned int>(globalSize);

	unsigned int i=0;
	unsigned int j=0;

	//Distributed samples
	if (id == 0) {
		double ns = double(globalSize);
		mat oldsamples = allSamples;
		//allWeights.print("Weights");
		rowvec Q = cumsum(allWeights,1);
		colvec T = zeros<colvec>(globalSize);
		int s;
		double u = randu() / ns;
		for (s = 0; s < globalSize; s ++) {
			T[s] = double(s) * 1.0/ns + u;
		}
		//To protect against overflow of j
		for (i = 0; i < globalSize;i ++) {
			while (Q[j] < T[i]) {
				j++;
			}
			globalIndex[i] = j;
			allSamples.unsafe_col(i) = oldsamples.unsafe_col(j);
		}
	}

	//Reset all the weights
	weights.fill( 1.0 / static_cast<double>(globalSize) );


	MPI_Scatter(&globalIndex.front(), size, MPI_UNSIGNED, &localIndex.front(), size, MPI_UNSIGNED, 0, com);
	MPI_Scatter(allSamples.memptr(), size*dim, MPI_DOUBLE, samples.memptr(), size*dim, MPI_DOUBLE, 0, com);
	return localIndex;
}

std::vector<unsigned int> WeightedSamplesMPI::systematic_resampling_index() {

	shareSamples();

	dirty_mean = true;
	dirty_cov = true;

	std::vector<unsigned int> localIndex = std::vector<unsigned int>(size);
	std::vector<unsigned int> globalIndex = std::vector<unsigned int>(globalSize);

	unsigned int i=0;
	unsigned int j=0;

	//Distributed samples
	if (id == 0) {
		double ns = double(globalSize);

		//allWeights.print("Weights");
		rowvec Q = cumsum(allWeights,1);
		colvec T = zeros<colvec>(globalSize);
		int s;
		double u = randu() / ns;
		for (s = 0; s < globalSize; s ++) {
			T[s] = double(s) * 1.0/ns + u;
		}
		//To protect against overflow of j
		for (i = 0; i < globalSize;i ++) {
			while (Q[j] < T[i]) {
				j++;
			}
			globalIndex[i] = j;
		}
	}
	MPI_Scatter(&globalIndex.front(), size, MPI_UNSIGNED, &localIndex.front(), size, MPI_UNSIGNED, 0, com);
	return localIndex;
}

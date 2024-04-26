#include "simulate.hpp"

//Get the normalized cross-correlation between two signals.
//Pads the signals with N -1 zeros on each sigde of signal
/*
*  Signal 1 and Signal 2 of lenght N
* For example if N = 4
* Lag = 0
* Signal 1  [0 0 0 1 1 1 1 0 0 0]
* Signal 2  [0 0 0 1 1 1 1 0 0 0]
*
* Lag = -1
* Signal 1  [0 0 0 1 1 1 1 0 0 0]       index used 0,1,2
* Signal 2  [0 0 1 1 1 1 0 0 0 0]       index used 1,2,3
*
* Lag = -3
* Signal 1  [0 0 0 1 1 1 1 0 0 0]       index used 0
* Signal 2  [1 1 1 1 0 0 0 0 0 0]       index use 3
*
* Lag = 2
* Signal 1  [0 0 0 1 1 1 1 0 0 0]   index used 2,3
* Signal 2  [0 0 0 0 0 1 1 1 1 0]   index used 0,1
*
* Negative lag -> not using the last part of the s1, and not using the beginning of s2
*/
double getCrossCorrelation(const rowvec & s1, const rowvec & s2, const double m1, const double sigma1, const double m2, const double sigma2, const int lag) {
  int N = s1.n_cols;
  double corr = 0.0;
  int iS1, iS2;
  for (int i = 0; i < N - abs(lag); i ++) {

    if (lag >= 0) {
      iS1 = i + lag;
      iS2 = i;
    } else {
      iS1 = i;
      iS2 = i + abs(lag);
    }

    corr += (s1[iS1] - m1)*(s2[iS2] - m2);
  }

  return corr/(double(N)*sigma1*sigma2);
}

int getShift(const mat & x, const mat & y, const int maxlag) {

  //Compute mean and variance
  rowvec xVector = x.row(0);
  rowvec yVector = y.row(0);

  //Get the mean of both signals
  double mx = as_scalar(mean(xVector));
  double my = as_scalar(mean(yVector));
  double sx = as_scalar(stddev(xVector));
  double sy = as_scalar(stddev(yVector));
  double maxCorr, corr;
  int maxIndex;
  maxCorr = 0.0;
  for (int i = -maxlag; i <= maxlag; i ++ ){
    corr = getCrossCorrelation( xVector, yVector, mx, sx, my, sy, i);
    //std::cout << "At lag = " << i << ", correlation is " << corr << std::endl;
    if (corr > maxCorr) {
      maxCorr = corr;
      maxIndex = i;
      //std::cout << "Max correlation found at " << i << " = " << maxCorr << std::endl;
    }
  }
  //std::cout << "**************Going with lag " << maxIndex << std::endl;
  return maxIndex;
}

mat getShiftedSignal(const mat & x, const mat & y, const int maxlag) {
  int lag = getShift(x,y,maxlag);
  mat temp = zeros<mat>(y.n_rows, y.n_cols);
  int n = y.n_cols;
  if (lag < 0) {
    //Negative lags, remove starting values and pads with 0 at the end
    for (int i = 0; i < n-abs(lag); i ++) {
      temp.unsafe_col(i) = y.unsafe_col(i+abs(lag));
    }
  }else if (lag > 0) {
    //Positive lags, remove ending values and pads with 0 at the beginning
    for (int i = 0; i < n-lag; i ++) {
      temp.unsafe_col(i+lag) = y.unsafe_col(i);
    }
  } else {
    return y;
  }
  return temp;
}

//Code used to simulate models using MCMC chains as input for parameters
void simulate( const unsigned N, const unsigned m, const unsigned qm, const colvec & ic, const mat & paramChain, const mat & ref, statespace & ss, const std::string name, bool shiftSignal, double delayedValue, const MPI_Comm& com ) {
    //Perform N Simulations of m iterations  using paramChain
    unsigned n,i,j,nlocal;
    int size,id,r,s, maxShiftL,maxShiftR, signalShift;
    colvec parameters, state;

    MPI_Comm_rank(com, &id);
    MPI_Comm_size(com, &size);

    //Get the size of the statespace
    n = ss.getStateVectorSize();
    mat path = zeros<mat>(n,m);
    mat mean = zeros<mat>(n,m);
    mat diagvar = zeros<mat>(n,m);
    nlocal = N / size;
    mat pathToWrite = zeros<mat>(n, m/10);

    //Create N/qm quantiles vector
    ///EXTREMELY MEMORY INTENSIVE
    /*
    std::vector<IQAgent> myagents_theta;
    std::vector<IQAgent> myagents_thetadot;
    std::vector<IQAgent> myagents_cm;
    for (j = 0; j < m; j ++) {
      if (j % qm == 0) {
        myagents_theta.push_back( IQAgent() );
        myagents_thetadot.push_back( IQAgent() );
        myagents_cm.push_back( IQAgent() );
      }
    }
    //Add the last agent
    myagents_theta.push_back( IQAgent() );
    myagents_thetadot.push_back( IQAgent() );
    myagents_cm.push_back( IQAgent() );
    */

    double max;
    for (i = 0; i < nlocal; i ++) {
      if (i % 100 == 0) {
        std::cout << "Iteration " << i << std::endl;
      }

      //Simulate the time series
      s = 0; //IQAgent index
    	ss.resetTime();
      path.col(0) = ic; //initial conditions set to zero
      mean.col(0) += path.col(0);
      diagvar.col(0) += square(path.col(0));
      r = randi(distr_param(0, paramChain.n_cols-1) );
      max = 0.0;
      parameters = paramChain.col( r );
      //parameters.print("Param=");
      for (j = 0; j < m-1; j ++) {
        path.col(j+1) = ss.evaluatef( path.col(j) , parameters );
        ss.timeIncrement();
      }

      if (shiftSignal) path = getShiftedSignal(ref, path, int(0.2*double(m)));


      for (j = 0; j < m/10; j ++) {
        pathToWrite.col(j) = path.col(j*10);
      }

      //Save shifted signal
      pathToWrite.save( "./data/" + name + "_" + std::to_string(i) + ".dat", raw_ascii);

      mean += path;
      diagvar += square(path);


      //Record quantiles
      /*
      for (j = 0; j < m-1; j ++) {
        if (j % qm == 0) {
          myagents_theta[s].add( path.at(0,j)  );
          myagents_thetadot[s].add( path.at(1,j)  );
          myagents_cm[s].add( path.at(2,j)  );
          s++;
        }
      }
      //Add last agent
      myagents_theta[s].add( path.at(0,j)  );
      myagents_thetadot[s].add( path.at(1,j)  );
      myagents_cm[s].add( path.at(2,j)  );
      */
  }

    //Reduce the mean and write it to file
    if (id == 0) {
      ////MPI_Reduce(MPI_IN_PLACE, mean.memptr(), n * m, MPI_DOUBLE, MPI_SUM, 0, com);
      ////MPI_Reduce(MPI_IN_PLACE, diagvar.memptr(), n * m, MPI_DOUBLE, MPI_SUM, 0, com);
    } else {
      ////MPI_Reduce(mean.memptr(), mean.memptr() , n * m, MPI_DOUBLE, MPI_SUM, 0, com);
      ////MPI_Reduce(diagvar.memptr(), diagvar.memptr() , n * m, MPI_DOUBLE, MPI_SUM, 0, com);
    }

    mat tmean, tdiagvar, tquantiles;
    ///mat quantiles_theta = zeros<mat>(3, myagents_theta.size() );
    ///mat quantiles_thetadot = zeros<mat>(3, myagents_thetadot.size() );
    ///mat quantiles_cm = zeros<mat>(3, myagents_cm.size() );

    ///if (id == 0) {
      mean = mean / double(N);
      //diagvar = (diagvar + (1.0 - 2.0*double(N))*square(mean))/ (double(N) - 1.0);
      diagvar = (diagvar - double(N)*square(mean))/(double(N) - 1.0);

      mean.save( name + "_mean.dat", raw_ascii);
      diagvar.save(name + "_diag_var.dat", raw_ascii);
      //Record quantiles
      /*
      for (s = 0; s < myagents_theta.size(); s++) {
        quantiles_theta.at(0,s) = myagents_theta[s].report(0.05);
        quantiles_theta.at(1,s) = myagents_theta[s].report(0.95);
        quantiles_theta.at(2,s) = myagents_theta[s].report(0.50);

        quantiles_thetadot.at(0,s) = myagents_thetadot[s].report(0.05);
        quantiles_thetadot.at(1,s) = myagents_thetadot[s].report(0.95);
        quantiles_thetadot.at(2,s) = myagents_thetadot[s].report(0.50);

        quantiles_cm.at(0,s) = myagents_cm[s].report(0.05);
        quantiles_cm.at(1,s) = myagents_cm[s].report(0.95);
        quantiles_cm.at(2,s) = myagents_cm[s].report(0.50);
      }
      //tquantiles = quantiles.cols(left,right);
      quantiles_theta.save( name + "_theta_quantiles.dat", raw_ascii);
      quantiles_thetadot.save( name + "_thetadot_quantiles.dat", raw_ascii);
      quantiles_cm.save( name + "_cm_quantiles.dat", raw_ascii);
      */
    ////}
}

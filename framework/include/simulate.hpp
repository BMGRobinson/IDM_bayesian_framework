#ifndef SIMULATE_HPP_
#define SIMULATE_HPP_
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "armadillo"
#include <iomanip> //std::setw
#include "statespace.hpp"
#include "IQAgent.hpp"
#include <mpi.h>
#include <iostream>
#pragma GCC diagnostic pop
using namespace arma;

int shift(const mat & referenceSignal, mat & signal, const unsigned maxIndex, const unsigned maxlag, double delayedValue);
void simulate( const unsigned N, const unsigned m, const unsigned qm, const colvec & ic, const mat & paramChain, const mat & ref, statespace & ss,  const std::string name, bool shiftSignal, double delayedValue, const MPI_Comm& com );

double getCrossCorrelation(const rowvec & s1, const rowvec & s2, const double m1, const double sigma1, const double m2, const double sigma2, const int lag);
int getShift(const mat & x, const mat & y, const int maxlag);
mat getShiftedSignal(const mat & x, const mat & y, const int maxlag);
#endif

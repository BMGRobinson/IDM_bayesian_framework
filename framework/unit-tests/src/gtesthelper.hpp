#ifndef GTESTHELPER_HPP_
#define GTESTHELPER_HPP_

#include <gtest/gtest.h>
#include <armadillo>
#include <mpi.h>
#include <cmath>
using namespace arma;

//Helper function to compare two matrices
::testing::AssertionResult compareMatrix(const mat& A, const mat& B) {
	int aCols, aRows, bCols, bRows;
	int i, j;

	aCols = A.n_cols;
	aRows = A.n_rows;
	bCols = B.n_cols;
	bRows = B.n_rows;

	if ((aCols != bCols) || (aRows != bRows) ) {
		return ::testing::AssertionFailure() << "Matrix size error. Actual (" << aRows << "," << aCols << ") != Expected (" << bRows << "," << bCols << ")";
	}

	double eps = 10e-8;
	for (i = 0; i < aRows; i++) {
		for (j = 0; j < aCols; j++) {
			if ( std::abs(A.at(i, j) - B.at(i, j)) > eps ) {
				return ::testing::AssertionFailure()
				       << "matrix[" << i << "," << j << "] (" << A.at(i, j) << ") != expected[" << i << "," << j << "] (" << B.at(i, j) << ")";
			}
		}
	}
	return ::testing::AssertionSuccess();
}

#endif

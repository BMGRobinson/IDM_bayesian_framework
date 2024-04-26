#!/bin/bash
if [ $# -eq 0 ]
	then
	echo "Available options:
	- List		: 	list all tests
	- All 		: 	run all tests
	- Samples 	: 	run samples related tests
	- MCMC 		: 	run mcmc related tests
	- EKF		:	run EKF related tests
	- ENKF		:	run EnKF related tests
	- PF		:	run PF related tests
	- Config : run config related tests
	"
fi

case $1 in
	List )
			mpirun -np 1 ./ut --gtest_list_tests ;;
    All )
			mpirun -np $2 ./ut ;;
    ENKF )
			mpirun -np $2 ./ut --gtest_filter=ENKFTest.* ;;
		SM )
			mpirun -np $2 ./ut --gtest_filter=SMTest.* ;;
    EKF )
      mpirun -np $2 ./ut --gtest_filter=EKFTest.* ;;
    PF )
      mpirun -np $2 ./ut --gtest_filter=PFTest.* ;;
    Samples )
      mpirun -np $2 ./ut --gtest_filter=*SamplesTest.* ;;
		Config )
				./ut.out --gtest_filter=Config* ;;
    esac

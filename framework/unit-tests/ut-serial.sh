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
	"
fi

case $1 in
	List )
				./ut --gtest_list_tests ;;
    All )
        ./ut ;;
		MCMC )
        ./ut --gtest_filter=MCMC.* ;;
    ENKF )
        ./ut --gtest_filter=ENKFTest.* ;;
    EKF )
        ./ut --gtest_filter=EKFTest.* ;;
    PF )
        ./ut --gtest_filter=PFTest.* ;;
    Samples )
        ./ut --gtest_filter=*SamplesTest.* ;;
    esac

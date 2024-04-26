#Getting machine name using uname
UNAME := $(shell uname)
current_dir = $(shell pwd)
ifeq ($(UNAME), Linux) ##Linux
	#MACHINE_DEPENDENT_FLAGS = -isystem $(current_dir)/../libconfig/include/ -isystem $(current_dir)/../armadillo/usr/include/ -L$(current_dir)/../armadillo/usr/lib64 -larmadillo -L$(current_dir)/../libconfig/lib -lconfig++ -rdynamic
	MACHINE_DEPENDENT_FLAGS = -isystem $(BFPATH)/libconfig/include/ -I $(BFPATH)/SOURCE/ -isystem $(BFPATH)/armadillo/usr/include/ -L$(BFPATH)/armadillo/usr/lib64 -larmadillo -L$(BFPATH)/libconfig/lib -lconfig++ -rdynamic
	MACHINE_DEPENDENT_FLAGS_LIB  = -fPIC -shared
endif
ifeq ($(UNAME), Darwin)		#Macbook pro
	MACHINE_DEPENDENT_FLAGS = -I ../armadillo/usr/local/include -I $(current_dir)/../libconfig/include/ -DARMA_DONT_USE_WRAPPER -framework Accelerate -L$(current_dir)/../libconfig/lib -lconfig++
	MACHINE_DEPENDENT_FLAGS_LIB  = -dynamiclib -flat_namespace
endif

CXX				= mpic++
FLAGS 			= -Wall -std=c++11 -O2 -L $(BFPATH)/lib/ -l bf  $(MACHINE_DEPENDENT_FLAGS)

main:
	$(CXX) rosenbrock.cpp $(FLAGS) -o run.out

clean:
	rm -rf *.out *.txt

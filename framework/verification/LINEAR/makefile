#Getting machine name using uname
UNAME := $(shell uname)
current_dir = $(shell pwd)
ifeq ($(UNAME), Linux) ##Linux
	MACHINE_DEPENDENT_FLAGS = -isystem $(BFPATH)/libconfig/include/ -I $(BFPATH)/include/ -isystem $(BFPATH)/armadillo/include/ -L$(BFPATH)/armadillo/lib/ -larmadillo -L$(BFPATH)/libconfig/lib -lconfig++ -rdynamic -ldl
	#MACHINE_DEPENDENT_FLAGS = -isystem $(BFPATH)/libconfig/include/ -I $(BFPATH)/SOURCE/ -isystem $(BFPATH)/armadillo/usr/include/ -L$(BFPATH)/armadillo/usr/lib64/ -larmadillo -L$(BFPATH)/libconfig/lib -lconfig++ -rdynamic
	MACHINE_DEPENDENT_FLAGS_LIB  = -fPIC -shared
endif
ifeq ($(UNAME), Darwin)		#Macbook pro
	MACHINE_DEPENDENT_FLAGS = -I ../armadillo/usr/local/include -I $(current_dir)/../libconfig/include/ -DARMA_DONT_USE_WRAPPER -framework Accelerate -L$(current_dir)/../libconfig/lib -lconfig++
	MACHINE_DEPENDENT_FLAGS_LIB  = -dynamiclib -flat_namespace
endif



CXX				= mpic++
FLAGS 			= -Wall -std=c++11 -O2 -L $(BFPATH) -l bf  $(MACHINE_DEPENDENT_FLAGS)
LIBFLAG			=
#VMFLAGS 		= -Wall -std=c++11 -O2 -isystem $(arma_path)/usr/include/ -isystem /home/phil/work/libconf/include/ -L $(arma_path)/usr/lib64 -L/home/phil/work/libconf/lib -lconfig++ -larmadillo
#GTEST			= -I/media/scratch/philippe/gtest-1.7.0/include -L/media/scratch/philippe/gtest-1.7.0/build -lgtest -lgtest_main -lpthread
#MFLAGS			= -Wall -std=c++11 -O2 -larmadillo
#MACFLAGS		= -Wall -std=c++11 -O2
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/media/scratch/philippe/gtest-1.7.0/build:/media/scratch/philippe/armadillo/usr/lib64:/media/scratch/philippe/gcc/lib64/

main:
	$(CXX) main.cpp $(FLAGS) -o run.out
post:
	$(CXX) postprocess.cpp $(FLAGS) -o post.out
quad:
	$(CXX) quad.cpp $(FLAGS) -o quad.out
data:
	$(CXX) gen_data.cpp $(FLAGS) -o data.out
func:
	$(CXX) $(FLAGS) $(MACHINE_DEPENDENT_FLAGS_LIB) functions.cpp -o libf.so
opt:
	$(CXX) opt.cpp $(FLAGS) -o opt.out
clean:
	rm -rf *.out *.txt

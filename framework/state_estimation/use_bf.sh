##GCC
#GCC_PATH=/media/scratch/Software/GCC
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GCC_PATH/lib64
#export PATH=$GCC_PATH/bin:$PATH

#ARMADILLO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/armadillo/lib

#LIBCONFIG
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/libconfig/lib

#OPENMPI
#OPENMPI_PATH=$PWD/../../software/OpenMPI-1.8.8
OPENMPI_PATH=/home/phil/software/OpenMPI-1.8.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENMPI_PATH/lib
export PATH=$OPENMPI_PATH/bin:$PATH

#GTEST
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/unit-tests/gtest-1.7.0/build

export BFPATH=$(pwd)

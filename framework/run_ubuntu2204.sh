rm CMakeCache.txt
rm -rf armadillo
rm -rf libconfig
rm -rf External/armadillo
rm -rf External/armadillo-8.500.1
rm -rf External/libconfig
rm -rf External/libconfig-1.7.2

cmake .
make armadillo
make libconfig
make 
export BFPATH=${PWD}

cd Examples
rm CMakeCache.txt
cmake .
make

module --force purge
module load nixpkgs/16.09 intel/2016.4 armadillo/8.500.1 openmpi/1.8.8

rm CMakeCache.txt
rm -rf armadillo
rm -rf libconfig
rm -rf External/armadillo
rm -rf External/armadillo-8.500.1
rm -rf External/libconfig
rm -rf External/libconfig-1.7.2

cmake CMakeLists_graham.txt
make 
export BFPATH=${PWD}

cd Examples
rm CMakeCache.txt
cmake .
make

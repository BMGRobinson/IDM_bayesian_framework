project (bf CXX)
include(ExternalProject)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_COMPILER mpic++)

add_library(bf
  src/bayesianPosterior.cpp
  src/config.cpp
  src/filters.cpp
  src/mcmc.cpp
  src/optimization.cpp
  src/pdf.cpp
  src/samples.cpp
  src/statespace.cpp
  src/wrapper.cpp
  src/tmcmc.cpp
  src/simulate.cpp
  src/IQAgent.cpp
)

target_include_directories(bf
  PUBLIC include
  PIRVATE src
)

#OpenMPI
find_package(MPI REQUIRED)
add_library(MPI INTERFACE IMPORTED)

################################################################################
# Armadillo C++
# see http://arma.sourceforge.net/
########


#TO UPDATE ARMADILLO C++ CHANGE THE VALUES OF THESE LINES AND DON'T FORGET TO DELETE THE armadillo
#FOLDER IN EXTERNAL
  #URL ${CMAKE_CURRENT_BINARY_DIR}/External/armadillo-7.950.1.tar.xz
  #SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/armadillo-7.950.1
ExternalProject_Add(armadillo
  URL ${CMAKE_CURRENT_BINARY_DIR}/External/armadillo-8.500.1.tar.xz
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/External/armadillo
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/armadillo-8.500.1
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/armadillo/ -DCMAKE_INSTALL_LIBDIR=lib
  INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/armadillo/
)

set_target_properties(armadillo PROPERTIES SUFFIX ".so")

################################################################################
### libconfig
### see https://hyperrealm.github.io/libconfig/

ExternalProject_Add(libconfig
  URL ${CMAKE_CURRENT_BINARY_DIR}/External/libconfig-1.7.2.tar.gz
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/External/libconfig
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/External/libconfig-1.7.2
  #INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/libconfig/
  CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/External/libconfig-1.7.2/configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/libconfig
  #PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libconfig
  BUILD_COMMAND make
  #BUILD_IN_SOURCE 1
)

################################################################################
### Linking

target_include_directories(bf PRIVATE "${CMAKE_BINARY_DIR}/armadillo/include/" )
target_link_libraries(bf ${armadillo_LIBRARIES})
add_dependencies(bf armadillo)

target_include_directories(bf PRIVATE "${CMAKE_BINARY_DIR}/libconfig/include/")
target_link_libraries(bf ${libconfig_LIBRARIES})
add_dependencies(bf libconfig)

target_link_libraries(bf ${MPI_LIBRARIES})
target_include_directories(bf PRIVATE ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "-O2")
set (CMAKE_CXX_STANDARD 11)
set_target_properties(bf PROPERTIES CXX_STANDARD 11)
set_target_properties(bf PROPERTIES CXX_STANDARD_REQUIRED TRUE)

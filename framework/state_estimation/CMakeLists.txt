project (run CXX)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_COMPILER mpic++)

################################################################################
### Adding targets
add_executable(run main.cpp)

################################################################################
### Linking to Open MPI
find_package(MPI REQUIRED)
add_library(MPI INTERFACE IMPORTED)

target_link_libraries(run ${MPI_LIBRARIES})
target_include_directories(run PRIVATE ${MPI_INCLUDE_PATH})

################################################################################
### Linking to bayesian-framework
add_library(bf STATIC IMPORTED)
set_target_properties(bf PROPERTIES IMPORTED_LOCATION  $ENV{BFPATH}/libbf.a)

target_link_libraries(run bf)
target_include_directories(run PRIVATE $ENV{BFPATH}/include)

################################################################################
### Linking to dl
target_link_libraries(run dl)

################################################################################
### Linking to libconfig++
add_library(config++ SHARED IMPORTED)
set_target_properties(config++ PROPERTIES
  IMPORTED_LOCATION  $ENV{BFPATH}/libconfig/lib/libconfig++.so
  INTERFACE_INCLUDE_DIRECTORIES $ENV{BFPATH}/libconfig/lib
)

target_include_directories(run PRIVATE "$ENV{BFPATH}/libconfig/include/")
target_link_libraries(run config++)

################################################################################
### Linking to armadillo c++
target_include_directories(run PRIVATE "$ENV{BFPATH}/armadillo/include/" )
target_link_libraries(run "$ENV{BFPATH}/armadillo/lib/libarmadillo.so")




################################################################################


set(CMAKE_CXX_FLAGS "-O2")
set (CMAKE_CXX_STANDARD 11)
set_target_properties(run PROPERTIES CXX_STANDARD 11)
set_target_properties(run PROPERTIES CXX_STANDARD_REQUIRED TRUE)

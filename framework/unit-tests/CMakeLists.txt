project (ut CXX)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_COMPILER mpic++)

include(ExternalProject)

################################################################################
### Adding targets

add_executable(ut
  #src/mcmc.cpp
  #src/config.cpp
  src/filters.cpp
  src/gausshermite.cpp
  src/bayesianposterior.cpp
  src/samples.cpp
  src/pdf.cpp
  src/unittests_mpi.cpp
)

################################################################################
### Linking to Open MPI
find_package(MPI REQUIRED)
add_library(MPI INTERFACE IMPORTED)

target_link_libraries(ut ${MPI_LIBRARIES})
target_include_directories(ut PRIVATE ${MPI_INCLUDE_PATH})


################################################################################
### Linking to bayesian-framework
add_library(bf STATIC IMPORTED)
set_target_properties(bf PROPERTIES IMPORTED_LOCATION  $ENV{BFPATH}/libbf.a)

target_link_libraries(ut bf)
target_include_directories(ut PRIVATE $ENV{BFPATH}/include)

#Adding lapack and blas
#find_package( LAPACK REQUIRED )
#target_link_libraries(ut
#  ${LAPACK_LIBRARIES}
#)

################################################################################
### Linking to libconfig++
add_library(config++ SHARED IMPORTED)
set_target_properties(config++ PROPERTIES
  IMPORTED_LOCATION  $ENV{BFPATH}/libconfig/lib/libconfig++.so
  IMPORTED_LOCATION  $ENV{BFPATH}/libconfig/lib/libconfig++.dylib
  INTERFACE_INCLUDE_DIRECTORIES $ENV{BFPATH}/libconfig/lib
)

target_include_directories(ut PRIVATE "$ENV{BFPATH}/libconfig/include/")
target_link_libraries(ut config++)

################################################################################
### Linking to armadillo c++
add_library(armadillo SHARED IMPORTED)
set_target_properties(armadillo PROPERTIES
  IMPORTED_LOCATION  $ENV{BFPATH}/armadillo/lib/libarmadillo.so
  IMPORTED_LOCATION  $ENV{BFPATH}/armadillo/lib/libarmadillo.dylib
  INTERFACE_INCLUDE_DIRECTORIES $ENV{BFPATH}/armadillo/lib/
)

target_include_directories(ut PRIVATE "$ENV{BFPATH}/armadillo/include/" )
target_link_libraries(ut armadillo)
#target_link_libraries(ut "$ENV{BFPATH}/armadillo/lib/libarmadillo.so")
#target_link_libraries(ut "$ENV{BFPATH}/armadillo/lib/libarmadillo.dylib")


################################################################################
### Installing and compiling GTEST
# We need thread support
find_package(Threads REQUIRED)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/master.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
)

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gtest source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)

# Set libgmock properties
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

include_directories("${source_dir}/googletest/include" "${source_dir}/googlemock/include")

target_link_libraries(ut
  libgtest
  libgmock
)

set(CMAKE_CXX_FLAGS "-O2")
set (CMAKE_CXX_STANDARD 11)
set_target_properties(ut PROPERTIES CXX_STANDARD 11)
set_target_properties(ut PROPERTIES CXX_STANDARD_REQUIRED TRUE)

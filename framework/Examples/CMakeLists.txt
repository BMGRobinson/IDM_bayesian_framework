project (run CXX)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_COMPILER mpic++)

################################################################################
### Adding targets

add_library(f SHARED functions.cpp )
add_executable(run main.cpp)
add_executable(data gen_data.cpp)

################################################################################
### Linking to Open MPI
find_package(MPI REQUIRED)
add_library(MPI INTERFACE IMPORTED)

target_link_libraries(data ${MPI_LIBRARIES})
target_include_directories(data PRIVATE ${MPI_INCLUDE_PATH})

target_link_libraries(run ${MPI_LIBRARIES})
target_include_directories(run PRIVATE ${MPI_INCLUDE_PATH})

target_link_libraries(f ${MPI_LIBRARIES})
target_include_directories(f PRIVATE ${MPI_INCLUDE_PATH})


################################################################################
### Linking to bayesian-framework
add_library(bf STATIC IMPORTED)
set_target_properties(bf PROPERTIES IMPORTED_LOCATION  $ENV{BFPATH}/libbf.a)

target_link_libraries(run bf)
target_include_directories(run PRIVATE $ENV{BFPATH}/include)

target_link_libraries(data bf)
target_include_directories(data PRIVATE $ENV{BFPATH}/include)

target_link_libraries(f bf)
target_include_directories(f PRIVATE $ENV{BFPATH}/include)

################################################################################
### Linking to dl
target_link_libraries(run dl)
target_link_libraries(data dl)

################################################################################
### Linking to libconfig++
target_include_directories(run PRIVATE "$ENV{BFPATH}/libconfig/include/" )
target_include_directories(data PRIVATE "$ENV{BFPATH}/libconfig/include/" )
target_include_directories(f PRIVATE "$ENV{BFPATH}/libconfig/include/" )
IF (APPLE)
  target_link_libraries(run "$ENV{BFPATH}/libconfig/lib/libconfig++.dylib")
  target_link_libraries(data "$ENV{BFPATH}/libconfig/lib/libconfig++.dylib")
  target_link_libraries(f "$ENV{BFPATH}/libconfig/lib/libconfig++.dylib")
ELSE (APPLE)
  target_link_libraries(run "$ENV{BFPATH}/libconfig/lib/libconfig++.so")
  target_link_libraries(data "$ENV{BFPATH}/libconfig/lib/libconfig++.so")
  target_link_libraries(f "$ENV{BFPATH}/libconfig/lib/libconfig++.so")
ENDIF (APPLE)

################################################################################
### Linking to armadillo c++
target_include_directories(run PRIVATE "$ENV{BFPATH}/armadillo/include/" )
target_include_directories(data PRIVATE "$ENV{BFPATH}/armadillo/include/" )
target_include_directories(f PRIVATE "$ENV{BFPATH}/armadillo/include/" )

IF (APPLE)
  target_link_libraries(run "$ENV{BFPATH}/armadillo/lib/libarmadillo.dylib")
  target_link_libraries(data "$ENV{BFPATH}/armadillo/lib/libarmadillo.dylib")
  target_link_libraries(f "$ENV{BFPATH}/armadillo/lib/libarmadillo.dylib")
ELSE (APPLE)
  target_link_libraries(run "$ENV{BFPATH}/armadillo/lib/libarmadillo.so")
  target_link_libraries(data "$ENV{BFPATH}/armadillo/lib/libarmadillo.so")
  target_link_libraries(f "$ENV{BFPATH}/armadillo/lib/libarmadillo.so")
ENDIF (APPLE)


################################################################################
### Linking to f
target_link_libraries(run f)
target_link_libraries(data f)


set(CMAKE_CXX_FLAGS "-O2")
set (CMAKE_CXX_STANDARD 11)
set_target_properties(run PROPERTIES CXX_STANDARD 11)
set_target_properties(run PROPERTIES CXX_STANDARD_REQUIRED TRUE)

set_target_properties(f PROPERTIES CXX_STANDARD 11)
set_target_properties(f PROPERTIES CXX_STANDARD_REQUIRED TRUE)

set_target_properties(data PROPERTIES CXX_STANDARD 11)
set_target_properties(data PROPERTIES CXX_STANDARD_REQUIRED TRUE)

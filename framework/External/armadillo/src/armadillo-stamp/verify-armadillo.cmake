# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if("/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1.tar.xz" STREQUAL "")
  message(FATAL_ERROR "LOCAL can't be empty")
endif()

if(NOT EXISTS "/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1.tar.xz")
  message(FATAL_ERROR "File not found: /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1.tar.xz")
endif()

if("" STREQUAL "")
  message(WARNING "File will not be verified since no URL_HASH specified")
  return()
endif()

if("" STREQUAL "")
  message(FATAL_ERROR "EXPECT_VALUE can't be empty")
endif()

message(STATUS "verifying file...
     file='/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1.tar.xz'")

file("" "/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1.tar.xz" actual_value)

if(NOT "${actual_value}" STREQUAL "")
  message(FATAL_ERROR "error:  hash of
  /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1.tar.xz
does not match expected value
  expected: ''
    actual: '${actual_value}'
")
endif()

message(STATUS "verifying file... done")

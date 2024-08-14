# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc

# Include any dependencies generated for this target.
include CMakeFiles/f.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/f.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/f.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/f.dir/flags.make

CMakeFiles/f.dir/functions.cpp.o: CMakeFiles/f.dir/flags.make
CMakeFiles/f.dir/functions.cpp.o: functions.cpp
CMakeFiles/f.dir/functions.cpp.o: CMakeFiles/f.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/f.dir/functions.cpp.o"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/f.dir/functions.cpp.o -MF CMakeFiles/f.dir/functions.cpp.o.d -o CMakeFiles/f.dir/functions.cpp.o -c /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc/functions.cpp

CMakeFiles/f.dir/functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/f.dir/functions.cpp.i"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc/functions.cpp > CMakeFiles/f.dir/functions.cpp.i

CMakeFiles/f.dir/functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/f.dir/functions.cpp.s"
	mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc/functions.cpp -o CMakeFiles/f.dir/functions.cpp.s

# Object files for target f
f_OBJECTS = \
"CMakeFiles/f.dir/functions.cpp.o"

# External object files for target f
f_EXTERNAL_OBJECTS =

libf.so: CMakeFiles/f.dir/functions.cpp.o
libf.so: CMakeFiles/f.dir/build.make
libf.so: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
libf.so: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
libf.so: /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/libbf.a
libf.so: /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/libconfig/lib/libconfig++.so
libf.so: /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/armadillo/lib/libarmadillo.so
libf.so: CMakeFiles/f.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libf.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/f.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/f.dir/build: libf.so
.PHONY : CMakeFiles/f.dir/build

CMakeFiles/f.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/f.dir/cmake_clean.cmake
.PHONY : CMakeFiles/f.dir/clean

CMakeFiles/f.dir/depend:
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/Examples/IDM_2024/sir_mcmc/CMakeFiles/f.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/f.dir/depend

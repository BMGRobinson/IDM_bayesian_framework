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
CMAKE_SOURCE_DIR = /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework

# Utility rule file for armadillo.

# Include any custom commands dependencies for this target.
include CMakeFiles/armadillo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/armadillo.dir/progress.make

CMakeFiles/armadillo: CMakeFiles/armadillo-complete

CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-install
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-mkdir
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-download
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-update
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-patch
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-configure
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-build
CMakeFiles/armadillo-complete: External/armadillo/src/armadillo-stamp/armadillo-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'armadillo'"
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles/armadillo-complete
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-done

External/armadillo/src/armadillo-stamp/armadillo-build: External/armadillo/src/armadillo-stamp/armadillo-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing build step for 'armadillo'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build && $(MAKE)
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-build

External/armadillo/src/armadillo-stamp/armadillo-configure: External/armadillo/tmp/armadillo-cfgcmd.txt
External/armadillo/src/armadillo-stamp/armadillo-configure: External/armadillo/src/armadillo-stamp/armadillo-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Performing configure step for 'armadillo'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build && /usr/bin/cmake -DCMAKE_INSTALL_PREFIX=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/armadillo/ -DCMAKE_INSTALL_LIBDIR=lib "-GUnix Makefiles" /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-configure

External/armadillo/src/armadillo-stamp/armadillo-download: External/armadillo/src/armadillo-stamp/armadillo-urlinfo.txt
External/armadillo/src/armadillo-stamp/armadillo-download: External/armadillo/src/armadillo-stamp/armadillo-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (verify and extract) for 'armadillo'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External && /usr/bin/cmake -P /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/verify-armadillo.cmake
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External && /usr/bin/cmake -P /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/extract-armadillo.cmake
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-download

External/armadillo/src/armadillo-stamp/armadillo-install: External/armadillo/src/armadillo-stamp/armadillo-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing install step for 'armadillo'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build && $(MAKE) install
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-install

External/armadillo/src/armadillo-stamp/armadillo-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'armadillo'"
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo-8.500.1
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-build
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/armadillo/
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/tmp
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-mkdir

External/armadillo/src/armadillo-stamp/armadillo-patch: External/armadillo/src/armadillo-stamp/armadillo-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'armadillo'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-patch

External/armadillo/src/armadillo-stamp/armadillo-update: External/armadillo/src/armadillo-stamp/armadillo-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No update step for 'armadillo'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/armadillo/src/armadillo-stamp/armadillo-update

armadillo: CMakeFiles/armadillo
armadillo: CMakeFiles/armadillo-complete
armadillo: External/armadillo/src/armadillo-stamp/armadillo-build
armadillo: External/armadillo/src/armadillo-stamp/armadillo-configure
armadillo: External/armadillo/src/armadillo-stamp/armadillo-download
armadillo: External/armadillo/src/armadillo-stamp/armadillo-install
armadillo: External/armadillo/src/armadillo-stamp/armadillo-mkdir
armadillo: External/armadillo/src/armadillo-stamp/armadillo-patch
armadillo: External/armadillo/src/armadillo-stamp/armadillo-update
armadillo: CMakeFiles/armadillo.dir/build.make
.PHONY : armadillo

# Rule to build all files generated by this target.
CMakeFiles/armadillo.dir/build: armadillo
.PHONY : CMakeFiles/armadillo.dir/build

CMakeFiles/armadillo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/armadillo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/armadillo.dir/clean

CMakeFiles/armadillo.dir/depend:
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles/armadillo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/armadillo.dir/depend

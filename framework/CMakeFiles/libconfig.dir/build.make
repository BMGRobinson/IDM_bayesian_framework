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

# Utility rule file for libconfig.

# Include any custom commands dependencies for this target.
include CMakeFiles/libconfig.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/libconfig.dir/progress.make

CMakeFiles/libconfig: CMakeFiles/libconfig-complete

CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-install
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-mkdir
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-download
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-update
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-patch
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-configure
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-build
CMakeFiles/libconfig-complete: External/libconfig/src/libconfig-stamp/libconfig-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'libconfig'"
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles/libconfig-complete
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-done

External/libconfig/src/libconfig-stamp/libconfig-build: External/libconfig/src/libconfig-stamp/libconfig-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing build step for 'libconfig'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build && make
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-build

External/libconfig/src/libconfig-stamp/libconfig-configure: External/libconfig/tmp/libconfig-cfgcmd.txt
External/libconfig/src/libconfig-stamp/libconfig-configure: External/libconfig/src/libconfig-stamp/libconfig-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Performing configure step for 'libconfig'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build && /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig-1.7.2/configure --prefix=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/libconfig
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-configure

External/libconfig/src/libconfig-stamp/libconfig-download: External/libconfig/src/libconfig-stamp/libconfig-urlinfo.txt
External/libconfig/src/libconfig-stamp/libconfig-download: External/libconfig/src/libconfig-stamp/libconfig-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (verify and extract) for 'libconfig'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External && /usr/bin/cmake -P /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/verify-libconfig.cmake
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External && /usr/bin/cmake -P /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/extract-libconfig.cmake
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-download

External/libconfig/src/libconfig-stamp/libconfig-install: External/libconfig/src/libconfig-stamp/libconfig-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing install step for 'libconfig'"
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build && $(MAKE) install
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build && /usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-install

External/libconfig/src/libconfig-stamp/libconfig-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'libconfig'"
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig-1.7.2
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-build
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/tmp
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src
	/usr/bin/cmake -E make_directory /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-mkdir

External/libconfig/src/libconfig-stamp/libconfig-patch: External/libconfig/src/libconfig-stamp/libconfig-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'libconfig'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-patch

External/libconfig/src/libconfig-stamp/libconfig-update: External/libconfig/src/libconfig-stamp/libconfig-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No update step for 'libconfig'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/External/libconfig/src/libconfig-stamp/libconfig-update

libconfig: CMakeFiles/libconfig
libconfig: CMakeFiles/libconfig-complete
libconfig: External/libconfig/src/libconfig-stamp/libconfig-build
libconfig: External/libconfig/src/libconfig-stamp/libconfig-configure
libconfig: External/libconfig/src/libconfig-stamp/libconfig-download
libconfig: External/libconfig/src/libconfig-stamp/libconfig-install
libconfig: External/libconfig/src/libconfig-stamp/libconfig-mkdir
libconfig: External/libconfig/src/libconfig-stamp/libconfig-patch
libconfig: External/libconfig/src/libconfig-stamp/libconfig-update
libconfig: CMakeFiles/libconfig.dir/build.make
.PHONY : libconfig

# Rule to build all files generated by this target.
CMakeFiles/libconfig.dir/build: libconfig
.PHONY : CMakeFiles/libconfig.dir/build

CMakeFiles/libconfig.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libconfig.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libconfig.dir/clean

CMakeFiles/libconfig.dir/depend:
	cd /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework /home/brandon/Documents/1_Carleton/2_Research/3_Code/Philippe/framework/CMakeFiles/libconfig.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libconfig.dir/depend


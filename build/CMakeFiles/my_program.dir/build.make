# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bingbing/Computation_Fluid/hw4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bingbing/Computation_Fluid/hw4/build

# Include any dependencies generated for this target.
include CMakeFiles/my_program.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/my_program.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/my_program.dir/flags.make

CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.o: CMakeFiles/my_program.dir/flags.make
CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.o: ../src/Lax_Wendroff.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bingbing/Computation_Fluid/hw4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.o -c /home/bingbing/Computation_Fluid/hw4/src/Lax_Wendroff.cpp

CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bingbing/Computation_Fluid/hw4/src/Lax_Wendroff.cpp > CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.i

CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bingbing/Computation_Fluid/hw4/src/Lax_Wendroff.cpp -o CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.s

CMakeFiles/my_program.dir/src/Three_Order.cpp.o: CMakeFiles/my_program.dir/flags.make
CMakeFiles/my_program.dir/src/Three_Order.cpp.o: ../src/Three_Order.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bingbing/Computation_Fluid/hw4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/my_program.dir/src/Three_Order.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_program.dir/src/Three_Order.cpp.o -c /home/bingbing/Computation_Fluid/hw4/src/Three_Order.cpp

CMakeFiles/my_program.dir/src/Three_Order.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_program.dir/src/Three_Order.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bingbing/Computation_Fluid/hw4/src/Three_Order.cpp > CMakeFiles/my_program.dir/src/Three_Order.cpp.i

CMakeFiles/my_program.dir/src/Three_Order.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_program.dir/src/Three_Order.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bingbing/Computation_Fluid/hw4/src/Three_Order.cpp -o CMakeFiles/my_program.dir/src/Three_Order.cpp.s

CMakeFiles/my_program.dir/src/Warming_Beam.cpp.o: CMakeFiles/my_program.dir/flags.make
CMakeFiles/my_program.dir/src/Warming_Beam.cpp.o: ../src/Warming_Beam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bingbing/Computation_Fluid/hw4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/my_program.dir/src/Warming_Beam.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_program.dir/src/Warming_Beam.cpp.o -c /home/bingbing/Computation_Fluid/hw4/src/Warming_Beam.cpp

CMakeFiles/my_program.dir/src/Warming_Beam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_program.dir/src/Warming_Beam.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bingbing/Computation_Fluid/hw4/src/Warming_Beam.cpp > CMakeFiles/my_program.dir/src/Warming_Beam.cpp.i

CMakeFiles/my_program.dir/src/Warming_Beam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_program.dir/src/Warming_Beam.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bingbing/Computation_Fluid/hw4/src/Warming_Beam.cpp -o CMakeFiles/my_program.dir/src/Warming_Beam.cpp.s

CMakeFiles/my_program.dir/src/main.cpp.o: CMakeFiles/my_program.dir/flags.make
CMakeFiles/my_program.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bingbing/Computation_Fluid/hw4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/my_program.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/my_program.dir/src/main.cpp.o -c /home/bingbing/Computation_Fluid/hw4/src/main.cpp

CMakeFiles/my_program.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/my_program.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bingbing/Computation_Fluid/hw4/src/main.cpp > CMakeFiles/my_program.dir/src/main.cpp.i

CMakeFiles/my_program.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/my_program.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bingbing/Computation_Fluid/hw4/src/main.cpp -o CMakeFiles/my_program.dir/src/main.cpp.s

# Object files for target my_program
my_program_OBJECTS = \
"CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.o" \
"CMakeFiles/my_program.dir/src/Three_Order.cpp.o" \
"CMakeFiles/my_program.dir/src/Warming_Beam.cpp.o" \
"CMakeFiles/my_program.dir/src/main.cpp.o"

# External object files for target my_program
my_program_EXTERNAL_OBJECTS =

my_program: CMakeFiles/my_program.dir/src/Lax_Wendroff.cpp.o
my_program: CMakeFiles/my_program.dir/src/Three_Order.cpp.o
my_program: CMakeFiles/my_program.dir/src/Warming_Beam.cpp.o
my_program: CMakeFiles/my_program.dir/src/main.cpp.o
my_program: CMakeFiles/my_program.dir/build.make
my_program: CMakeFiles/my_program.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bingbing/Computation_Fluid/hw4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable my_program"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_program.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my_program.dir/build: my_program

.PHONY : CMakeFiles/my_program.dir/build

CMakeFiles/my_program.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/my_program.dir/cmake_clean.cmake
.PHONY : CMakeFiles/my_program.dir/clean

CMakeFiles/my_program.dir/depend:
	cd /home/bingbing/Computation_Fluid/hw4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bingbing/Computation_Fluid/hw4 /home/bingbing/Computation_Fluid/hw4 /home/bingbing/Computation_Fluid/hw4/build /home/bingbing/Computation_Fluid/hw4/build /home/bingbing/Computation_Fluid/hw4/build/CMakeFiles/my_program.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/my_program.dir/depend


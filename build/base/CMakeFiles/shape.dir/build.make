# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_SOURCE_DIR = /home/syple/code/auto_engine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/syple/code/auto_engine/build

# Include any dependencies generated for this target.
include base/CMakeFiles/shape.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include base/CMakeFiles/shape.dir/compiler_depend.make

# Include the progress variables for this target.
include base/CMakeFiles/shape.dir/progress.make

# Include the compile flags for this target's objects.
include base/CMakeFiles/shape.dir/flags.make

base/CMakeFiles/shape.dir/shape.cc.o: base/CMakeFiles/shape.dir/flags.make
base/CMakeFiles/shape.dir/shape.cc.o: /home/syple/code/auto_engine/base/shape.cc
base/CMakeFiles/shape.dir/shape.cc.o: base/CMakeFiles/shape.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/syple/code/auto_engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object base/CMakeFiles/shape.dir/shape.cc.o"
	cd /home/syple/code/auto_engine/build/base && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT base/CMakeFiles/shape.dir/shape.cc.o -MF CMakeFiles/shape.dir/shape.cc.o.d -o CMakeFiles/shape.dir/shape.cc.o -c /home/syple/code/auto_engine/base/shape.cc

base/CMakeFiles/shape.dir/shape.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/shape.dir/shape.cc.i"
	cd /home/syple/code/auto_engine/build/base && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/syple/code/auto_engine/base/shape.cc > CMakeFiles/shape.dir/shape.cc.i

base/CMakeFiles/shape.dir/shape.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/shape.dir/shape.cc.s"
	cd /home/syple/code/auto_engine/build/base && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/syple/code/auto_engine/base/shape.cc -o CMakeFiles/shape.dir/shape.cc.s

# Object files for target shape
shape_OBJECTS = \
"CMakeFiles/shape.dir/shape.cc.o"

# External object files for target shape
shape_EXTERNAL_OBJECTS =

base/libshape.a: base/CMakeFiles/shape.dir/shape.cc.o
base/libshape.a: base/CMakeFiles/shape.dir/build.make
base/libshape.a: base/CMakeFiles/shape.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/syple/code/auto_engine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libshape.a"
	cd /home/syple/code/auto_engine/build/base && $(CMAKE_COMMAND) -P CMakeFiles/shape.dir/cmake_clean_target.cmake
	cd /home/syple/code/auto_engine/build/base && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/shape.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
base/CMakeFiles/shape.dir/build: base/libshape.a
.PHONY : base/CMakeFiles/shape.dir/build

base/CMakeFiles/shape.dir/clean:
	cd /home/syple/code/auto_engine/build/base && $(CMAKE_COMMAND) -P CMakeFiles/shape.dir/cmake_clean.cmake
.PHONY : base/CMakeFiles/shape.dir/clean

base/CMakeFiles/shape.dir/depend:
	cd /home/syple/code/auto_engine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/syple/code/auto_engine /home/syple/code/auto_engine/base /home/syple/code/auto_engine/build /home/syple/code/auto_engine/build/base /home/syple/code/auto_engine/build/base/CMakeFiles/shape.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : base/CMakeFiles/shape.dir/depend


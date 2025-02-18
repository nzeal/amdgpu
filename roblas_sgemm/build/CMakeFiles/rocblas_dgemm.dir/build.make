# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_SOURCE_DIR = /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build

# Include any dependencies generated for this target.
include CMakeFiles/rocblas_dgemm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/rocblas_dgemm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/rocblas_dgemm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rocblas_dgemm.dir/flags.make

CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o: CMakeFiles/rocblas_dgemm.dir/flags.make
CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o: ../rocblas_dgemm.cpp
CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o: CMakeFiles/rocblas_dgemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o"
	/usr/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o -MF CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o.d -o CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o -c /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/rocblas_dgemm.cpp

CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.i"
	/usr/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/rocblas_dgemm.cpp > CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.i

CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.s"
	/usr/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/rocblas_dgemm.cpp -o CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.s

CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o: CMakeFiles/rocblas_dgemm.dir/flags.make
CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o: ../kernel0_rocblas.cpp
CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o: CMakeFiles/rocblas_dgemm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o"
	/usr/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o -MF CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o.d -o CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o -c /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/kernel0_rocblas.cpp

CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.i"
	/usr/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/kernel0_rocblas.cpp > CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.i

CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.s"
	/usr/bin/hipcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/kernel0_rocblas.cpp -o CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.s

# Object files for target rocblas_dgemm
rocblas_dgemm_OBJECTS = \
"CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o" \
"CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o"

# External object files for target rocblas_dgemm
rocblas_dgemm_EXTERNAL_OBJECTS =

rocblas_dgemm: CMakeFiles/rocblas_dgemm.dir/rocblas_dgemm.cpp.o
rocblas_dgemm: CMakeFiles/rocblas_dgemm.dir/kernel0_rocblas.cpp.o
rocblas_dgemm: CMakeFiles/rocblas_dgemm.dir/build.make
rocblas_dgemm: /opt/rocm-6.0.3/lib/libhipblas.so
rocblas_dgemm: /opt/rocm-6.0.3/lib/librocblas.so
rocblas_dgemm: CMakeFiles/rocblas_dgemm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable rocblas_dgemm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rocblas_dgemm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rocblas_dgemm.dir/build: rocblas_dgemm
.PHONY : CMakeFiles/rocblas_dgemm.dir/build

CMakeFiles/rocblas_dgemm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rocblas_dgemm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rocblas_dgemm.dir/clean

CMakeFiles/rocblas_dgemm.dir/depend:
	cd /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build /pfs/lustrep1/scratch/project_465000972/test/HPCTrainingExamples/HIP/rocblasother/build/CMakeFiles/rocblas_dgemm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rocblas_dgemm.dir/depend


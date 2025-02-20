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
CMAKE_SOURCE_DIR = /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build

# Include any dependencies generated for this target.
include CMakeFiles/DGEMM_CUDA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/DGEMM_CUDA.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/DGEMM_CUDA.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DGEMM_CUDA.dir/flags.make

CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.o: ../dgemm_cuda.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.o -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/dgemm_cuda.cpp

CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.i"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/dgemm_cuda.cpp > CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.i

CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.s"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/dgemm_cuda.cpp -o CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.s

CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o: ../run_dgemm.cu
CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/run_dgemm.cu -o CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o

CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o: ../cudafun/kernel_runner.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/kernel_runner.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o: ../cudafun/matrix_management.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/matrix_management.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o: ../cudafun/data_transfer.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/data_transfer.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o: ../cudafun/matrix_mul_kernel1.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/matrix_mul_kernel1.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o: ../cudafun/matrix_mul_kernel2.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/matrix_mul_kernel2.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o: ../cudafun/matrix_mul_kernel3.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/matrix_mul_kernel3.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o: ../cudafun/matrix_mul_kernel4.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/matrix_mul_kernel4.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o: ../cudafun/matrix_mul_kernel5.cu
CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o: CMakeFiles/DGEMM_CUDA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CUDA object CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o -MF CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o.d -x cu -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/cudafun/matrix_mul_kernel5.cu -o CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.o: CMakeFiles/DGEMM_CUDA.dir/flags.make
CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.o: ../utility/print_summary.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.o"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.o -c /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/utility/print_summary.cpp

CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.i"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/utility/print_summary.cpp > CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.i

CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.s"
	/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/compilers/bin/nvc++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/utility/print_summary.cpp -o CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.s

# Object files for target DGEMM_CUDA
DGEMM_CUDA_OBJECTS = \
"CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.o" \
"CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o" \
"CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.o"

# External object files for target DGEMM_CUDA
DGEMM_CUDA_EXTERNAL_OBJECTS =

DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/dgemm_cuda.cpp.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/run_dgemm.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/kernel_runner.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_management.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/data_transfer.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel1.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel2.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel3.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel4.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/cudafun/matrix_mul_kernel5.cu.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/utility/print_summary.cpp.o
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/build.make
DGEMM_CUDA: /leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/lib64/libcudart_static.a
DGEMM_CUDA: /usr/lib64/librt.so
DGEMM_CUDA: CMakeFiles/DGEMM_CUDA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable DGEMM_CUDA"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DGEMM_CUDA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DGEMM_CUDA.dir/build: DGEMM_CUDA
.PHONY : CMakeFiles/DGEMM_CUDA.dir/build

CMakeFiles/DGEMM_CUDA.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DGEMM_CUDA.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DGEMM_CUDA.dir/clean

CMakeFiles/DGEMM_CUDA.dir/depend:
	cd /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build /leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/CMakeFiles/DGEMM_CUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DGEMM_CUDA.dir/depend


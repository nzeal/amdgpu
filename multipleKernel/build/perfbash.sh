#!/bin/bash

# Path to the dgemm_cuda executable
EXE_PATH="/leonardo_scratch/large/userinternal/nshukla1/myGIT/amdgpu/multipleKernel/build/DGEMM_CUDA"

# Kernel 1
echo "Running Kernel 1..."
perf stat -e cycles,instructions,cache-references,cache-misses $EXE_PATH 1 1024

# Kernel 2
echo "Running Kernel 2..."
perf stat -e cycles,instructions,cache-references,cache-misses $EXE_PATH 2 1024

# Kernel 3
echo "Running Kernel 3..."
perf stat -e cycles,instructions,cache-references,cache-misses $EXE_PATH 3 1024

# Kernel 4
echo "Running Kernel 4..."
perf stat -e cycles,instructions,cache-references,cache-misses $EXE_PATH 4 1024

# Kernel 5
echo "Running Kernel 5..."
perf stat -e cycles,instructions,cache-references,cache-misses $EXE_PATH 5 1024


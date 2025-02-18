cmake_minimum_required(VERSION 3.18)
project(DGEMM_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(includes)

# Source files
set(SOURCES
    src/dgemm_cuda.cu
    src/matrix_mul_kernel1.cu
    src/matrix_mul_kernel2.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set CUDA architectures (adjust for your GPU)
set(CUDA_ARCHITECTURES 80)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)

# Installation (optional)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

cmake_minimum_required(VERSION 3.18)
project(DGEMM_CUDA LANGUAGES CXX CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(includes)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    dgemm_cuda.cpp
    cudafun/matrix_mul_kernel1.cu
    cudafun/matrix_mul_kernel2.cu
    cudafun/matrix_mul_kernel3.cu
    cudafun/matrix_mul_kernel4.cu
    cudafun/run_dgemm.cu
    utility/print_summary.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set CUDA architectures (adjust for your GPU)
set(CUDA_ARCHITECTURES 80)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}"
)

# Link against CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

# Installation (optional)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)


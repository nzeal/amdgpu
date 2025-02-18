cmake_minimum_required(VERSION 3.20.4)
project(xGEMM)

# ----------------------------------------------------------------SET CXX STANDARD--
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------------------------------INCLUDES--
include(GNUInstallDirs)

# Set the path to ROCm if it's not already set
if (NOT DEFINED HIP_PATH)
  if (NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH "/opt/rocm-6.0.3" CACHE PATH "HIP path")
  else()
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "HIP path")
  endif()
endif()

# Set the paths manually, since FindHIP.cmake is missing
set(HIP_INCLUDE_DIR "${HIP_PATH}/include")
set(HIP_LIBRARIES "${HIP_PATH}/lib")
set(HIP_BIN_DIR "${HIP_PATH}/bin")

# Ensure HIP paths are set correctly
if (NOT EXISTS "${HIP_INCLUDE_DIR}")
  message(FATAL_ERROR "HIP_INCLUDE_DIR not found: ${HIP_INCLUDE_DIR}")
endif()

if (NOT EXISTS "${HIP_LIBRARIES}")
  message(FATAL_ERROR "HIP_LIBRARIES not found: ${HIP_LIBRARIES}")
endif()

# Include hipblas (this should be found if HIP is properly set up)
# Fixed path to HIPBLAS include directory
set(HIPBLAS_INCLUDE_DIR "/opt/rocm-6.0.3/include/hipblas")
set(HIPBLAS_LIBRARIES "/opt/rocm-6.0.3/lib/libhipblas.so")  # Correct path to the hipblas library

if (NOT EXISTS "${HIPBLAS_INCLUDE_DIR}")
  message(FATAL_ERROR "HIPBLAS_INCLUDE_DIR not found: ${HIPBLAS_INCLUDE_DIR}")
endif()

if (NOT EXISTS "${HIPBLAS_LIBRARIES}")
  message(FATAL_ERROR "HIPBLAS_LIBRARIES not found: ${HIPBLAS_LIBRARIES}")
endif()

# Path to rocblas include directory
set(ROCBLAS_INCLUDE_DIR "/opt/rocm-6.0.3/include/rocblas")

if (NOT EXISTS "${ROCBLAS_INCLUDE_DIR}")
  message(FATAL_ERROR "ROCBLAS_INCLUDE_DIR not found: ${ROCBLAS_INCLUDE_DIR}")
endif()

# Set the compiler to hipcc
set(CMAKE_CXX_COMPILER /usr/bin/hipcc)
message(STATUS "CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")

# Compiler flags
set(xDGEMM_HIP_FLAGS "-fno-gpu-rdc -munsafe-fp-atomics -fPIC -w" CACHE STRING "Compiler flags for HIP")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${xDGEMM_HIP_FLAGS}" CACHE STRING "Flags used by the CXX compiler during all build types." FORCE)

# Specify the executable name and source files
add_executable(rocblas_dgemm rocblas_dgemm.cpp)

# Include the directories for HIP, hipblas, and rocblas
target_include_directories(rocblas_dgemm PRIVATE ${HIP_INCLUDE_DIR} ${HIPBLAS_INCLUDE_DIR} ${ROCBLAS_INCLUDE_DIR})

# Link the libraries for HIP, hipblas, and rocblas
# Add specific library files, not just directories
target_link_libraries(rocblas_dgemm PRIVATE 
    ${HIP_LIBRARIES} 
    ${HIPBLAS_LIBRARIES}
    /opt/rocm-6.0.3/lib/librocblas.so  # Link to the specific rocblas library
)

# Set the executable output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


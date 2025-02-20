#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

#include "includes/error_checking.h"
#include "includes/performance_result.h"
#include "includes/performance_utils.h"
#include "includes/matrix_mul_kernel.h"
#include "includes/dgemm_functions.h"
#include "includes/kernel_registry.h"
#include "includes/cuda_version_check.h"

// Function to print device information
void printDeviceInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming you're using the first device (0)
    printf("=== CUDA Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

int main() {
    // Initialize CUDA and check versions
    if (!initializeCuda()) {
        fprintf(stderr, "Failed to initialize CUDA. Exiting.\n");
        return 1;
    }

    // Register kernels only if CUDA initialization succeeded
    auto& registry = KernelRegistry::getInstance();
    registry.registerKernel("Kernel1", matrixMulKernel1);
    registry.registerKernel("Kernel2", matrixMulKernel2);
    registry.registerKernel("Kernel3", matrixMulKernel3);
    registry.registerKernel("Kernel4", matrixMulKernel4);
    registry.registerKernel("Kernel5", matrixMulKernel5);

    std::vector<int> sizes = {1024}; // Add more sizes as needed
    std::vector<PerformanceResult> results;

    // Run tests
    for (int size : sizes) {
        try {
            runDGEMM(size, results);
        } catch (const std::exception& e) {
            std::cerr << "Error testing size " << size << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Print final summary if we have results
    if (!results.empty()) {
        printSummary(results);
    }

    return 0;
}
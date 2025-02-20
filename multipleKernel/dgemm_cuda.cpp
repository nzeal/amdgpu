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

int main() {
    // Add this debug code at the beginning of main
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    std::vector<int> sizes = {1024};
    std::vector<PerformanceResult> results;

    // Print device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== CUDA Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Run tests
    for (int size : sizes) {
        try {
            runDGEMM(size, results);
        } catch (const std::exception& e) {
            std::cerr << "Error testing size " << size << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Print final summary
    printSummary(results);

    return 0;
}



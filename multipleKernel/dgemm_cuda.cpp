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

// Declaration of runDGEMM from run_dgemm.cu with kernel_id parameter
extern "C" void runDGEMM(int size, int kernel_id, std::vector<PerformanceResult>& results);

int main(int argc, char *argv[]) {
    // Parse command-line argument for kernel_id
    int kernel_id = 0; // Default: invalid, will trigger usage message if not overridden
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel_id> [size]\n";
        std::cerr << "  <kernel_id>: 1-5 to run a specific kernel, or 0 to run all (not supported for perf stat)\n";
        std::cerr << "  [size]: Optional matrix size (default: 1024)\n";
        return 1;
    }

    kernel_id = atoi(argv[1]);
    if (kernel_id < 1 || kernel_id > 5) {
        std::cerr << "Invalid kernel_id: " << kernel_id << ". Must be 1-5 for perf stat.\n";
        return 1;
    }

    int size = 1024; // Default size
    if (argc > 2) {
        size = atoi(argv[2]);
        if (size <= 0) {
            std::cerr << "Invalid size: " << size << ". Must be positive.\n";
            return 1;
        }
    }

    // Debug code
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "CUDA Driver Version: " << driverVersion << std::endl;
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    std::vector<PerformanceResult> results;

    // Print device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== CUDA Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Run test for the specified kernel
    try {
        runDGEMM(size, kernel_id, results);
    } catch (const std::exception& e) {
        std::cerr << "Error testing kernel " << kernel_id << " with size " << size << ": " << e.what() << std::endl;
        return 1;
    }

    // Print final summary
    printDetailedKernelStats(results);

    return 0;
}

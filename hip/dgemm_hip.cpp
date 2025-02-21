#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "./includes/error_checking.h"
#include "./includes/performance_result.h"
#include "./includes/performance_utils.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>

__global__ void matrixMulKernel(const double *A, const double *B, double *C, 
                               int M, int N, int K, double alpha, double beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N)
    {
        double acc_c = 0.0;
        for (int k = 0; k < K; ++k)
        {
            acc_c += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc_c + beta * C[row * N + col];
    }
}

void runDGEMM(int size, std::vector<PerformanceResult>& results) {
    const int m = size;
    const int k = size;
    const int n = size;
    const double alpha = 1.0;
    const double beta = 0.0;

    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);
    size_t total_memory = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printf("\nMatrix size: %d x %d (%zu MB)\n", size, size, total_memory);

    // Error checking for memory allocation
    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);

    if (!h_A || !h_B || !h_C) {
        throw std::runtime_error("Host memory allocation failed");
    }

    // Initialize matrices
    for(int i = 0; i < m * k; i++) h_A[i] = 1.0;
    for(int i = 0; i < k * n; i++) h_B[i] = 2.0;
    for(int i = 0; i < m * n; i++) h_C[i] = 0.0;

    // Allocate device memory with error checking
    double *d_A, *d_B, *d_C;
    if (hipMalloc(&d_A, size_A) != hipSuccess ||
        hipMalloc(&d_B, size_B) != hipSuccess ||
        hipMalloc(&d_C, size_C) != hipSuccess) {
        free(h_A); free(h_B); free(h_C);
        throw std::runtime_error("Device memory allocation failed");
    }

    PerformanceResult result;
    result.size = size;

    // Transfer data to device with error checking
    hipDeviceSynchronize();
    auto transfer_start = getCurrentTime();
    
    if (hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        free(h_A); free(h_B); free(h_C);
        throw std::runtime_error("Host to device transfer failed");
    }
    
    hipDeviceSynchronize();
    auto transfer_end = getCurrentTime();

    result.transfer_to_device_time = calculateDurationInSeconds(transfer_start, transfer_end);
    result.bandwidth_to_device = ((size_A + size_B) / (1024.0 * 1024.0 * 1024.0)) / result.transfer_to_device_time;

    printf("H2D Transfer: %.2f ms (%.2f GB/s)\n",
           result.transfer_to_device_time * 1000,
           result.bandwidth_to_device);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel with error checking
    auto compute_start = getCurrentTime();
    hipLaunchKernelGGL(matrixMulKernel, numBlocks, threadsPerBlock, 0, 0, 
                       d_A, d_B, d_C, m, n, k, alpha, beta);
    
    hipError_t kernel_error = hipGetLastError();
    if (kernel_error != hipSuccess) {
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        free(h_A); free(h_B); free(h_C);
        throw std::runtime_error("Kernel launch failed: " + 
                               std::string(hipGetErrorString(kernel_error)));
    }
    
    hipDeviceSynchronize();
    auto compute_end = getCurrentTime();

    result.computation_time = calculateDurationInSeconds(compute_start, compute_end);
    double flops = 2.0 * m * n * k;  // Each multiply-add is 2 operations
    result.gflops = (result.computation_time > 0) ? 
                    (flops / (result.computation_time * 1e9)) : 0.0;

    printf("Computation: %.2f ms (%.2f GFLOPS)\n",
           result.computation_time * 1000,
           result.gflops);

    // Transfer result back with error checking
    hipDeviceSynchronize();
    auto transfer_back_start = getCurrentTime();
    
    if (hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost) != hipSuccess) {
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        free(h_A); free(h_B); free(h_C);
        throw std::runtime_error("Device to host transfer failed");
    }
    
    hipDeviceSynchronize();
    auto transfer_back_end = getCurrentTime();

    result.transfer_from_device_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
    result.bandwidth_from_device = (size_C / (1024.0 * 1024.0 * 1024.0)) / result.transfer_from_device_time;

    printf("D2H Transfer: %.2f ms (%.2f GB/s)\n",
           result.transfer_from_device_time * 1000,
           result.bandwidth_from_device);

    // Result verification
    double expected_value = k * alpha * 2.0;  // Updated for alpha parameter
    bool verification_failed = false;
    
    // Check more elements for verification
    for (int i = 0; i < std::min(10, m * n); i++) {
        if (std::abs(h_C[i] - expected_value) > 1e-5) {
            verification_failed = true;
            break;
        }
    }
    
    if (verification_failed) {
        printf("Verification FAILED: Expected %.2f, Got %.2f (first element)\n",
               expected_value, h_C[0]);
    } else {
        printf("Verification PASSED\n");
    }

    results.push_back(result);

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

void printSummary(const std::vector<PerformanceResult>& results) {
    printf("\n=== Performance Summary ===\n");
    printf("%8s %12s %15s %15s %15s\n",
           "Size", "GFLOPS", "Compute(ms)", "H2D BW(GB/s)", "D2H BW(GB/s)");
    printf("%s\n", std::string(70, '-').c_str());

    for (const auto& result : results) {
        printf("%8d %12.2f %15.2f %15.2f %15.2f\n",
               result.size,
               result.gflops,
               result.computation_time * 1000,
               result.bandwidth_to_device,
               result.bandwidth_from_device);
    }
}

int main() {
    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    std::vector<PerformanceResult> results;

    // Print device information
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        std::cerr << "Failed to get device properties" << std::endl;
        return 1;
    }

    printf("=== HIP Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Global Memory: %.2f GB\n\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

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

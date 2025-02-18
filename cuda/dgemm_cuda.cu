#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "./includes/error_checking.h"
#include "./includes/performance_result.h"
#include "./includes/performance_utils.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

__global__ void matrixMulKernel(const double *A, const double *B, double *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void runDGEMM(int size, std::vector<PerformanceResult>& results) {
    const int m = size;
    const int k = size;
    const int n = size;

    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);
    size_t total_memory = (size_A + size_B + size_C) / (1024.0 * 1024.0);

    printf("\nMatrix size: %d x %d (%zu MB)\n", size, size, total_memory);

    // Allocate host memory
    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);

    // Initialize matrices
    for(int i = 0; i < m * k; i++) h_A[i] = 1.0;
    for(int i = 0; i < k * n; i++) h_B[i] = 2.0;
    for(int i = 0; i < m * n; i++) h_C[i] = 0.0;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    PerformanceResult result;
    result.size = size;

    // Transfer data to device
    cudaDeviceSynchronize();
    auto transfer_start = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    auto transfer_end = getCurrentTime();

    result.transfer_to_device_time = calculateDurationInSeconds(transfer_start, transfer_end);
    result.bandwidth_to_device = ((size_A + size_B) / (1024.0 * 1024.0 * 1024.0)) / result.transfer_to_device_time;

    printf("H2D Transfer: %.2f ms (%.2f GB/s)\n",
           result.transfer_to_device_time * 1000,
           result.bandwidth_to_device);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Number of blocks

    // Launch the matrix multiplication kernel
    auto compute_start = getCurrentTime();
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    auto compute_end = getCurrentTime();

    result.computation_time = calculateDurationInSeconds(compute_start, compute_end);
    double flops = 2.0 * m * n * k;
    result.gflops = (result.computation_time > 0) ? (flops / (result.computation_time * 1e9)) : 0.0;

    printf("Computation: %.2f ms (%.2f GFLOPS)\n",
           result.computation_time * 1000,
           result.gflops);

    // Transfer result back to host
    cudaDeviceSynchronize();
    auto transfer_back_start = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    auto transfer_back_end = getCurrentTime();

    result.transfer_from_device_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
    result.bandwidth_from_device = (size_C / (1024.0 * 1024.0 * 1024.0)) / result.transfer_from_device_time;

    printf("D2H Transfer: %.2f ms (%.2f GB/s)\n",
           result.transfer_from_device_time * 1000,
           result.bandwidth_from_device);

    // Result verification
    double expected_value = k * 2.0;
    if (std::abs(h_C[0] - expected_value) > 1e-5 ||
        std::abs(h_C[m * n - 1] - expected_value) > 1e-5) {
        printf("Verification FAILED: Expected %.2f, Got %.2f (first) %.2f (last)\n",
               expected_value, h_C[0], h_C[m * n - 1]);
    }

    results.push_back(result);

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
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
    std::vector<int> sizes = {1024};
   // {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
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

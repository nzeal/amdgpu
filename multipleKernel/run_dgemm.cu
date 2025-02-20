#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "../includes/error_checking.h"
#include "../includes/performance_result.h"
#include "../includes/performance_utils.h"
#include "../includes/matrix_mul_kernel.h"
#include "../includes/dgemm_functions.h"
#include "kernel_registry.h"
#include "performance_result.h"

// Function to allocate memory and initialize matrices
void allocateAndInitializeMatrices(double **h_A, double **h_B, double **h_C, double **d_A, double **d_B, double **d_C, int m, int k, int n) {
    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);

    // Allocate host memory
    *h_A = (double*)malloc(size_A);
    *h_B = (double*)malloc(size_B);
    *h_C = (double*)malloc(size_C);

    // Initialize matrices
    for(int i = 0; i < m * k; i++) (*h_A)[i] = 1.0;
    for(int i = 0; i < k * n; i++) (*h_B)[i] = 2.0;
    for(int i = 0; i < m * n; i++) (*h_C)[i] = 0.0;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(d_A, size_A));
    CHECK_CUDA(cudaMalloc(d_B, size_B));
    CHECK_CUDA(cudaMalloc(d_C, size_C));
}

// Function to transfer data to device
void transferDataToDevice(double *h_A, double *h_B, double *d_A, double *d_B, int m, int k, int n, PerformanceResult &result) {
    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);

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
}

// Function to run a kernel and measure its performance
void runKernel(void (*kernel)(const double*, const double*, double*, int, int, int, double, double), 
               const double *d_A, const double *d_B, double *d_C, 
               int m, int n, int k, dim3 numBlocks, dim3 threadsPerBlock, 
               const char* kernelName, KernelResult &kernel_result, double alpha, double beta) {
    auto compute_start = getCurrentTime();
    kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k, alpha, beta);
    cudaDeviceSynchronize();
    auto compute_end = getCurrentTime();

    kernel_result.computation_time = calculateDurationInSeconds(compute_start, compute_end);
    double flops = 2.0 * m * n * k;
    kernel_result.gflops = (kernel_result.computation_time > 0) ? 
        (flops / (kernel_result.computation_time * 1e9)) : 0.0;

    printf("%s Computation: %.2f ms (%.2f GFLOPS)\n",
           kernelName, kernel_result.computation_time * 1000, kernel_result.gflops);
}

// Function to transfer data from device to host
void transferDataFromDevice(double *h_C, double *d_C, int m, int n, 
    const char* kernelName, KernelResult &kernel_result) {
size_t size_C = m * n * sizeof(double);

cudaDeviceSynchronize();
auto transfer_back_start = getCurrentTime();
CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
cudaDeviceSynchronize();
auto transfer_back_end = getCurrentTime();

kernel_result.transfer_from_device_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
kernel_result.bandwidth_from_device = (size_C / (1024.0 * 1024.0 * 1024.0)) / 
                  kernel_result.transfer_from_device_time;

printf("D2H Transfer %s: %.2f ms (%.2f GB/s)\n", 
kernelName, 
kernel_result.transfer_from_device_time * 1000, 
kernel_result.bandwidth_from_device);
}

// Updated verifyResults function
void verifyResults(const double *h_A, const double *h_B, double *h_C, int m, int n, int k, double alpha, double beta) {
    double epsilon = 1e-6;  // Tolerance for floating-point comparison
    bool correct = true;
    int errors = 0;
    const int max_errors_to_print = 10;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = 0.0;
            for (int p = 0; p < k; ++p) {
                expected += h_A[i * k + p] * h_B[p * n + j];
            }
            expected = alpha * expected + beta * h_C[i * n + j];

            double actual = h_C[i * n + j];
            if (std::abs(expected - actual) > epsilon) {
                if (errors < max_errors_to_print) {
                    printf("Error at position (%d, %d): Expected %.8f, Got %.8f\n", i, j, expected, actual);
                }
                errors++;
                correct = false;
            }
        }
    }

    if (correct) {
        printf("Results verified: CORRECT\n");
    } else {
        printf("Results verified: INCORRECT. %d errors found.\n", errors);
    }
}

void runDGEMM(int size, std::vector<PerformanceResult>& results) {
    const int m = size;
    const int k = size;
    const int n = size;

    size_t total_memory = (m * k + k * n + m * n) * sizeof(double) / (1024.0 * 1024.0);
    printf("\nMatrix size: %d x %d (%zu MB)\n", size, size, total_memory);

    double *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    allocateAndInitializeMatrices(&h_A, &h_B, &h_C, &d_A, &d_B, &d_C, m, k, n);

    PerformanceResult result;
    result.size = size;

    transferDataToDevice(h_A, h_B, d_A, d_B, m, k, n, result);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    double alpha = 1.0;
    double beta = 0.0;

    // Get all registered kernels
    const auto& kernels = KernelRegistry::getInstance().getKernels();

    // Run each kernel
    for (const auto& kernel : kernels) {
        printf("----------------------------------------------- %s\n", kernel.name.c_str());
        
        // Reset C matrix
        for(int i = 0; i < m * n; i++) h_C[i] = 0.0;
        CHECK_CUDA(cudaMemcpy(d_C, h_C, m * n * sizeof(double), cudaMemcpyHostToDevice));

        KernelResult kernel_result;
        kernel_result.kernel_name = kernel.name;

        // Run kernel and measure performance
        runKernel(kernel.function, d_A, d_B, d_C, m, n, k, 
                 numBlocks, threadsPerBlock, kernel.name.c_str(), 
                 kernel_result, alpha, beta);

        // Transfer and measure results
        transferDataFromDevice(h_C, d_C, m, n, kernel.name.c_str(), kernel_result);

        // Verify results
        printf("Verifying %s results:\n", kernel.name.c_str());
        verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

        // Store results
        result.kernel_results.push_back(kernel_result);
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
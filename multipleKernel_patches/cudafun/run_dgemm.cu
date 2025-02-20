#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "../includes/error_checking.h"
#include "../includes/performance_result.h"
#include "../includes/performance_utils.h"
#include "../includes/matrix_mul_kernel.h"
#include "../includes/dgemm_functions.h"

// Function to allocate memory and initialize matrices
void allocateAndInitializeMatrices(double **h_A, double **h_B, double **h_C, double **d_A, double **d_B, double **d_C, int max_size) {
    size_t max_size_bytes = max_size * max_size * sizeof(double);

    // Allocate host memory
    *h_A = (double*)malloc(max_size_bytes);
    *h_B = (double*)malloc(max_size_bytes);
    *h_C = (double*)malloc(max_size_bytes);

    // Initialize matrices with some values (you can modify this as needed)
    for(int i = 0; i < max_size * max_size; i++) {
        (*h_A)[i] = 1.0;
        (*h_B)[i] = 2.0;
        (*h_C)[i] = 0.0;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(d_A, max_size_bytes));
    CHECK_CUDA(cudaMalloc(d_B, max_size_bytes));
    CHECK_CUDA(cudaMalloc(d_C, max_size_bytes));
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
               const char* kernelName, PerformanceResult &result, double alpha, double beta) {
    auto compute_start = getCurrentTime();
    kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k, alpha, beta);
    cudaDeviceSynchronize();
    auto compute_end = getCurrentTime();

    double computation_time = calculateDurationInSeconds(compute_start, compute_end);
    double flops = 2.0 * m * n * k;
    double gflops = (computation_time > 0) ? (flops / (computation_time * 1e9)) : 0.0;

    printf("%s Computation: %.2f ms (%.2f GFLOPS)\n",
           kernelName, computation_time * 1000, gflops);

    if (strcmp(kernelName, "Kernel 1") == 0) {
        result.computation_time_kernel1 = computation_time;
        result.gflops_kernel1 = gflops;
    } else {
        result.computation_time_kernel2 = computation_time;
        result.gflops_kernel2 = gflops;
    }
}

// Function to transfer data from device to host
void transferDataFromDevice(double *h_C, double *d_C, int m, int n, const char* kernelName, PerformanceResult &result) {
    size_t size_C = m * n * sizeof(double);

    cudaDeviceSynchronize();
    auto transfer_back_start = getCurrentTime();
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    auto transfer_back_end = getCurrentTime();

    double transfer_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
    double bandwidth = (size_C / (1024.0 * 1024.0 * 1024.0)) / transfer_time;

    printf("D2H Transfer %s: %.2f ms (%.2f GB/s)\n", kernelName, transfer_time * 1000, bandwidth);

    if (strcmp(kernelName, "Kernel 1") == 0) {
        result.transfer_from_device_time_kernel1 = transfer_time;
        result.bandwidth_from_device_kernel1 = bandwidth;
    } else {
        result.transfer_from_device_time_kernel2 = transfer_time;
        result.bandwidth_from_device_kernel2 = bandwidth;
    }
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

void runDGEMM(int size, std::vector<PerformanceResult>& results, double *h_A, double *h_B, double *h_C, double *d_A, double *d_B, double *d_C) {
    const int m = size;
    const int k = size;
    const int n = size;

    size_t total_memory = (m * k + k * n + m * n) * sizeof(double) / (1024.0 * 1024.0);
    printf("\nMatrix size: %d x %d (%zu MB)\n", size, size, total_memory);

    PerformanceResult result;
    result.size = size;

    transferDataToDevice(h_A, h_B, d_A, d_B, m, k, n, result);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    double alpha = 1.0;
    double beta = 0.0;

    // Run and verify Kernel 1
    runKernel(matrixMulKernel1, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 1", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 1", result);
    printf("Verifying Kernel 1 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    // Reset h_C to initial values (all zeros)
    for(int i = 0; i < m * n; i++) h_C[i] = 0.0;
    CHECK_CUDA(cudaMemcpy(d_C, h_C, m * n * sizeof(double), cudaMemcpyHostToDevice));

    // Run and verify Kernel 2
    runKernel(matrixMulKernel2, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 2", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 2", result);
    printf("Verifying Kernel 2 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    results.push_back(result);
}

void runAllDGEMM(const std::vector<int>& sizes, std::vector<PerformanceResult>& results) {
    // Find the maximum size to allocate memory
    int max_size = *std::max_element(sizes.begin(), sizes.end());

    double *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    allocateAndInitializeMatrices(&h_A, &h_B, &h_C, &d_A, &d_B, &d_C, max_size);

    for (int size : sizes) {
        try {
            runDGEMM(size, results, h_A, h_B, h_C, d_A, d_B, d_C);
        } catch (const std::exception& e) {
            std::cerr << "Error testing size " << size << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}
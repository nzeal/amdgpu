#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "includes/error_checking.h"
#include "includes/performance_result.h"
#include "includes/performance_utils.h"
#include "includes/matrix_mul_kernel.h"
#include "includes/dgemm_functions.h"
#include "includes/matrix_management.h"
#include "includes/kernel_runner.h"

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

    // Transfer data to device
    transferDataToDevice(h_A, h_B, d_A, d_B, m, k, n, result);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    double alpha = 1.0;
    double beta = 0.0;

    // Run and verify Kernel 1
    printf("----------------------------------------------- Kernel-1\n");
    runKernel(matrixMulKernel1, d_A, d_B, d_C,
              m, n, k, numBlocks, threadsPerBlock,
              "Kernel 1", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 1", result);
    printf("Verifying Kernel 1 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    // Reset h_C to initial values (all zeros)
    for (int i = 0; i < m * n; i++) h_C[i] = 0.0;
    CHECK_HIP(hipMemcpy(d_C, h_C, m * n * sizeof(double), hipMemcpyHostToDevice));

  // Run and verify Kernel 2
    printf("----------------------------------------------- Kernel-2\n");
    runKernel(matrixMulKernel2, d_A, d_B, d_C,
              m, n, k, numBlocks, threadsPerBlock,
              "Kernel 2", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 2", result);
    printf("Verifying Kernel 2 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    // Run and verify Kernel 3
    printf("----------------------------------------------- Kernel-3\n");
    runKernel(matrixMulKernel3, d_A, d_B, d_C,
              m, n, k, numBlocks, threadsPerBlock,
              "Kernel 3", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 3", result);
    printf("Verifying Kernel 3 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    // Run and verify Kernel 4
    printf("----------------------------------------------- Kernel-4\n");
    runKernel(matrixMulKernel4, d_A, d_B, d_C,
              m, n, k, numBlocks, threadsPerBlock,
              "Kernel 4", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 4", result);
    printf("Verifying Kernel 4 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    // Run and verify Kernel 5
    printf("----------------------------------------------- Kernel-5\n");
    runKernel(matrixMulKernel5, d_A, d_B, d_C,
              m, n, k, numBlocks, threadsPerBlock,
              "Kernel 5", result, alpha, beta);
    transferDataFromDevice(h_C, d_C, m, n, "Kernel 5", result);
    printf("Verifying Kernel 5 results:\n");
    verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);

    // Add the result to the results vector
    results.push_back(result);

    // Print detailed kernel performance stats
    //printDetailedKernelStats(results);
    

    // Cleanup
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}


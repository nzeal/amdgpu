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
#include "../includes/matrix_management.h"
#include "../includes/kernel_runner.h"

// Define constants matching matrixMulKernel5.cu
#define BLOCK_SIZE 256
#define BN 128
#define BM 128

void runDGEMM(int size, int kernel_id, std::vector<PerformanceResult>& results) {
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

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (kernel_id == 5) {
        threadsPerBlock = dim3(BLOCK_SIZE); // 256 threads
        numBlocks = dim3((n + BN - 1) / BN, (m + BM - 1) / BM);
    } else {
        threadsPerBlock = dim3(32, 32);
        numBlocks = dim3((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    }

    double alpha = 1.0;
    double beta = 0.0;

    // Run the specified kernel
    switch (kernel_id) {
        case 1:
            printf("----------------------------------------------- Kernel-1\n");
            runKernel(matrixMulKernel1, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 1", result, alpha, beta);
            transferDataFromDevice(h_C, d_C, m, n, "Kernel 1", result);
            printf("Verifying Kernel 1 results:\n");
            verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);
            break;

        case 2:
            printf("----------------------------------------------- Kernel-2\n");
            runKernel(matrixMulKernel2, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 2", result, alpha, beta);
            transferDataFromDevice(h_C, d_C, m, n, "Kernel 2", result);
            printf("Verifying Kernel 2 results:\n");
            verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);
            break;

        case 3:
            printf("----------------------------------------------- Kernel-3\n");
            runKernel(matrixMulKernel3, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 3", result, alpha, beta);
            transferDataFromDevice(h_C, d_C, m, n, "Kernel 3", result);
            printf("Verifying Kernel 3 results:\n");
            verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);
            break;

        case 4:
            printf("----------------------------------------------- Kernel-4\n");
            runKernel(matrixMulKernel4, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 4", result, alpha, beta);
            transferDataFromDevice(h_C, d_C, m, n, "Kernel 4", result);
            printf("Verifying Kernel 4 results:\n");
            verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);
            break;

        case 5:
            printf("----------------------------------------------- Kernel-5\n");
            runKernel(matrixMulKernel5, d_A, d_B, d_C, m, n, k, numBlocks, threadsPerBlock, "Kernel 5", result, alpha, beta);
            transferDataFromDevice(h_C, d_C, m, n, "Kernel 5", result);
            printf("Verifying Kernel 5 results:\n");
            verifyResults(h_A, h_B, h_C, m, n, k, alpha, beta);
            break;

        default:
            printf("Invalid kernel ID: %d. Please use 1-5.\n", kernel_id);
            return;
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

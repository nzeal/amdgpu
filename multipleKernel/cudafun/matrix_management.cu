// matrix_management.cu

#include <cuda_runtime.h>
#include "../includes/error_checking.h"
#include "../includes/performance_result.h"
#include "../includes/performance_utils.h"
#include "../includes/matrix_management.h"

// Function to allocate memory and initialize matrices
void allocateAndInitializeMatrices(double **h_A, double **h_B, double **h_C, 
    double **d_A, double **d_B, double **d_C, int m, int k, int n) 
    {
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

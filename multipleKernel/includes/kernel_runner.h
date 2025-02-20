// kernel_runner.h
#pragma once
#include <cuda_runtime.h>
#include "../includes/performance_result.h"
#include "../includes/matrix_mul_kernel.h"

void runKernel(void (*kernel)(const double*, const double*, double*, int, int, int, double, double),
              const double* d_A, const double* d_B, double* d_C,
              int m, int n, int k, dim3 numBlocks, dim3 threadsPerBlock,
              const char* kernelName, PerformanceResult& result, double alpha, double beta);

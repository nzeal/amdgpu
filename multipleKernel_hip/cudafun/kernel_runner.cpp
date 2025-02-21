// kernel_runner.cpp
#include "../includes/kernel_runner.h"
#include "../includes/performance_utils.h"
#include <hip/hip_runtime.h>

void runKernel(void (*kernel)(const double*, const double*, double*, int, int, int, double, double),
                        const double* d_A, const double* d_B, double* d_C,
                        int m, int n, int k, dim3 numBlocks, dim3 threadsPerBlock,
                        const char* kernelName, PerformanceResult& result, double alpha, double beta) {
       auto compute_start = getCurrentTime();
       hipLaunchKernelGGL(kernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, m, n, k, alpha, beta);
       hipDeviceSynchronize();
       auto compute_end = getCurrentTime();

       double computation_time = calculateDurationInSeconds(compute_start, compute_end);
       double flops = 2.0 * m * n * k;
       double gflops = (computation_time > 0) ? (flops / (computation_time * 1e9)) : 0.0;

       printf("%s Computation: %.2f ms (%.2f GFLOPS)\n",
                 kernelName, computation_time * 1000, gflops);

       // Add a new kernel result
       result.addKernelResult(kernelName, computation_time, gflops, 0.0, 0.0);
}



#include "../includes/matrix_mul_kernel.h"

__global__ void matrixMulKernel2(const double *A, const double *B, double *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // maps to i
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // maps to j

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


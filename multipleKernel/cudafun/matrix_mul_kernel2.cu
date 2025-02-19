#include "../includes/matrix_mul_kernel.h"

__global__ void matrixMulKernel2(const double *A, const double *B, double *C, int M, int N, int K, double alpha, double beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double acc_c = 0.0;
        for (int k = 0; k < K; k++) {
            acc_c += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc_c + beta * C[row * N + col];
    }
}


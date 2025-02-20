// matrix_mul_kernel2.cu - Optimized with shared memory
#include "../includes/matrix_mul_kernel.h"

__global__ void matrixMulKernel3(const double *A, const double *B, double *C, 
                                int M, int N, int K, double alpha, double beta) {
    const int TILE_SIZE = 32;
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && A_col < K) ? A[row*K + A_col] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (B_row < K && col < N) ? B[B_row*N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = alpha * sum + beta * C[row*N + col];
    }
}

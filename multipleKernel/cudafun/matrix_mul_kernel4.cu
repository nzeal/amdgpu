#include "../includes/matrix_mul_kernel.h"

__global__ void matrixMulKernel4(const double *A, const double *B, double *C, 
    int M, int N, int K, double alpha, double beta) {
// Define shared memory tiles - we'll still use 32x32 tiles to match thread blocks
const int TILE_SIZE = 32;
__shared__ double As[TILE_SIZE][TILE_SIZE];
__shared__ double Bs[TILE_SIZE][TILE_SIZE];

// Calculate global indices
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int ty = threadIdx.y;
const int tx = threadIdx.x;

// Initialize accumulator
double acc_c = 0.0;

// Loop over tiles
const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

for (int tile = 0; tile < numTiles; tile++) {
// Calculate the starting point for this tile
const int tileK = tile * TILE_SIZE;

// Load data into shared memory with bounds checking
// Each thread loads one element
if (row < M && (tileK + tx) < K) {
As[ty][tx] = A[row * K + tileK + tx];
} else {
As[ty][tx] = 0.0;
}

if ((tileK + ty) < K && col < N) {
Bs[ty][tx] = B[(tileK + ty) * N + col];
} else {
Bs[ty][tx] = 0.0;
}

// Make sure all threads have loaded their data
__syncthreads();

// Compute partial dot product for this tile
if (row < M && col < N) {
// Only go up to the actual K dimension or tile size, whichever is smaller
const int kMax = min(TILE_SIZE, K - tileK);
for (int k = 0; k < kMax; k++) {
acc_c += As[ty][k] * Bs[k][tx];
}
}

// Synchronize before loading the next tile
__syncthreads();
}

// Write final result to global memory
if (row < M && col < N) {
C[row * N + col] = alpha * acc_c + beta * C[row * N + col];
}
}

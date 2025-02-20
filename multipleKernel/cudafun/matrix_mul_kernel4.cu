/* -----------------------------------------------------------------------------------------Ref--
* Sebastien: sev-v
* --------------------------------------------------------------------------------------------*/

#include "../includes/matrix_mul_kernel.h"

#define BLOCK_SIZE 256

__global__ void matrixMulKernel4(const double *A, const double *B, double *C,
                                int M, int N, int K, double alpha, double beta) {
// --------------------------------------------------------------------------- Block Tile Size
   /* These constants define the size of the tiles used for matrix blocks. 
      The tiles help to divide the large matrices into smaller chunks, 
      which are loaded into shared memory for efficient access. */

    constexpr int BN = 128;
    constexpr int BM = 128;
    constexpr int BK = 8;    // Number of Row or column we read per batch

/* ---------------------------------------------------------------------------- Thread Tile size
   The kernel organizes threads into "waves", which are groups of 32 threads. 
   This part helps assign each thread to work on a specific sub-task within the block.  
/ --------------------------------------------------------------------------------------------*/
    
    constexpr int TN = 4;
    constexpr int TM = 4;
    constexpr int nbWaves = BLOCK_SIZE / 32;
    // Wave Tile size
    constexpr int WN = 64;
    constexpr int WM = (BN * BM) / (nbWaves * WN);

    // Number of wave on X & Y axis in the Block tile
    constexpr int nbWaveX = BN / WN;
    constexpr int nbWaveY = BM / WM;

    const int waveIndex = threadIdx.x / 32;
    const int waveIdx = waveIndex % nbWaveX;
    const int waveIdy = waveIndex / nbWaveX;
    const int indexInWave = threadIdx.x % 32;

    // A wave is a block of 8x4 of the output matrix
    constexpr int nbThreadXPerWave = 8;
    constexpr int nbThreadYPerWave = 4;

    // Thread coordinates in Wave
    const int idxInWave = indexInWave % nbThreadXPerWave;
    const int idyInWave = indexInWave / nbThreadXPerWave;

    constexpr int nbIterWaveN = WN / (nbThreadXPerWave * TN);
    constexpr int nbIterWaveM = WM / (nbThreadYPerWave * TM);

    // Wave Sub-tile size
    constexpr int SUBWN = WN / nbIterWaveN;
    constexpr int SUBWM = WM / nbIterWaveM;

    // Thread mapping to read BKxBN block from A
    int rAIdx = threadIdx.x % BK;
    int rAIdy = threadIdx.x / BK;
    // Thread mapping to read BNxBK block from B
    int rBIdx = threadIdx.x % BN;
    int rBIdy = threadIdx.x / BN;

    constexpr int strideReadB = BLOCK_SIZE / BN;
    constexpr int strideReadA = BLOCK_SIZE / BK;
    constexpr int nbReadsB = (BN * BK) / BLOCK_SIZE;
    constexpr int nbReadsA = (BM * BK) / BLOCK_SIZE;

    double A_col[nbIterWaveM * TM];
    double B_row[nbIterWaveN * TN];

    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    double c_regs[TM * nbIterWaveM * TN * nbIterWaveN] = {0.0};

    // Iteration over BK blocks
    for (int kId = 0; kId < K; kId += BK) {
        // Populate shared memory with A and B tiles 
        // These loops load the data from global memory into shared memory.
        // Each thread computes a small portion of a matrix (a tile) and loads it into shared memory.
        // This enables faster access to the matrix elements during computation.
        for (int i = 0; i < nbReadsB; i++) {
            int index_x = BN * blockIdx.x + rBIdx;
            int index_y = rBIdy + i * strideReadB + kId;
            if (index_y < K && index_x < N) {
                Bs[index_y % BK][index_x % BN] = B[index_y * N + index_x];
            } else {
                Bs[index_y % BK][index_x % BN] = 0.0;
            }
        }

        for (int i = 0; i < nbReadsA; i++) {
            int index_x = rAIdx + kId;
            int index_y = BM * blockIdx.y + rAIdy + i * strideReadA;
            if (index_x < K && index_y < M) {
                As[index_x % BK][index_y % BM] = A[index_y * K + index_x];
            } else {
                As[index_x % BK][index_y % BM] = 0.0;
            }
        }

        __syncthreads();

        // Main computation loop
        for (int k = 0; k < BK; k++) {
            // Load A and B tiles into registers
            for (int iterWave = 0; iterWave < nbIterWaveN; iterWave++) {
                for (int i = 0; i < TN; i++) {
                    int index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i;
                    B_row[iterWave * TN + i] = Bs[k][index % BN];
                }
            }

            for (int iterWave = 0; iterWave < nbIterWaveM; iterWave++) {
                for (int i = 0; i < TM; i++) {
                    int index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i;
                    A_col[iterWave * TM + i] = As[k][index % BM];
                }
            }

            // Accumulate results
            for (int iterWaveM = 0; iterWaveM < nbIterWaveM; iterWaveM++) {
                for (int iterWaveN = 0; iterWaveN < nbIterWaveN; iterWaveN++) {
                    for (int yt = 0; yt < TM; yt++) {
                        for (int xt = 0; xt < TN; xt++) {
                            const int regIndex = (iterWaveM * TM + yt) * (TN * nbIterWaveN) + 
                                                (iterWaveN * TN + xt);
                            c_regs[regIndex] += A_col[iterWaveM * TM + yt] * 
                                               B_row[iterWaveN * TN + xt];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store results to global memory
    for (int iterWaveM = 0; iterWaveM < nbIterWaveM; iterWaveM++) {
        for (int iterWaveN = 0; iterWaveN < nbIterWaveN; iterWaveN++) {
            int xOut = blockIdx.x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave;
            int yOut = blockIdx.y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave;
            
            for (int yt = 0; yt < TM; yt++) {
                for (int xt = 0; xt < TN; xt++) {
                    if ((yOut + yt) < M && (xOut + xt) < N) {
                        int indexC = (yOut + yt) * N + (xOut + xt);
                        C[indexC] = alpha * c_regs[(iterWaveM * TM + yt) * (TN * nbIterWaveN) + 
                                                  (iterWaveN * TN + xt)] + 
                                   beta * C[indexC];
                    }
                }
            }
        }
    }
}

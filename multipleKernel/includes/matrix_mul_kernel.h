#pragma once

#ifndef MATRIX_MUL_KERNEL_H
#define MATRIX_MUL_KERNEL_H

__global__ void matrixMulKernel1(const double *A, const double *B, double *C, int m, int n, int k);
__global__ void matrixMulKernel2(const double *A, const double *B, double *C, int M, int N, int K);
#endif // MATRIX_MUL_KERNEL_H

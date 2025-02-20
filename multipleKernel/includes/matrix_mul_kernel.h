#pragma once

__global__ void matrixMulKernel1(const double *A, const double *B, double *C, int M, int N, int K, double alpha, double beta);
__global__ void matrixMulKernel2(const double *A, const double *B, double *C, int M, int N, int K, double alpha, double beta);
__global__ void matrixMulKernel3(const double *A, const double *B, double *C, int M, int N, int K, double alpha, double beta); 
__global__ void matrixMulKernel4(const double *A, const double *B, double *C, int M, int N, int K, double alpha, double beta);

// matrix_management.h

#pragma once
#include "error_checking.h"
#include "performance_utils.h"
#include "performance_result.h"  // Add this line to include PerformanceResult definition


void allocateAndInitializeMatrices(double** h_A, double** h_B, double** h_C,
                                   double** d_A, double** d_B, double** d_C,
                                   int m, int k, int n);

void transferDataToDevice(double* h_A, double* h_B, double* d_A, double* d_B,
                         int m, int k, int n, PerformanceResult& result);

void transferDataFromDevice(double* h_C, double* d_C, int m, int n,
                           const char* kernelName, PerformanceResult& result);


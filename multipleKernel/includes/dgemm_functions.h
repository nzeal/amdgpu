#pragma once

#include <vector>
#include "performance_result.h"

void runDGEMM(int size, int kernel_id, std::vector<PerformanceResult>& results);
void printDetailedKernelStats(const std::vector<PerformanceResult>& results);
void verifyResults(const double *h_A, const double *h_B, double *h_C, 
                  int m, int n, int k, double alpha, double beta);

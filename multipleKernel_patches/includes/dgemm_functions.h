#ifndef DGEMM_FUNCTIONS_H
#define DGEMM_FUNCTIONS_H

#include <vector>
#include "performance_result.h"

void runDGEMM(int size, std::vector<PerformanceResult>& results, double *h_A, double *h_B, double *h_C, double *d_A, double *d_B, double *d_C);
void runAllDGEMM(const std::vector<int>& sizes, std::vector<PerformanceResult>& results);

#endif // DGEMM_FUNCTIONS_H
#pragma once

#include <vector>
#include "performance_result.h"

void runDGEMM(int size, std::vector<PerformanceResult>& results);
void printSummary(const std::vector<PerformanceResult>& results);

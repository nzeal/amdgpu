#ifndef PERFORMANCE_UTILS_H
#define PERFORMANCE_UTILS_H

#include <chrono>
#include <vector>
#include "performance_result.h"

std::chrono::high_resolution_clock::time_point getCurrentTime();
double calculateDurationInSeconds(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);

// Add this line
void printSummary(const std::vector<PerformanceResult>& results);

#endif // PERFORMANCE_UTILS_H
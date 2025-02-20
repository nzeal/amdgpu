#include <vector>
#include <string>
#include <iostream>
#include "../includes/performance_result.h"
#include "../includes/dgemm_functions.h"

void printSummary(const std::vector<PerformanceResult>& results) {
    printf("\n=== Performance Summary ===\n");
    
    // Print header
    printf("%-8s", "Size");
    for (const auto& kernel_result : results[0].kernel_results) {
        printf(" | %-20s", kernel_result.kernel_name.c_str());
    }
    printf("\n%s\n", std::string(8 + results[0].kernel_results.size() * 23, '-').c_str());

    // Print results for each size
    for (const auto& result : results) {
        printf("%-8d", result.size);
        for (const auto& kernel_result : result.kernel_results) {
            printf(" | %7.2f GFLOPS", kernel_result.gflops);
        }
        printf("\n");
    }
}
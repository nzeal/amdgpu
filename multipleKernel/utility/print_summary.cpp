#include <vector>
#include <string>
#include <iostream>
#include "../includes/performance_result.h"
#include "../includes/dgemm_functions.h"

void printSummary(const std::vector<PerformanceResult>& results) {
    printf("\n=== Performance Summary ===\n");
    printf("%8s %12s %15s %15s %15s %15s %15s\n",
           "Size", "GFLOPS1", "Compute1(ms)", "H2D BW1(GB/s)", "D2H BW1(GB/s)", "GFLOPS2", "Compute2(ms)");
    printf("%s\n", std::string(90, '-').c_str());

    for (const auto& result : results) {
        printf("%8d %12.2f %15.2f %15.2f %15.2f %12.2f %15.2f\n",
               result.size,
               result.gflops_kernel1,
               result.computation_time_kernel1 * 1000,
               result.bandwidth_to_device,
               result.bandwidth_from_device_kernel1,
               result.gflops_kernel2,
               result.computation_time_kernel2 * 1000);
    }
}


#include "../includes/performance_utils.h"
#include <iostream>
#include <iomanip>

std::chrono::high_resolution_clock::time_point getCurrentTime() {
    return std::chrono::high_resolution_clock::now();
}

double calculateDurationInSeconds(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

void printSummary(const std::vector<PerformanceResult>& results) {
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "Kernel 1 (GFLOPS)" 
              << std::setw(15) << "Kernel 2 (GFLOPS)" 
              << std::setw(20) << "H2D Bandwidth (GB/s)" 
              << std::setw(20) << "D2H Bandwidth 1 (GB/s)" 
              << std::setw(20) << "D2H Bandwidth 2 (GB/s)" 
              << std::endl;

    for (const auto& result : results) {
        std::cout << std::setw(10) << result.size
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gflops_kernel1
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gflops_kernel2
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.bandwidth_to_device
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.bandwidth_from_device_kernel1
                  << std::setw(20) << std::fixed << std::setprecision(2) << result.bandwidth_from_device_kernel2
                  << std::endl;
    }
}
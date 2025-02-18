#pragma once
#include <chrono>
#include <iostream>

// Function to get current time in seconds
inline double getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(now.time_since_epoch());
    return duration.count();
}

// Function to calculate duration between two time points
inline double calculateDurationInSeconds(double start, double end) {
    double duration = end - start;
    if (duration < 0) {
        std::cerr << "Warning: Negative duration detected: " << duration << " seconds" << std::endl;
        return 0.0;
    }
    return duration;
}

// Function to compute and print metrics
inline void computeAndPrintMetrics(double start, double end, size_t dataSize, size_t numOperations) {
    double duration = calculateDurationInSeconds(start, end);
    double gflops = (numOperations > 0 && duration > 0) ? 
                   (static_cast<double>(numOperations) / (duration * 1e9)) : 0.0;
    double bandwidth = (dataSize > 0 && duration > 0) ? 
                      (static_cast<double>(dataSize) / (duration * 1024 * 1024 * 1024)) : 0.0;
    
    printf("  Duration: %.2f ms\n", duration * 1000);
    if (numOperations > 0) printf("  GFLOPS: %.2f\n", gflops);
    if (dataSize > 0) printf("  Bandwidth: %.2f GB/s\n", bandwidth);
}

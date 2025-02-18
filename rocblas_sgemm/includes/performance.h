#pragma once
#include <chrono>
#include <iostream>

/**
 * @brief Get the current time point using high resolution clock.
 *
 * This function returns the current time point using the high resolution clock,
 * which is typically the most accurate clock available on the system.
 *
 * @return std::chrono::high_resolution_clock::time_point The current time point.
 */
inline std::chrono::high_resolution_clock::time_point getCurrentTime() {
    return std::chrono::high_resolution_clock::now();
}

// Function to calculate duration in seconds
inline double calculateDurationInSeconds(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

/**
 * Function to compute and print bandwidth and GFLOPS.
 *
 * @param start The start time point of the operation.
 * @param end The end time point of the operation.
 * @param dataSize The size of the data processed in bytes.
 * @param numOperations The number of floating-point operations performed (in FLOPs).
 */
inline void computeAndPrintMetrics(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end,
    size_t dataSize,
    size_t numOperations) {
    double time = calculateDurationInSeconds(start, end);
    double gflops = static_cast<double>(numOperations) / (time * 1e9);
    double bandwidth = (static_cast<double>(dataSize) / (1024 * 1024 * 1024)) / time;
    std::cout << "Time: " << time << " s\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n\n";
}

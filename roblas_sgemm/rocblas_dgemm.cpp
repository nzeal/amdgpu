#include "kernel0_rocblas.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Helper function to check HIP errors
#define CHECK_HIP_ERROR(err)                                                        \
    if (err != hipSuccess)                                                          \
    {                                                                               \
        std::cerr << "HIP error: " << hipGetErrorString(err) << " in " << __FILE__  \
                  << " line " << __LINE__ << std::endl;                             \
        exit(EXIT_FAILURE);                                                         \
    }

// Function to calculate duration in seconds
double get_duration_seconds(const std::chrono::steady_clock::time_point& start, 
                             const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

// Function to calculate memory throughput (GB/s)
double calculate_memory_throughput(double size_in_bytes, double duration_seconds) {
    return (size_in_bytes / (1024.0 * 1024.0 * 1024.0)) / duration_seconds;
}

// Function to calculate GFLOPS
double calculate_gflops(int n, double duration_seconds) {
    // 2 * N^3 operations for matrix multiplication (SGEMM)
    double num_operations = 2.0 * n * n * n;
    return num_operations / (duration_seconds * 1e9);  // GFLOPS = operations / (seconds * 1e9)
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        return EXIT_FAILURE;
    }
    const int N = atoi(argv[1]);

    // Allocate host memory
    std::vector<float> h_a(N * N, 1.0f); // Initialize with 1s
    std::vector<float> h_b(N * N, 1.0f);
    std::vector<float> h_c(N * N, 0.0f);

    float *d_a, *d_b, *d_c;
    // Allocate device memory
    CHECK_HIP_ERROR(hipMalloc(&d_a, N * N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, N * N * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, N * N * sizeof(float)));

    // Measure time for data transfer from Host to Device
    auto start_time = std::chrono::steady_clock::now();
    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a.data(), N * N * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b.data(), N * N * sizeof(float), hipMemcpyHostToDevice));
    auto end_time = std::chrono::steady_clock::now();

    double memory_transfer_time = get_duration_seconds(start_time, end_time);
    double memory_throughput = calculate_memory_throughput(2 * N * N * sizeof(float), memory_transfer_time);

    // Initialize and run kernel
    Kernel0ROCBLAS kernel;
    kernel.init();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Run SGEMM computation kernel and measure time
    start_time = std::chrono::steady_clock::now();
    kernel.run(d_a, d_b, d_c, alpha, beta, N);
    kernel.finalize();
    end_time = std::chrono::steady_clock::now();

    double computation_time = get_duration_seconds(start_time, end_time);
    double gflops = calculate_gflops(N, computation_time);

    // Measure time for data transfer from Device to Host
    start_time = std::chrono::steady_clock::now();
    CHECK_HIP_ERROR(hipMemcpy(h_c.data(), d_c, N * N * sizeof(float), hipMemcpyDeviceToHost));
    end_time = std::chrono::steady_clock::now();

    double memory_transfer_back_time = get_duration_seconds(start_time, end_time);
    double memory_throughput_back = calculate_memory_throughput(N * N * sizeof(float), memory_transfer_back_time);

    // Free device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));

    // Optionally verify results here

    // Print performance summary
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "Matrix size (N): " << N << "\n";
    std::cout << "Memory Transfer (Host to Device) Time: " << memory_transfer_time << " seconds\n";
    std::cout << "Memory Throughput (Host to Device): " << memory_throughput << " GB/s\n";
    std::cout << "Computation Time: " << computation_time << " seconds\n";
    std::cout << "Performance (GFLOPS): " << gflops << " GFLOPS\n";
    std::cout << "Memory Transfer (Device to Host) Time: " << memory_transfer_back_time << " seconds\n";
    std::cout << "Memory Throughput (Device to Host): " << memory_throughput_back << " GB/s\n";
    std::cout << "--------------------------------\n";

    std::cout << "ROCBLAS SGEMM completed successfully." << std::endl;
    return EXIT_SUCCESS;
}


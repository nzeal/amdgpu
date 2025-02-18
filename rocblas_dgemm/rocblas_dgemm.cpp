#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <hip/hip_runtime.h>
#include "rocblas.h"
#include <sys/time.h>

// Error checking macro for HIP
#define CHECK_HIP(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << " " << __LINE__ << ": " << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Error checking macro for ROCBLAS
#define CHECK_ROCBLAS(call) \
do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error at " << __FILE__ << " " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Struct to store performance results
struct PerformanceResult {
    int size;
    double transfer_to_device_time;
    double computation_time;
    double transfer_from_device_time;
    double gflops;
    double bandwidth_to_device;
    double bandwidth_from_device;
};

// Function to calculate duration in seconds
double calculateDurationInSeconds(double start, double end) {
    return (end - start) / 1000.0;
}

// Function to get current time in milliseconds
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

// Function to initialize matrix with a pattern
void initMatrix(double* matrix, int rows, int cols, double value) {
    for(int i = 0; i < rows * cols; i++) {
        matrix[i] = value;
    }
}

void runDGEMM(rocblas_handle handle, int size, std::vector<PerformanceResult>& results) {
    // Matrix dimensions
    const int m = size;  // rows of A and C
    const int n = size;  // cols of B and C
    const int k = size;  // cols of A and rows of B

    // Calculate matrix sizes in bytes
    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);
    size_t size_C = m * n * sizeof(double);
    size_t total_data_size = size_A + size_B + size_C;
    size_t num_operations = 2ULL * m * n * k;

    std::cout << "\nMatrix size: " << size << "x" << size << "\n";
    std::cout << "Memory required: " << std::fixed << std::setprecision(2)
              << (total_data_size / (1024.0 * 1024.0)) << " MB\n";

    // Allocate host memory
    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed\n";
        exit(EXIT_FAILURE);
    }

    // Initialize matrices
    initMatrix(h_A, m, k, 1.0);
    initMatrix(h_B, k, n, 2.0);
    initMatrix(h_C, m, n, 0.0);

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size_A));
    CHECK_HIP(hipMalloc(&d_B, size_B));
    CHECK_HIP(hipMalloc(&d_C, size_C));

    PerformanceResult result;
    result.size = size;

    // Transfer data to device
    auto transfer_start = getCurrentTime();
    CHECK_HIP(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice));
    auto transfer_end = getCurrentTime();

    result.transfer_to_device_time = calculateDurationInSeconds(transfer_start, transfer_end);
    result.bandwidth_to_device = ((size_A + size_B + size_C) / (1024.0 * 1024.0 * 1024.0)) / result.transfer_to_device_time;

    // DGEMM parameters
    const double alpha = 1.0;
    const double beta = 0.0;

    CHECK_HIP(hipDeviceSynchronize());

    // Perform DGEMM
    auto compute_start = getCurrentTime();
    
    CHECK_ROCBLAS(rocblas_dgemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        m, n, k,
        &alpha,
        d_A, m,
        d_B, k,
        &beta,
        d_C, m
    ));

    CHECK_HIP(hipDeviceSynchronize());
    auto compute_end = getCurrentTime();

    result.computation_time = calculateDurationInSeconds(compute_start, compute_end);
    result.gflops = static_cast<double>(num_operations) / (result.computation_time * 1e9);

    // Transfer result back to host
    auto transfer_back_start = getCurrentTime();
    CHECK_HIP(hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost));
    auto transfer_back_end = getCurrentTime();

    result.transfer_from_device_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
    result.bandwidth_from_device = (size_C / (1024.0 * 1024.0 * 1024.0)) / result.transfer_from_device_time;

    // Verify result
    double expected_value = k * 2.0;
    if (std::abs(h_C[0] - expected_value) > 1e-5 ||
        std::abs(h_C[m*n-1] - expected_value) > 1e-5) {
        std::cerr << "Result verification failed!\n";
        std::cerr << "Expected: " << expected_value << "\n";
        std::cerr << "Got: First=" << h_C[0] << ", Last=" << h_C[m*n-1] << "\n";
    } else {
        std::cout << "Result verification passed!\n";
    }

    results.push_back(result);

    // Cleanup
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

void printSummary(const std::vector<PerformanceResult>& results) {
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << std::setw(8) << "Size"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Compute(ms)"
              << std::setw(15) << "H2D BW(GB/s)"
              << std::setw(15) << "D2H BW(GB/s)"
              << "\n";
    std::cout << std::string(65, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << result.size
                  << std::setw(12) << result.gflops
                  << std::setw(15) << result.computation_time * 1000
                  << std::setw(15) << result.bandwidth_to_device
                  << std::setw(15) << result.bandwidth_from_device
                  << "\n";
    }
}

int main() {
    // Get number of devices
    int deviceCount;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }

    // Set and get device
    CHECK_HIP(hipSetDevice(0));

    std::vector<int> sizes = {1024};
    std::vector<PerformanceResult> results;

    // Create rocBLAS handle
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    // Print device information
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));

    std::cout << "=== HIP Device Information ===\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";

    // Run benchmark
    for (int size : sizes) {
        try {
            runDGEMM(handle, size, results);
        } catch (const std::exception& e) {
            std::cerr << "Error testing size " << size << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Print results
    printSummary(results);
     
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "ROCBLAS DGEMM completed successfully." << std::endl;


    // Cleanup
    CHECK_ROCBLAS(rocblas_destroy_handle(handle));

    return 0;
}

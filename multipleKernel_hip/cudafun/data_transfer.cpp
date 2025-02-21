#include "../includes/matrix_management.h"
#include "../includes/error_checking.h"
#include "../includes/performance_utils.h"

void transferDataToDevice(double *h_A, double *h_B, double *d_A, double *d_B, 
                        int m, int k, int n, PerformanceResult &result) {
    size_t size_A = m * k * sizeof(double);
    size_t size_B = k * n * sizeof(double);

    hipDeviceSynchronize();
    auto transfer_start = getCurrentTime();
    CHECK_HIP(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice));
    hipDeviceSynchronize();
    auto transfer_end = getCurrentTime();

    result.transfer_to_device_time = calculateDurationInSeconds(transfer_start, transfer_end);
    result.bandwidth_to_device = ((size_A + size_B) / (1024.0 * 1024.0 * 1024.0)) / 
                                result.transfer_to_device_time;

    printf("H2D Transfer: %.2f ms (%.2f GB/s)\n",
           result.transfer_to_device_time * 1000,
           result.bandwidth_to_device);
}

void transferDataFromDevice(double *h_C, double *d_C, int m, int n, 
                          const char* kernelName, PerformanceResult &result) {
    size_t size_C = m * n * sizeof(double);

    hipDeviceSynchronize();
    auto transfer_back_start = getCurrentTime();
    CHECK_HIP(hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost));
    hipDeviceSynchronize();
    auto transfer_back_end = getCurrentTime();

    double transfer_time = calculateDurationInSeconds(transfer_back_start, transfer_back_end);
    double bandwidth = (size_C / (1024.0 * 1024.0 * 1024.0)) / transfer_time;

    printf("D2H Transfer %s: %.2f ms (%.2f GB/s)\n", kernelName, transfer_time * 1000, bandwidth);

    // Find the corresponding kernel result and update its transfer metrics
    for (auto& kr : result.kernel_results) {
        if (kr.name == kernelName) {
            kr.transfer_from_device_time = transfer_time;
            kr.bandwidth_from_device = bandwidth;
            break;
        }
    }
}

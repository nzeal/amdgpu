#pragma once

#ifndef PERFORMANCE_RESULT_H
#define PERFORMANCE_RESULT_H

// In performance_result.h
class PerformanceResult {
public:
    int size;
    double computation_time;
    double gflops;
    double transfer_to_device_time;
    double transfer_from_device_time;
    double bandwidth_to_device;
    double bandwidth_from_device;

    // Add members for Kernel 1
    double computation_time_kernel1;
    double gflops_kernel1;
    double transfer_to_device_time_kernel1;
    double transfer_from_device_time_kernel1;
    double bandwidth_to_device_kernel1;
    double bandwidth_from_device_kernel1;

    // Add members for Kernel 2
    double computation_time_kernel2;
    double gflops_kernel2;
    double transfer_to_device_time_kernel2;
    double transfer_from_device_time_kernel2;
    double bandwidth_to_device_kernel2;
    double bandwidth_from_device_kernel2;
};

#endif // PERFORMANCE_RESULT_H


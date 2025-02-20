#pragma once

#ifndef PERFORMANCE_RESULT_H
#define PERFORMANCE_RESULT_H

#include <string>
#include <vector>

struct KernelResult {
    std::string kernel_name;
    double computation_time;
    double gflops;
    double transfer_from_device_time;
    double bandwidth_from_device;
};

struct PerformanceResult {
    int size;
    double transfer_to_device_time;
    double bandwidth_to_device;
    std::vector<KernelResult> kernel_results;
};

#endif // PERFORMANCE_RESULT_H


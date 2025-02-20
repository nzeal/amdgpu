#pragma once

#ifndef PERFORMANCE_RESULT_H
#define PERFORMANCE_RESULT_H

#include <vector>
#include <string>

#pragma once
#include <vector>
#include <string>

struct PerformanceResult {
    int size;
    
    // Host-to-Device transfer metrics
    double transfer_to_device_time;
    double bandwidth_to_device;
    
    // Kernel results structure
    struct KernelResult {
        std::string name;
        double computation_time;
        double gflops;
        double transfer_from_device_time;
        double bandwidth_from_device;
    };
    
    // Vector to store results for multiple kernels
    std::vector<KernelResult> kernel_results;

    // Helper function to add kernel result
    void addKernelResult(const std::string& name, double comp_time, double gflops, 
                        double transfer_time, double bandwidth) {
        KernelResult kr;
        kr.name = name;
        kr.computation_time = comp_time;
        kr.gflops = gflops;
        kr.transfer_from_device_time = transfer_time;
        kr.bandwidth_from_device = bandwidth;
        kernel_results.push_back(kr);
    }
};

#endif // PERFORMANCE_RESULT_H


#pragma once

#ifndef PERFORMANCE_RESULT_H
#define PERFORMANCE_RESULT_H

struct PerformanceResult {
    int size;
    double transfer_to_device_time;
    double computation_time;
    double transfer_from_device_time;
    double gflops;
    double bandwidth_to_device;
    double bandwidth_from_device;
};

#endif // PERFORMANCE_RESULT_H


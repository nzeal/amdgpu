#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "../includes/performance_result.h"

void printDetailedKernelStats(const std::vector<PerformanceResult>& results) {
    for (const auto& res : results) {
        printf("\nDetailed Kernel Performance (Size: %d):\n", res.size);
        printf("%-10s %-12s %-12s %-12s %-12s %-12s %-12s\n", 
               "Kernel", "GFLOPS", "Compute(ms)", "D2H(GB/s)", "H2D(GB/s)", 
               "Total Tput", "Speed");
        printf("---------------------------------------------------------------------------------------------------------\n");
        
        // Find the best GFLOPS for speed comparison
        double best_gflops = 0;
        for (const auto& kr : res.kernel_results) {
            best_gflops = std::max(best_gflops, kr.gflops);
        }
        
        // Print each kernel's results
        for (const auto& kr : res.kernel_results) {
            const double total_throughput = kr.gflops + 
                (res.bandwidth_to_device + kr.bandwidth_from_device) / 2.0;
            const double speed = (best_gflops > 0) ? 
                (kr.gflops / best_gflops) * 100.0 : 0.0;
            
            printf("%-10s %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f %-12.2f%%\n",
                   kr.name.c_str(),
                   kr.gflops,
                   kr.computation_time * 1000,
                   kr.bandwidth_from_device,
                   res.bandwidth_to_device,
                   total_throughput,
                   speed);
        }
        
        // Print total memory throughput
        double total_mem_throughput = res.bandwidth_to_device;
        for (const auto& kr : res.kernel_results) {
            total_mem_throughput += kr.bandwidth_from_device;
        }
        printf("\nTotal Memory Throughput: %.2f GB/s\n", total_mem_throughput);
    }
}


void verifyResults(const double *h_A, const double *h_B, double *h_C, int m, int n, int k, double alpha, double beta) {
    double epsilon = 1e-6;  // Tolerance for floating-point comparison
    bool correct = true;
    int errors = 0;
    const int max_errors_to_print = 10;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double expected = 0.0;
            for (int p = 0; p < k; ++p) {
                expected += h_A[i * k + p] * h_B[p * n + j];
            }
            expected = alpha * expected + beta * h_C[i * n + j];

            double actual = h_C[i * n + j];
            if (std::abs(expected - actual) > epsilon) {
                if (errors < max_errors_to_print) {
                    printf("Error at position (%d, %d): Expected %.8f, Got %.8f\n", i, j, expected, actual);
                }
                errors++;
                correct = false;
            }
        }
    }

    if (correct) {
        printf("Results verified: CORRECT\n");
    } else {
        printf("Results verified: INCORRECT. %d errors found.\n", errors);
    }
}
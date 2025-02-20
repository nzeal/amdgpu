#pragma once
#include <vector>
#include <string>
#include <functional>

typedef void (*KernelFunction)(const double*, const double*, double*, int, int, int, double, double);

struct KernelInfo {
    std::string name;
    KernelFunction function;
};

class KernelRegistry {
public:
    static KernelRegistry& getInstance() {
        static KernelRegistry instance;
        return instance;
    }

    void registerKernel(const std::string& name, KernelFunction func) {
        kernels.push_back({name, func});
    }

    const std::vector<KernelInfo>& getKernels() const { return kernels; }

private:
    KernelRegistry() {} // Private constructor for singleton
    std::vector<KernelInfo> kernels;
};

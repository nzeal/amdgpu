// cuda_version_check.h
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

struct CudaVersionInfo {
    int driver_version;
    int runtime_version;
    bool is_compatible;
    std::string error_message;
};

inline CudaVersionInfo checkCudaVersions() {
    CudaVersionInfo info;
    cudaDriverGetVersion(&info.driver_version);
    cudaRuntimeGetVersion(&info.runtime_version);
    
    info.is_compatible = info.driver_version >= info.runtime_version;
    
    if (!info.is_compatible) {
        char msg[256];
        snprintf(msg, sizeof(msg), 
                "CUDA version mismatch detected!\n"
                "Driver Version: %d\n"
                "Runtime Version: %d\n"
                "The CUDA driver version must be equal to or greater than the runtime version.\n"
                "Please update your CUDA driver to version %d or later.",
                info.driver_version, info.runtime_version, info.runtime_version);
        info.error_message = msg;
    }
    
    return info;
}

inline bool initializeCuda() {
    // Check CUDA versions
    CudaVersionInfo version_info = checkCudaVersions();
    if (!version_info.is_compatible) {
        fprintf(stderr, "CUDA Initialization Error: %s\n", version_info.error_message.c_str());
        return false;
    }

    // Check device availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Unable to get device count: %s\n", 
                cudaGetErrorString(error));
        return false;
    }
    
    if (deviceCount == 0) {
        fprintf(stderr, "CUDA Error: No CUDA-capable devices found\n");
        return false;
    }

    // Set device and check its properties
    cudaError_t setDeviceError = cudaSetDevice(0);
    if (setDeviceError != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to set device: %s\n", 
                cudaGetErrorString(setDeviceError));
        return false;
    }

    // Get and print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Driver Version: %d\n", version_info.driver_version);
    printf("Runtime Version: %d\n", version_info.runtime_version);

    return true;
}

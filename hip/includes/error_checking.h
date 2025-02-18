#pragma once
#ifndef ERROR_CHECKING_H
#define ERROR_CHECKING_H

#include <hip/hip_runtime.h>
//#include <hipblas.h>
#include <cstdio>
#include <cstdlib>

// HIP error checking macro
#define CHECK_HIP(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP error at %s %d: %s\n", __FILE__, __LINE__, \
               hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// hipBLAS error checking macro
#define CHECK_HIPBLAS(call) \
do { \
    hipblasStatus_t status = call; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        printf("hipBLAS error at %s %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#endif // ERROR_CHECKING_H


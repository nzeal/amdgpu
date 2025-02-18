#pragma once
#include <rocblas/rocblas.h>
#include "sgemm.h"

class Kernel0ROCBLAS : public ISgemm
{
public: // Add this line to make the functions public
    virtual std::string name() const override {
        return "Kernel 0 : ROCBLAS";
    }
    virtual void init() override;
    virtual void run(float *d_a, float *d_b, float *d_c, float alpha, float beta, int N) override;
    virtual void finalize() override;

private: // Add this line to make the handle private
    rocblas_handle handle;
};

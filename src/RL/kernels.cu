
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RL/kernels.h"

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

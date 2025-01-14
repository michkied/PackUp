
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RL/kernels.h"

__global__ void rlCompressKernel(unsigned char* input, unsigned char* output)
{
    int i = threadIdx.x;
	output[i] = input[i] + 1;
}

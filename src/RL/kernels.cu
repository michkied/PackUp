
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RL/kernels.h"

__global__ void rlCompressKernel(unsigned char* input, long unsigned int input_size, unsigned char* A, unsigned char* B)
{
    int i = threadIdx.x;
	if (i >= input_size) return;

	// Generate neighbor array
	if (i != 0)
	{
		B[i] = 1 - (char)(input[i] == input[i - 1]);
	}
	else {
		B[i] = 1;
	}
	A[i] = 1;
}

__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned char* A, unsigned char* B, unsigned char* output)
{
	int i = threadIdx.x;
	if (i == input_size - 1 || B[i] != B[i + 1]) {
		output[B[i] - 1] = A[i];
		output[B[i] - 1 + B[input_size-1]] = input[i];
	}
}


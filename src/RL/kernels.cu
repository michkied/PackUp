
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RL/kernels.h"

__global__ void rlCompressKernel(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned char* A, unsigned char* B)
{
    int i = threadIdx.x;
	if (i >= input_size / symbol_size) return;

	A[i] = 1;
	if (i == 0)
	{
		B[i] = 1;
		return;
	}

	char areEqual = 1;
	for (int j = 0; j < symbol_size; j++)
	{
		//if (i * symbol_size + j >= input_size)
		//{
		//	B[i] = 1;
		//	return;
		//}
		areEqual *= (char)(input[i * symbol_size + j] == input[(i - 1) * symbol_size + j]);
	}
	B[i] = 1 - areEqual;

	// Generate neighbor array
	//if (i != 0)
	//{
	//	B[i] = 1 - (char)(input[i] == input[i - 1]);
	//}
	//else {
	//	B[i] = 1;
	//}

}

__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned int symbol_size,  unsigned char* A, unsigned char* B, unsigned char* output)
{
	int i = threadIdx.x;
	if (i != input_size / symbol_size - 1 && B[i] == B[i + 1]) return;

	unsigned char symbol_index = B[i] - 1;
	unsigned char bound = B[input_size / symbol_size - 1];
	output[symbol_index] = A[i];
	for (int j = 0; j < symbol_size; j++) {
		unsigned int byte_index = symbol_index * symbol_size + j;
		output[bound + byte_index] = input[byte_index];
	}

}


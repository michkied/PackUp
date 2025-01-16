
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RL/kernels.h"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void rlCompressKernel(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
		// HANDLE UNEVEN KERNELS
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

__global__ void rlScan(unsigned int* array, long unsigned int array_size)
{
	extern __shared__ int temp[];
	int thid = threadIdx.x;
	array += blockIdx.x * blockDim.x;
	int offset = 1;
	int ai = thid;
	int bi = thid + (array_size / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = array[ai];
	temp[bi + bankOffsetB] = array[bi];

	for (int d = array_size >> 1; d > 0; d >>= 1)
		// build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0)
	{
		temp[array_size - 1 + CONFLICT_FREE_OFFSET(array_size - 1)] = 0;
	}

	for (int d = 1; d < array_size; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{

			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	array[ai] += temp[ai + bankOffsetA];
	array[bi] += temp[bi + bankOffsetB];
}

__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned int symbol_size,  unsigned int* A, unsigned int* B, unsigned int* output_counts, unsigned char* output_symbols)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= input_size / symbol_size) return;
	if (i != input_size / symbol_size - 1 && B[i] == B[i + 1]) return;

	unsigned int symbol_index = B[i] - 1;
	//unsigned char bound = B[input_size / symbol_size - 1];
	output_counts[symbol_index] = A[i];
	for (int j = 0; j < symbol_size; j++) {
		unsigned int byte_index = symbol_index * symbol_size + j;
		output_symbols[byte_index] = input[i * symbol_size + j];
	}

}


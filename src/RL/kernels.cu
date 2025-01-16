﻿
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

//__global__ void rlPrescan(unsigned int* array, unsigned int* out, long unsigned int array_size)
//{
//	//extern __shared__ int temp[]; // allocated on invocation
//	//int thid = threadIdx.x;
//	//int pout = 0, pin = 1;
//	//// Load input into shared memory.
//	//// This is exclusive scan, so shift right by one
//	//// and set first element to 0
//	//temp[pout * array_size + thid] = (thid > 0) ? array[thid - 1] : 0;
//	//__syncthreads();
//	//for (int offset = 1; offset < array_size; offset *= 2)
//	//{
//	//	pout = 1 - pout; // swap double buffer indices
//	//	pin = 1 - pout;
//	//	if (thid >= offset)
//	//		temp[pout * array_size + thid] += temp[pin * array_size + thid - offset];
//	//	else
//	//		temp[pout * array_size + thid] = temp[pin * array_size + thid];
//	//	__syncthreads();
//	//}
//	//out[thid] = temp[pout * array_size + thid]; // write output
//
//
//
//
//	extern __shared__ int temp[];
//	int thid = threadIdx.x;
//	int offset = 1;
//	//if (thid >= array_size) return;
//
//	temp[2 * thid] = array[2 * thid]; // load input into shared memory
//	temp[2*thid+1] = g_idata[2*thid+1];
//
//	for (int d = array_size >> 1; d > 0; d >>= 1)
//		// build sum in place up the tree
//	{
//		__syncthreads();
//		if (thid < d)
//		{
//			int ai = offset * (2 * thid + 1) - 1;
//			int bi = offset * (2 * thid + 2) - 1;
//
//			temp[bi] += temp[ai];
//		}
//		offset *= 2;
//	}
//	if (thid == 0)
//	{
//		temp[array_size - 1] = 0;
//	} // clear the last element
//	for (int d = 1; d < array_size; d *= 2) // traverse down tree & build scan
//	{
//		offset >>= 1;
//		__syncthreads();
//		if (thid < d)
//		{
//			int ai = offset * (2 * thid + 1) - 1;
//			int bi = offset * (2 * thid + 2) - 1;
//			float t = temp[ai];
//			temp[ai] = temp[bi];
//			temp[bi] += t;
//		}
//	}
//	__syncthreads();
//
//	out[2 * thid] = temp[2 * thid]; // write results to device memory
//	out[2 * thid + 1] = temp[2 * thid + 1];
//}

__global__ void rlPrescan(unsigned int* array, unsigned int* out, long unsigned int array_size)
{
	extern __shared__ int temp[];
	int thid = threadIdx.x;
	//if (thid >= array_size) return;
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


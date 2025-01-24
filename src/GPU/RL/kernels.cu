
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPU/RL/kernels.h"

__global__ void rlNeighborArrays(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B)
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
		areEqual *= (char)(input[i * symbol_size + j] == input[(i - 1) * symbol_size + j]);
	}
	B[i] = 1 - areEqual;
}

__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned int symbol_size,  unsigned int* A, unsigned int* B, unsigned int* output_counts, unsigned char* output_symbols, unsigned int partition_size, unsigned int* repetitions)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= input_size / symbol_size) return;
	if (i != input_size / symbol_size - 1 && B[i] == B[i + 1]) return;

	unsigned int symbol_index = B[i] - 1;
	output_counts[symbol_index] = A[i];
	repetitions[symbol_index] = A[i] / partition_size + (A[i] % partition_size != 0);
	for (int j = 0; j < symbol_size; j++) {
		unsigned int byte_index = symbol_index * symbol_size + j;
		output_symbols[byte_index] = input[i * symbol_size + j];
	}
}

__global__ void rlGenerateOutput(unsigned int bound, unsigned int symbol_size, unsigned char* output_symbols, unsigned int* output_counts, unsigned int partition_size, unsigned int* repetitions, unsigned int* repetitions_scan, unsigned char* output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= bound) return;

	unsigned int adj_bound = repetitions_scan[bound - 1] + repetitions[bound - 1];
	unsigned int count = output_counts[i];

	unsigned int remainder = count % partition_size;
	if (remainder != 0)
		output[repetitions_scan[i]] = count % partition_size;
	else
		output[repetitions_scan[i]] = partition_size;
	for (int byte = 0; byte < symbol_size; byte++) {
		output[adj_bound + repetitions_scan[i] * symbol_size + byte] = output_symbols[i * symbol_size + byte];
	}

	for (int rep = 1; rep < repetitions[i]; rep++)
	{
		output[repetitions_scan[i] + rep] = partition_size;
		for (int byte = 0; byte < symbol_size; byte++)
		{
			output[adj_bound + (repetitions_scan[i] + rep) * symbol_size + byte] = output_symbols[i * symbol_size + byte];
		}
	}
}

__global__ void rlPrepareForScan(unsigned int array_size, unsigned char* input, unsigned int* array)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= array_size) return;
	array[i] = input[i];
}

__global__ void rlDecompress(unsigned int* offsets, unsigned int array_size, unsigned char* input, unsigned int symbol_size, unsigned char* output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= array_size) return;

	unsigned char* array = input;
	unsigned char* data = input + array_size;

	unsigned int output_offset = offsets[i];
	unsigned int count = array[i];
	for (int j = 0; j < count; j++)
	{
		for (int byte = 0; byte < symbol_size; byte++)
		{
			output[(output_offset + j) * symbol_size + byte] = data[i * symbol_size + byte];
		}
	}

}


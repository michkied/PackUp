
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPU/RL/RunLengthKernels.h"

// For every symbol in the input, set A to 1 
// and B to 1 if the symbol is different from the previous one, 0 otherwise
__global__ void rlNeighborArrays(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
	if (tID >= input_size / symbol_size) return;

	A[tID] = 1;
	if (tID == 0)
	{
		B[tID] = 1;
		return;
	}

	char areEqual = 1;
	for (int j = 0; j < symbol_size; j++)
	{
		areEqual *= (char)(input[tID * symbol_size + j] == input[(tID - 1) * symbol_size + j]);
	}
	B[tID] = 1 - areEqual;
}

// Create an array of symbols and their counts
// We start with a thread for each symbol.
// If the symbol is on the boundary between two different symbols, we collect relevant data from A and B at the same index.
__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned int symbol_size,  unsigned int* A, unsigned int* B, unsigned int* output_counts, unsigned char* output_symbols, unsigned int partition_size, unsigned int* repetitions)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;
	if (tID >= input_size / symbol_size) return;
	if (tID != input_size / symbol_size - 1 && B[tID] == B[tID + 1]) return;

	unsigned int symbol_index = B[tID] - 1;
	output_counts[symbol_index] = A[tID];
	repetitions[symbol_index] = A[tID] / partition_size + (A[tID] % partition_size != 0);
	for (int j = 0; j < symbol_size; j++) {
		unsigned int byte_index = symbol_index * symbol_size + j;
		output_symbols[byte_index] = input[tID * symbol_size + j];
	}
}

// Generate the output byte array
__global__ void rlGenerateOutput(unsigned int bound, unsigned int symbol_size, unsigned char* output_symbols, unsigned int* output_counts, unsigned int partition_size, unsigned int* repetitions, unsigned int* repetitions_scan, unsigned char* output)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;
	if (tID >= bound) return;

	unsigned int adj_bound = repetitions_scan[bound - 1] + repetitions[bound - 1];
	unsigned int count = output_counts[tID];

	unsigned int remainder = count % partition_size;
	if (remainder != 0)
		output[repetitions_scan[tID]] = count % partition_size;
	else
		output[repetitions_scan[tID]] = partition_size;
	for (int byte = 0; byte < symbol_size; byte++) {
		output[adj_bound + repetitions_scan[tID] * symbol_size + byte] = output_symbols[tID * symbol_size + byte];
	}

	for (int rep = 1; rep < repetitions[tID]; rep++)
	{
		output[repetitions_scan[tID] + rep] = partition_size;
		for (int byte = 0; byte < symbol_size; byte++)
		{
			output[adj_bound + (repetitions_scan[tID] + rep) * symbol_size + byte] = output_symbols[tID * symbol_size + byte];
		}
	}
}

// Convert array of bytes to array of unsigned ints
__global__ void rlPrepareForScan(unsigned int array_size, unsigned char* input, unsigned int* array)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;
	if (tID >= array_size) return;
	array[tID] = input[tID];
}

// Decompress the input array
__global__ void rlDecompress(unsigned int* offsets, unsigned int array_size, unsigned char* input, unsigned int symbol_size, unsigned char* output)
{
	int tID = blockIdx.x * blockDim.x + threadIdx.x;
	if (tID >= array_size) return;

	unsigned char* array = input;
	unsigned char* data = input + array_size;

	unsigned int output_offset = offsets[tID];
	unsigned int count = array[tID];
	for (int j = 0; j < count; j++)
	{
		for (int byte = 0; byte < symbol_size; byte++)
		{
			output[(output_offset + j) * symbol_size + byte] = data[tID * symbol_size + byte];
		}
	}

}


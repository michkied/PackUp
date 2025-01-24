#pragma once

#include "cuda_runtime.h"

__global__ void rlNeighborArrays(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B);

__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B, unsigned int* output_counts, unsigned char* output_symbols, unsigned int partition_size, unsigned int* repetitions);

__global__ void rlGenerateOutput(unsigned int bound, unsigned int symbol_size, unsigned char* output_symbols, unsigned int* output_counts, unsigned int partition_size, unsigned int* repetitions, unsigned int* repetitions_scan, unsigned char* output);

__global__ void rlPrepareForScan(unsigned int array_size, unsigned char* input, unsigned int* array);

__global__ void rlDecompress(unsigned int* offsets, unsigned int array_size, unsigned char* input, unsigned int symbol_size, unsigned char* output);
#pragma once

#include "cuda_runtime.h"

__global__ void rlCompressKernel(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B);

__global__ void rlScan(unsigned int* array, long unsigned int array_size);

__global__ void rlCollectResults(unsigned char* input, long unsigned int input_size, unsigned int symbol_size, unsigned int* A, unsigned int* B, unsigned int* output_counts, unsigned char* output_symbols);

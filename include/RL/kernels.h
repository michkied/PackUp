#pragma once

#include "cuda_runtime.h"

__global__ void rlCompressKernel(unsigned char* input, long int input_size, unsigned char* A, unsigned char* B);

__global__ void rlCollectResults(unsigned char* input, long int input_size, unsigned char* A, unsigned char* B, unsigned char* output);

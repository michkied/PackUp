#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

cudaError_t run_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);

cudaError_t run_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);

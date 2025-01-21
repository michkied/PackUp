#pragma once

#include "cuda_runtime.h"

cudaError_t fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter);

cudaError_t fixed_length_compress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter);

cudaError_t fixed_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);

cudaError_t fixed_length_decompress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);

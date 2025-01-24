#pragma once

#include "cuda_runtime.h"

class RunLength {
public:
	cudaError_t compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter);

	cudaError_t decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);

	~RunLength();

private:
	struct CompressDevPointers {
		unsigned char* input = nullptr;
		unsigned int* A = nullptr;
		unsigned int* B = nullptr;
		unsigned int* output_counts = nullptr;
		unsigned char* output_symbols = nullptr;
		unsigned int* output_repetitions = nullptr;
		unsigned int* output_repetitions_scan = nullptr;
		unsigned char* output = nullptr;
	} c_dev;

	struct DecompressDevPointers {
		unsigned int* offsets = nullptr;
		unsigned char* input = nullptr;
		unsigned char* output = nullptr;
	} d_dev;
};

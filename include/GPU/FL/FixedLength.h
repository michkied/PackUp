#pragma once

#include "cuda_runtime.h"

#include "GPU/FL/types.hpp"

class FixedLength {
public:
	cudaError_t compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter);

	cudaError_t decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);

	~FixedLength();

private:
	struct CompressDevPointers {
		unsigned char* input = nullptr;
		unsigned int* seg_sizes = nullptr;
		unsigned int* seg_offsets = nullptr;
		unsigned int* insig_bits_count = nullptr;
		unsigned int* division_seg_sizes = nullptr;
		unsigned int* division_zeros = nullptr;
		DivisionWrapper* divisions = nullptr;
		DivisionWrapper* division_scan = nullptr;
		unsigned char* output = nullptr;
	} c_dev;

	struct DecompressDevPointers {
		unsigned char* input = nullptr;
		unsigned int* frame_lengths = nullptr;
		unsigned int* frame_lengths_scan = nullptr;
		unsigned char* output = nullptr;
	} d_dev;

	void free_compression_memory();

	void free_decompression_memory();

	cudaError_t compress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter);

	cudaError_t decompress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);
};

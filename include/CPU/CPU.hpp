#pragma once

namespace CPU
{
	void run_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter);

	void run_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size);
}


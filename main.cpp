#include "RL/RL.h"
#include "FL/FL.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

//#define PRINT_USAGE std::cout << "Usage: " << argv[0] << " <c/d> <fl/rl> <input file> <output file>" << std::endl; return 1

int main(int argc, char* argv[])
{
    //   if (argc != 5) PRINT_USAGE;
       //if (argv[1][0] != 'c' && argv[1][0] != 'd') PRINT_USAGE;
       //if (argv[2][0] != 'f' && argv[2][0] != 'r') PRINT_USAGE;

    std::ifstream input_file("test_files/fl_output.txt", std::ios::in | std::ios::binary);
    std::ofstream output_file("test_files/fl_output2.txt", std::ios::out | std::ios::binary);
    if (!input_file || !output_file) {
        return 1;
    }
    long unsigned int input_size;
	input_file.seekg(0, std::ios::end);
	input_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
	unsigned char* input = new unsigned char[input_size];
	input_file.read(reinterpret_cast<char*>(input), input_size);
    
    unsigned char* output;
    long unsigned int output_size;

    input_file.close();

	auto compressed = fixed_length_decompress(input, input_size, output, output_size);
    output_file.write(reinterpret_cast<char*>(output), output_size);
	output_file.close();

    delete[] output;

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
#include "RL/kernels.h"
#include "RL/RL.h"

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

    std::ifstream input_file("test_files/input2.txt", std::ios::in | std::ios::binary);
    std::ofstream output_file("test_files/output.txt", std::ios::out | std::ios::binary);
    if (!input_file || !output_file) {
        return 1;
    }
    long unsigned int input_size;
	input_file.seekg(0, std::ios::end);
	input_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
	unsigned char* input = new unsigned char[input_size];
	input_file.read(reinterpret_cast<char*>(input), input_size);
    
    std::vector<unsigned char> output;

    input_file.close();

	auto compressed = run_length_compress(input, input_size, output);
    output_file.write(reinterpret_cast<const char*>(output.data()), output.size());
	output_file.close();

    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
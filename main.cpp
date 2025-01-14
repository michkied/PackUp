#include "RL/kernels.h"
#include "RL/RL.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstddef>

#define __cpp_lib_byte

//#define PRINT_USAGE std::cout << "Usage: " << argv[0] << " <c/d> <fl/rl> <input file> <output file>" << std::endl; return 1

int main(int argc, char* argv[])
{
    //   if (argc != 5) PRINT_USAGE;
       //if (argv[1][0] != 'c' && argv[1][0] != 'd') PRINT_USAGE;
       //if (argv[2][0] != 'f' && argv[2][0] != 'r') PRINT_USAGE;

    std::ifstream input("input.txt", std::ios::in | std::ios::binary);
    std::ofstream output("output.txt", std::ios::out | std::ios::binary);
    if (!input || !output) {
        return 1;
    }
    std::vector<unsigned char> bytes(std::istreambuf_iterator<char>{input}, {});

    input.close();

	auto compressed = run_length_compress(bytes);
	output.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());

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

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}
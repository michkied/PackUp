#include "RL/RL.h"
#include "FL/FL.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

void print_usage(const char* path)
{
    std::cout << "Usage: " << path << " <c/d> <fl/rl> <input file> <output file> [frame size (fl) / symbol size (rl)]" << std::endl;
}

//int main(int argc, char* argv[])
//{
//    //   if (argc != 5) PRINT_USAGE;
//       //if (argv[1][0] != 'c' && argv[1][0] != 'd') PRINT_USAGE;
//       //if (argv[2][0] != 'f' && argv[2][0] != 'r') PRINT_USAGE;
//
//    std::ifstream input_file("test_files/fl_output.txt", std::ios::in | std::ios::binary);
//    std::ofstream output_file("test_files/fl_output2.txt", std::ios::out | std::ios::binary);
//    if (!input_file || !output_file) {
//        return 1;
//    }
//    long unsigned int input_size;
//	input_file.seekg(0, std::ios::end);
//	input_size = input_file.tellg();
//    input_file.seekg(0, std::ios::beg);
//	unsigned char* input = new unsigned char[input_size];
//	input_file.read(reinterpret_cast<char*>(input), input_size);
//    
//    unsigned char* output;
//    long unsigned int output_size;
//
//    input_file.close();
//
//	auto compressed = fixed_length_decompress(input, input_size, output, output_size);
//    output_file.write(reinterpret_cast<char*>(output), output_size);
//	output_file.close();
//
//    delete[] output;
//
//    cudaError_t cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}


int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 6)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    std::string algorithm = argv[2];
    std::string input_path = argv[3];
    std::string output_path = argv[4];

    std::ifstream input_file(input_path, std::ios::in | std::ios::binary);
    std::ofstream output_file(output_path, std::ios::out | std::ios::binary);

    unsigned int parameter = 0;
    if (argc == 6)
    {
        try
        {
            parameter = std::stoi(argv[5]);
        }
        catch (const std::invalid_argument& e)
        {
            std::cerr << "Invalid frame size or symbol size provided." << std::endl;
            return 1;
        }

        if (algorithm == "rl" && (parameter > 255 || parameter <= 0))
        {
			std::cerr << "Symbol size must be between 1 and 255 bytes." << std::endl;
			return 1;
        }
        if (algorithm == "fl" && (parameter > 4098 || parameter <= 0))
        {
            std::cerr << "Frame size must be between 1 and 4098 bytes." << std::endl;
            return 1;
        }
    }
    else
    {
        if (algorithm == "rl")
        {
            parameter = 4;
        }
        else if (algorithm == "fl")
        {
            parameter = 512;
        }
    }

    if (!input_file || !output_file)
    {
        std::cerr << "Error opening input or output file." << std::endl;
        return 1;
    }

    input_file.seekg(0, std::ios::end);
    long unsigned int input_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    unsigned char* input = new unsigned char[input_size];
    input_file.read(reinterpret_cast<char*>(input), input_size);
    input_file.close();

    unsigned char* output = nullptr;
    long unsigned int output_size = 0;

    if (mode == "c")
    {
        if (algorithm == "fl")
        {
            if (fixed_length_compress(input, input_size, output, output_size, parameter) != cudaSuccess)
            {
                std::cerr << "Fixed-length compression failed." << std::endl;
                delete[] input;
                return 1;
            }
        }
        else if (algorithm == "rl")
        {
            if (run_length_compress(input, input_size, output, output_size, parameter) != cudaSuccess)
            {
                std::cerr << "Run-length compression failed." << std::endl;
                delete[] input;
                return 1;
            }
        }
        else
        {
            print_usage(argv[0]);
            delete[] input;
            return 1;
        }
    }
    else if (mode == "d")
    {
        if (algorithm == "fl")
        {
            if (fixed_length_decompress(input, input_size, output, output_size) != cudaSuccess)
            {
                std::cerr << "Fixed-length decompression failed." << std::endl;
                delete[] input;
                return 1;
            }
        }
        else if (algorithm == "rl")
        {
            if (run_length_decompress(input, input_size, output, output_size) != cudaSuccess)
            {
                std::cerr << "Run-length decompression failed." << std::endl;
                delete[] input;
                return 1;
            }
        }
        else
        {
            print_usage(argv[0]);
            delete[] input;
            return 1;
        }
    }
    else
    {
        print_usage(argv[0]);
        delete[] input;
        return 1;
    }

    output_file.write(reinterpret_cast<char*>(output), output_size);
    output_file.close();

    delete[] input;
    delete[] output;

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    return 0;
}

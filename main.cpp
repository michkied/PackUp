#include "GPU/RL/RL.h"
#include "GPU/FL/FL.h"
#include "CPU/CPU.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

void print_usage(const char* path)
{
    std::cout << "Usage: " << path << " <c/d> <fl/rl> <input file> <output file> [no_cmp/cmp] [frame size (fl) / symbol size (rl)]" << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc < 5 || argc > 7)
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
    bool compare = false;
    if (argc > 5)
    {
        std::string cmp = argv[5];
        compare = (cmp == "cmp");
    }

    if (argc == 7)
    {
        try
        {
            parameter = std::stoi(argv[6]);
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

    unsigned char* _ = nullptr;
    long unsigned int __ = 0;

    if (mode == "c")
    {
        if (algorithm == "fl")
        {
            if (compare) std::cout << "CPU comparison available only for RL algorithm" << std::endl;
            if (fixed_length_compress(input, input_size, output, output_size, parameter) != cudaSuccess)
            {
                std::cerr << "Fixed-length compression failed." << std::endl;
                delete[] input;
                return 1;
            }
        }
        else if (algorithm == "rl")
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            std::cout << "Running on GPU..." << std::endl;
            if (run_length_compress(input, input_size, output, output_size, parameter) != cudaSuccess)
            {
                std::cerr << "Run-length compression failed." << std::endl;
                delete[] input;
                return 1;
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "GPU finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";

            if (compare)
            {
                std::cout << "Running on CPU..." << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                CPU::run_length_compress(input, input_size, _, __, parameter);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::cout << "CPU finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
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
            if (compare) std::cout << "CPU comparison available only for RL algorithm" << std::endl;
            if (fixed_length_decompress(input, input_size, output, output_size) != cudaSuccess)
            {
                std::cerr << "Fixed-length decompression failed." << std::endl;
                delete[] input;
                return 1;
            }
        }
        else if (algorithm == "rl")
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            std::cout << "Running on GPU..." << std::endl;
            if (run_length_decompress(input, input_size, output, output_size) != cudaSuccess)
            {
                std::cerr << "Run-length decompression failed." << std::endl;
                delete[] input;
                return 1;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "GPU finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";

            if (compare)
            {
                std::cout << "Running on CPU..." << std::endl;
                auto start_time = std::chrono::high_resolution_clock::now();
                CPU::run_length_decompress(input, input_size, _, __);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::cout << "CPU finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
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
    delete[] _;

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    return 0;
}

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

cudaError_t run_length_compress(unsigned char* input, long unsigned int input_size, std::vector<unsigned char>& output);

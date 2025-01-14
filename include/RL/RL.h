#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

cudaError_t run_length_compress(std::vector<unsigned char>& input, std::vector<unsigned char>& output);

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

std::vector<unsigned char> run_length_compress(std::vector<unsigned char>& data);

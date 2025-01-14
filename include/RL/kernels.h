#pragma once

#include "cuda_runtime.h"

__global__ void rlCompressKernel(unsigned char* input, unsigned char* output);
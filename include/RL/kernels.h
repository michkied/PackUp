#pragma once

#include "cuda_runtime.h"

__global__ void addKernel(int* c, const int* a, const int* b);
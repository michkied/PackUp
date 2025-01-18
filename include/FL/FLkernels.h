#pragma once

#include "cuda_runtime.h"

__global__ void flFindInsigBits(unsigned char* frame, unsigned int frame_size_B, unsigned int* seg_sizes, unsigned int* seg_offsets, unsigned int* insig_bits);

__global__ void flComputeNumOfZeros(unsigned int* insig_bits, unsigned int* division_ends, unsigned int* division_zeros, unsigned int* seg_sizes, unsigned int frame_size_b);

__global__ void flProduceOutput(unsigned char* frame, unsigned int seg_size, unsigned int insig_zeros, unsigned char* output);
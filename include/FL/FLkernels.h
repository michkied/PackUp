#pragma once

#include "cuda_runtime.h"

#include <FL/types.hpp>

__global__ void flFindInsigBits(unsigned int seg_count, unsigned char* input, unsigned int frame_size_B, unsigned int* seg_sizes, unsigned int* seg_offsets, unsigned int* insig_bits);

__global__ void flComputeNumOfZeros(unsigned int divisions_count, unsigned int* division_zeros, unsigned int* division_seg_sizes, unsigned int frame_size_b, DivisionWrapper* output);

__global__ void flProduceOutput(unsigned char* input, DivisionWrapper* divisions, DivisionWrapper* totals, unsigned int frame_size_b, unsigned char* output, unsigned int header_size);

__global__ void flHandleRemainders(unsigned int frame_count, unsigned char* input, DivisionWrapper* divisions, DivisionWrapper* totals, unsigned int frame_size_b, unsigned char* output, unsigned int header_size);

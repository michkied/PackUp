#pragma once

#include "cuda_runtime.h"

#include <GPU/FL/types.hpp>

// Compression

__global__ void flFindInsigBits(unsigned int seg_count, unsigned char* input, unsigned int frame_size_B, unsigned int* seg_sizes, unsigned int* seg_offsets, unsigned int* insig_bits);

__global__ void flComputeNumOfZeros(unsigned int divisions_count, unsigned int* division_zeros, unsigned int* division_seg_sizes, unsigned int frame_size_b, DivisionWrapper* output);

__global__ void flProduceOutput(unsigned char* input, DivisionWrapper* divisions, DivisionWrapper* totals, unsigned int frame_size_b, unsigned char* output, unsigned int header_array_size);

__global__ void flAddHeadersAndRemainders(unsigned int frame_count, unsigned char* input, DivisionWrapper* divisions, DivisionWrapper* totals, unsigned int frame_size_b, unsigned char* output, unsigned int header_array_size);


// Decompression

__global__ void flComputeFrameLengths(unsigned int frame_count, unsigned int frame_size_B, unsigned char* header_array, unsigned int* frame_lengths);

__global__ void flDecompressFrames(unsigned int frame_count, unsigned char* input, unsigned int* frame_lengths, unsigned int* comp_frame_offsets, unsigned int frame_size_B, unsigned char* output);

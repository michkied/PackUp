#include "device_launch_parameters.h"

#include "FL/FLkernels.h"

__global__ void flFindInsigBits(unsigned char* frame, unsigned int frame_size_B, unsigned int* seg_sizes, unsigned int* seg_offsets, unsigned int* insig_bits)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int seg_size = seg_sizes[i];
	unsigned int seg_offset = seg_offsets[i];
	
	unsigned int insig_bits_count = 0;
	unsigned int bit_offset = seg_offset - 1;
	for (unsigned int bit_num = 0; bit_num < seg_size - 1; ++bit_num)
	{
		++bit_offset;
		if (bit_offset >= frame_size_B * 8)
		{
			++insig_bits_count;
			continue;
		}

		unsigned char bit = frame[bit_offset / 8] & (1 << (7 - (bit_offset % 8)));
		if (bit != 0) break;
		++insig_bits_count;
	}

	insig_bits[i] = insig_bits_count;
}

__global__ void flComputeNumOfZeros(unsigned int* insig_bits, unsigned int* division_ends, unsigned int* division_zeros, unsigned int* seg_sizes, unsigned int frame_size_b) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int division_end = division_ends[i];
	unsigned int minimum = insig_bits[division_end - 1];
	unsigned int seg_size = seg_sizes[division_end - 1];
	unsigned int regular_zeros = minimum * (frame_size_b / seg_size);

	unsigned int remainder_zeros = 0;
	unsigned int remainder_size = frame_size_b % seg_size;
	if (remainder_size != 0)
	{
		if (minimum >= remainder_size)
		{
			remainder_zeros = remainder_size - 1;
		}
		else
		{
			remainder_zeros = minimum;
		}
	}

	division_zeros[i] = regular_zeros + remainder_zeros;
}
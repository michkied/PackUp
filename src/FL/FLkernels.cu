#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include "device_launch_parameters.h"

#include "FL/FLkernels.h"
#include "FL/types.hpp"

__device__ char atomicOrChar(unsigned char* address, char val) {
	// Calculate the location of the `char` within its containing `int`
	unsigned int* baseAddress = (unsigned int*)((uintptr_t)address & ~3); // Align to 4 bytes
	unsigned int shift = ((uintptr_t)address & 3) * 8;                   // Offset in bits
	unsigned int mask = 0xFF << shift;                                  // Mask for the `char`

	unsigned int old, assumed, newVal;
	do {
		old = *baseAddress; // Load the full 4-byte word
		char currentVal = (old & mask) >> shift; // Extract the current `char`
		currentVal |= val; // Apply the OR operation
		newVal = (old & ~mask) | ((currentVal & 0xFF) << shift); // Construct the new word
		assumed = atomicCAS(baseAddress, old, newVal); // Atomic compare-and-swap
	} while (assumed != old);

	return (old & mask) >> shift; // Return the previous value of the `char`
}

__global__ void flFindInsigBits(unsigned int seg_count, unsigned char* input, unsigned int frame_size_B, unsigned int* seg_sizes, unsigned int* seg_offsets, unsigned int* insig_bits)
{
	//extern __shared__ int temp[];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= seg_count) return;

	input += blockIdx.x * frame_size_B; // move pointer to the beginning of the frame

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

		unsigned char bit = input[bit_offset / 8] & (1 << (7 - (bit_offset % 8)));
		if (bit != 0) break;
		++insig_bits_count;
	}

	insig_bits[blockIdx.x * seg_count + i] = insig_bits_count;
}

__global__ void flComputeNumOfZeros(unsigned int divisions_count, unsigned int* division_zeros, unsigned int* division_seg_sizes, unsigned int frame_size_b, DivisionWrapper* output)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= divisions_count) return;

	unsigned int frame_offset = blockIdx.x * divisions_count;

	unsigned int seg_size = division_seg_sizes[i];
	unsigned int minimum = division_zeros[frame_offset + i];
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

	output[frame_offset + i] = DivisionWrapper(regular_zeros + remainder_zeros, seg_size, minimum);
}

__global__ void flProduceOutput(unsigned char* input, DivisionWrapper* divisions, DivisionWrapper* totals, unsigned int frame_size_b, unsigned char* output, unsigned int header_array_size)
{
	int frame_num = blockIdx.x;
	DivisionWrapper division = divisions[frame_num];
	unsigned int seg_size = division.seg_size;

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= frame_size_b / seg_size) return;

	unsigned int insig_zeros = division.insig_zeros;
	unsigned int frame_offset_b = frame_size_b * frame_num;
	unsigned int output_frame_offset_b = header_array_size * 8 + frame_offset_b - (totals[frame_num].removed_zeros - division.removed_zeros);  // since totals is an inclusive scan, we need to subtract current removed zeros

	unsigned int seg_offset = frame_offset_b + i * seg_size;
	unsigned int output_offset = output_frame_offset_b + i * (seg_size - insig_zeros);
	for (int bit_num = 0; bit_num < seg_size - insig_zeros; ++bit_num)
	{
		unsigned int bit_offset = seg_offset + insig_zeros + bit_num;
		unsigned char bit = input[bit_offset / 8] & (1 << (7 - (bit_offset % 8)));
		
		unsigned int output_bit_offset = output_offset + bit_num;
		atomicOrChar(output + output_bit_offset / 8, (bit != 0) << (7 - (output_bit_offset % 8)));
	}
}

__global__ void flAddHeadersAndRemainders(unsigned int frame_count, unsigned char* input, DivisionWrapper* divisions, DivisionWrapper* totals, unsigned int frame_size_b, unsigned char* output, unsigned int header_array_size) 
{
	int frame_num = blockIdx.x * blockDim.x + threadIdx.x;
	if (frame_num >= frame_count) return;

	DivisionWrapper division = divisions[frame_num];
	unsigned int frame_offset_b = frame_size_b * frame_num;
	unsigned int output_frame_offset_b = header_array_size * 8 + frame_offset_b - (totals[frame_num].removed_zeros - division.removed_zeros);

	// Add header
	unsigned int out_seg_size = division.seg_size - division.insig_zeros;
	output[frame_num * 2] = out_seg_size & 0xFF;
	output[frame_num * 2 + 1] = (out_seg_size >> 8) & 0xFF;
	output[(frame_count + frame_num) * 2] = division.removed_zeros & 0xFF;
	output[(frame_count + frame_num) * 2 + 1] = (division.removed_zeros >> 8) & 0xFF;

	// Add remainder
	unsigned int remainder_size = frame_size_b % division.seg_size;
	if (remainder_size == 0) return;

	unsigned int remainder_zeros = division.insig_zeros;
	if (division.insig_zeros >= remainder_size)
	{
		remainder_zeros = remainder_size - 1;
	}
	unsigned int remainder_offset = frame_offset_b + frame_size_b - remainder_size + remainder_zeros;
	unsigned int output_end_offset = output_frame_offset_b + frame_size_b - division.removed_zeros;
	unsigned int output_offset = output_end_offset - remainder_size + remainder_zeros;
	for (unsigned int bit_num = 0; bit_num < remainder_size - remainder_zeros; ++bit_num)
	{
		unsigned int bit_offset = remainder_offset + bit_num;
		unsigned char bit = input[bit_offset / 8] & (1 << (7 - (bit_offset % 8)));
		if (bit != 0)
		{
			atomicOrChar(output + output_offset / 8, (bit != 0) << (7 - (output_offset % 8)));
		}
		++output_offset;
	}
}

// "frame size" is the size of the frame before compression
// "frame length" is the size of the frame after compression
__global__ void flComputeFrameLengths(unsigned int frame_count, unsigned int frame_size_B, unsigned char* header_array, unsigned int* frame_lengths)
{
	int frame_num = blockIdx.x * blockDim.x + threadIdx.x;
	if (frame_num >= frame_count) return;

	//unsigned int out_seg_size = header_array[frame_num * 2] + (header_array[frame_num * 2 + 1] << 8);
	unsigned int removed_zeros = header_array[(frame_count + frame_num) * 2] + (header_array[(frame_count + frame_num) * 2 + 1] << 8);

	frame_lengths[frame_num] = frame_size_B * 8 - removed_zeros;
}

__global__ void flDecompressFrames(unsigned int frame_count, unsigned char* input, unsigned int* frame_lengths, unsigned int* comp_frame_offsets, unsigned int frame_size_B, unsigned char* output)
{
	int frame_num = blockIdx.x * blockDim.x + threadIdx.x;
	if (frame_num >= frame_count) return;

	unsigned int frame_offset = comp_frame_offsets[frame_num];
	unsigned int output_frame_offset = frame_num * frame_size_B;
	unsigned int comp_frame_length = frame_lengths[frame_num];

	unsigned int out_seg_size = input[frame_num * 2] + (input[frame_num * 2 + 1] << 8);
	unsigned int removed_zeros = input[(frame_count + frame_num) * 2] + (input[(frame_count + frame_num) * 2 + 1] << 8);

	unsigned int zeros_per_segment = removed_zeros / out_seg_size;
	unsigned int seg_size = out_seg_size + zeros_per_segment;

	//for (unsigned int byte_num = 0; byte_num < frame_size_B; ++byte_num)
	//{
	//	output[output_frame_offset + byte_num] = input[comp_frame_offset + byte_num];
	//}
}
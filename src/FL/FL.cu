#include <vector>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <FL/FL.h>
#include <FL/FLkernels.h>

cudaError_t fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	unsigned int frame_size_B = 8;
	unsigned int frame_size_b = frame_size_B * 8;
	unsigned int frame_count = input_size / frame_size_B;

	// TODO: Perform compression only if first bit of the frame is 0

	// Precompute helper arrays
	unsigned int thread_count = 0;
	unsigned int divisions_count = 0;
	std::vector<unsigned int> seg_sizes;
	std::vector<unsigned int> seg_offsets;
	std::vector<unsigned int> division_end_offsets;
	for (unsigned int seg_size = 2; seg_size <= frame_size_b; ++seg_size)
	{
		unsigned int threads = frame_size_b / seg_size + (unsigned int)(frame_size_b % seg_size != 0);
		for (unsigned int i = 0; i < threads; ++i)
		{
			seg_sizes.push_back(seg_size);
			seg_offsets.push_back(i * seg_size);
		}
		thread_count += threads;
		++divisions_count;
		division_end_offsets.push_back(thread_count);
	}

	unsigned char* dev_input;
	unsigned int* dev_seg_sizes;
	unsigned int* dev_seg_offsets;
	unsigned int* dev_insig_bits_count;
	unsigned int* dev_division_ends;
	unsigned int* dev_division_zeros;
	unsigned char* dev_output;
	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Cleanup;
	}

	// Allocate memory
	cudaStatus = cudaMalloc((void**)&dev_input, input_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_seg_sizes, thread_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_seg_offsets, thread_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_insig_bits_count, thread_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_division_ends, divisions_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_division_zeros, divisions_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}

	// Copy data to device
	cudaStatus = cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(dev_seg_sizes, seg_sizes.data(), thread_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(dev_seg_offsets, seg_offsets.data(), thread_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(dev_division_ends, division_end_offsets.data(), divisions_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	// Find the number of insignificant bits in each segment
	flFindInsigBits << <1, thread_count >> > (dev_input, frame_size_B, dev_seg_sizes, dev_seg_offsets, dev_insig_bits_count);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flFindInsigBits launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flFindInsigBits!\n", cudaStatus);
		goto Cleanup;
	}
	//unsigned int* insig_bits_count = new unsigned int[thread_count];
	//cudaStatus = cudaMemcpy(insig_bits_count, dev_insig_bits_count, thread_count * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Cleanup;
	//}

	// Find minimums within each division
	thrust::inclusive_scan_by_key(thrust::device, dev_seg_sizes, dev_seg_sizes + thread_count, dev_insig_bits_count, dev_insig_bits_count, thrust::equal_to<unsigned int>{}, thrust::minimum<unsigned int>{});
	//// Copy output data to host
	//unsigned int* insig_bits_count2 = new unsigned int[thread_count];
	//cudaStatus = cudaMemcpy(insig_bits_count2, dev_insig_bits_count, thread_count * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Cleanup;
	//}

	// Compute the number of insignificant zeros removed by each division
	flComputeNumOfZeros << <1, divisions_count >> > (dev_insig_bits_count, dev_division_ends, dev_division_zeros, dev_seg_sizes, frame_size_b);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flFindInsigBits launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flFindInsigBits!\n", cudaStatus);
		goto Cleanup;
	}
	//unsigned int* debug = new unsigned int[divisions_count];
	//cudaStatus = cudaMemcpy(debug, dev_division_zeros, divisions_count * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Cleanup;
	//}

	unsigned int* dev_max_elem = thrust::max_element(thrust::device, dev_division_zeros, dev_division_zeros + divisions_count);
	unsigned int max_elem_index = dev_max_elem - dev_division_zeros;

	unsigned int best_seg_index, total_zeros_removed, best_seg_size, insig_zeros;
	cudaStatus = cudaMemcpy(&best_seg_index, dev_division_ends + max_elem_index, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(&total_zeros_removed, dev_max_elem, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(&best_seg_size, dev_seg_sizes + best_seg_index - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(&insig_zeros, dev_insig_bits_count + best_seg_index - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	// Produce output
	unsigned int output_size_b = frame_size_b - total_zeros_removed;
	output_size = output_size_b / 8 + (output_size_b % 8 != 0);
	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&dev_output, output_size + output_size % 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemset(dev_output, 0, output_size + output_size % 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Cleanup;
	}

	flProduceOutput << <1, frame_size_b / best_seg_size >> > (dev_input, best_seg_size, insig_zeros, dev_output);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flProduceOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flProduceOutput!\n", cudaStatus);
		goto Cleanup;
	}
	
	cudaStatus = cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	// Factor in the remainder
	unsigned int remainder_size = frame_size_b % best_seg_size;
	if (remainder_size != 0)
	{
		unsigned int remainder_zeros = insig_zeros;
		if (insig_zeros >= remainder_size)
		{
			remainder_zeros = remainder_size - 1;
		}
		unsigned int remainder_offset = frame_size_b - remainder_size;
		unsigned int output_offset = output_size_b - remainder_size + remainder_zeros;
		for (unsigned int bit_num = 0; bit_num < remainder_size - remainder_zeros; ++bit_num)
		{
			unsigned int bit_offset = remainder_offset + remainder_zeros + bit_num;
			unsigned char bit = input[bit_offset / 8] & (1 << (7 - (bit_offset % 8)));
			if (bit != 0)
			{
				output[output_offset / 8] |= ((bit != 0) << (7 - (output_offset % 8)));
			}
			++output_offset;
		}
	}

Cleanup:
	cudaFree(dev_input);
	cudaFree(dev_seg_sizes);
	cudaFree(dev_seg_offsets);
	cudaFree(dev_insig_bits_count);
	cudaFree(dev_division_ends);
	cudaFree(dev_division_zeros);
	cudaFree(dev_output);

	return cudaStatus;
}

cudaError_t fixed_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	return cudaSuccess;
}

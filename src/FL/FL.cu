#include <vector>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <FL/FL.h>
#include <FL/FLkernels.h>

cudaError_t fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	unsigned int frame_size_B = 2;
	unsigned int frame_size_b = frame_size_B * 8;
	unsigned int frame_count = input_size / frame_size_B;

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
	flFindInsigBits << <1, thread_count >> > (dev_input + 4, frame_size_B, dev_seg_sizes, dev_seg_offsets, dev_insig_bits_count);
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
	unsigned int* debug = new unsigned int[divisions_count];
	cudaStatus = cudaMemcpy(debug, dev_division_zeros, divisions_count * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	unsigned int* dev_max_elem = thrust::max_element(thrust::device, dev_division_zeros, dev_division_zeros + divisions_count);
	unsigned int max_elem_index = dev_max_elem - dev_division_zeros;
	unsigned int best_seg_index;
	cudaStatus = cudaMemcpy(&best_seg_index, dev_division_ends + max_elem_index, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	unsigned int best_seg_size;
	cudaStatus = cudaMemcpy(&best_seg_size, dev_seg_sizes + best_seg_index - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	





Cleanup:
	cudaFree(dev_input);
	cudaFree(dev_seg_sizes);
	cudaFree(dev_seg_offsets);
	cudaFree(dev_insig_bits_count);
	cudaFree(dev_division_ends);
	cudaFree(dev_division_zeros);

	return cudaStatus;
}

cudaError_t fixed_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	return cudaSuccess;
}

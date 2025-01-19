#include <vector>
#include <thrust/scan.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <FL/FL.h>
#include <FL/FLkernels.h>
#include <FL/types.hpp>

cudaError_t fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	output_size = 0;
	output = nullptr;

	unsigned int frame_size_B = 2;
	unsigned int frame_size_b = frame_size_B * 8;
	unsigned int frame_count = input_size / frame_size_B;
	unsigned int threads_per_block = 256;

	// Precompute helper arrays
	unsigned int seg_count = 0;
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
		seg_count += threads;
		++divisions_count;
		division_end_offsets.push_back(seg_count);
	}

	unsigned char* dev_input;
	unsigned int* dev_seg_sizes;
	unsigned int* dev_seg_offsets;
	unsigned int* dev_insig_bits_count;
	unsigned int* dev_division_seg_sizes;
	unsigned int* dev_division_zeros;
	DivisionWrapper* dev_divisions;
	DivisionWrapper* dev_division_scan;
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
	cudaStatus = cudaMalloc((void**)&dev_seg_sizes, seg_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_seg_offsets, seg_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_division_seg_sizes, divisions_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_insig_bits_count, seg_count * frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_division_zeros, divisions_count * frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_divisions, divisions_count * frame_count * sizeof(DivisionWrapper));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_division_scan, frame_count * sizeof(DivisionWrapper));
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
	cudaStatus = cudaMemcpy(dev_seg_sizes, seg_sizes.data(), seg_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemcpy(dev_seg_offsets, seg_offsets.data(), seg_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	// Find the number of insignificant bits in each segment
	// Each block is one-dimensional and each row of blocks processes one frame. The y coordinate of the block is the frame number.
	flFindInsigBits << < dim3{ seg_count / threads_per_block + 1, frame_count }, threads_per_block >> > (seg_count, dev_input, frame_size_B, dev_seg_sizes, dev_seg_offsets, dev_insig_bits_count);
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

	// Find minimums within each division for each frame
	thrust::reduce_by_key(
		thrust::device,
		CyclicIterator(dev_seg_sizes, seg_count),
		CyclicIterator(dev_seg_sizes, seg_count, frame_count* seg_count),
		dev_insig_bits_count,
		dev_division_seg_sizes,
		dev_division_zeros,
		thrust::equal_to<unsigned int>{},
		thrust::minimum<unsigned int>{}
	);

	// Compute the number of insignificant zeros removed by each division
	flComputeNumOfZeros << < dim3{ divisions_count / threads_per_block + 1, frame_count }, threads_per_block >> > (divisions_count, dev_division_zeros, dev_division_seg_sizes, frame_size_b, dev_divisions);
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

	// Find the division with the most zeros removed for each frame
	thrust::reduce_by_key(
		thrust::device,
		CyclicIterator(divisions_count),
		CyclicIterator(divisions_count, divisions_count * frame_count),
		dev_divisions,
		thrust::make_discard_iterator(),
		dev_divisions,
		thrust::less_equal<unsigned int>{},
		thrust::maximum<DivisionWrapper>{}
	);

	// Compute the prefix sum of removed zeros
	thrust::inclusive_scan(
		thrust::device,
		dev_divisions,
		dev_divisions + frame_count,
		dev_division_scan,
		thrust::plus<DivisionWrapper>{}
	);

	// Produce output
	DivisionWrapper totals;
	cudaStatus = cudaMemcpy(&totals, dev_division_scan + frame_count - 1, sizeof(DivisionWrapper), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
	unsigned int compressed_size_b = input_size * 8 - totals.removed_zeros;
	output_size = compressed_size_b / 8 + (compressed_size_b % 8 != 0) + input_size % frame_size_B;
	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&dev_output, output_size + output_size % 4);  // Add padding to ensure that the output size is a multiple of 4 (necessary for atomicCAS)
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemset(dev_output, 0, output_size + output_size % 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Cleanup;
	}

	flProduceOutput << < dim3{ (frame_size_b / 2) / threads_per_block + 1, frame_count }, threads_per_block >> > (dev_input, dev_divisions, dev_division_scan, frame_size_b, dev_output);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flProduceOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	flHandleRemainders << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, dev_input, dev_divisions, dev_division_scan, frame_size_b, dev_output);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flIncludeRemainders launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}

	// Add bytes to the end that were not processed (input size didn't divide evenly by frame size)
	for (unsigned int i = 1; i <= input_size % frame_size_B; ++i)
	{
		output[output_size - i] = input[input_size - i];
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flProduceOutput and flIncludeRemainders!\n", cudaStatus);
		goto Cleanup;
	}
	
	// Copy output to host
	cudaStatus = cudaMemcpy(output, dev_output, output_size - input_size % frame_size_B, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}


Cleanup:
	cudaFree(dev_input);
	cudaFree(dev_seg_sizes);
	cudaFree(dev_seg_offsets);
	cudaFree(dev_insig_bits_count);
	cudaFree(dev_division_seg_sizes);
	cudaFree(dev_division_zeros);
	cudaFree(dev_output);
	cudaFree(dev_divisions);
	cudaFree(dev_division_scan);

	return cudaStatus;
}

cudaError_t fixed_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	return cudaSuccess;
}

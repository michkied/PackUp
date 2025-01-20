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
	unsigned int threads_per_block = 1024;

	// Precompute helper arrays
	fprintf(stderr, "Precomputing helper arrays\n");
	unsigned int seg_count = 0;
	unsigned int divisions_count = 0;
	std::vector<unsigned int> seg_sizes;
	std::vector<unsigned int> seg_offsets;
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
	}
	fprintf(stderr, "    Done\n");

	unsigned char* dev_input = nullptr;
	unsigned int* dev_seg_sizes = nullptr;
	unsigned int* dev_seg_offsets = nullptr;
	unsigned int* dev_insig_bits_count = nullptr;
	unsigned int* dev_division_seg_sizes = nullptr;
	unsigned int* dev_division_zeros = nullptr;
	DivisionWrapper* dev_divisions = nullptr;
	DivisionWrapper* dev_division_scan = nullptr;
	unsigned char* dev_output = nullptr;
	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Cleanup;
	}

	// Allocate memory
	fprintf(stderr, "Allocating memory and copying data\n");
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
	fprintf(stderr, "    Done\n");

	// Find the number of insignificant bits in each segment
	// Each block is one-dimensional and each row of blocks processes one frame. The x coordinate of the block is the frame number.
	fprintf(stderr, "Finding insignificant bits for every division\n");
	flFindInsigBits << < dim3{ frame_count, seg_count / threads_per_block + 1 }, dim3{ 1, threads_per_block } >> > (seg_count, dev_input, frame_size_B, dev_seg_sizes, dev_seg_offsets, dev_insig_bits_count);
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
	fprintf(stderr, "    Done\n");

	// Find minimums within each division for each frame
	fprintf(stderr, "Finding minimums within divisions\n");
	thrust::reduce_by_key(
		thrust::device,
		CyclicIterator(dev_seg_sizes, seg_count),
		CyclicIterator(dev_seg_sizes, seg_count, frame_count * seg_count),
		dev_insig_bits_count,
		dev_division_seg_sizes,
		dev_division_zeros,
		thrust::equal_to<unsigned int>{},
		thrust::minimum<unsigned int>{}
	);
	fprintf(stderr, "    Done\n");

	// Free memory that is no longer needed
	cudaFree(dev_seg_offsets);
	cudaFree(dev_seg_sizes);
	cudaFree(dev_insig_bits_count);

	// Allocate memory for division wrappers
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

	// Compute the number of insignificant zeros removed by each division
	fprintf(stderr, "Computing the number of insignificant zeros removed by each division\n");
	flComputeNumOfZeros << < dim3{ frame_count, divisions_count / threads_per_block + 1 }, dim3{ 1, threads_per_block } >> > (divisions_count, dev_division_zeros, dev_division_seg_sizes, frame_size_b, dev_divisions);
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
	fprintf(stderr, "    Done\n");

	// Find the division with the most zeros removed for each frame
	fprintf(stderr, "Fiding best division\n");
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
	fprintf(stderr, "    Done\n");

	// Compute the prefix sum of best divisions
	fprintf(stderr, "Computing the prefix sum of best divisions\n");
	thrust::inclusive_scan(
		thrust::device,
		dev_divisions,
		dev_divisions + frame_count,
		dev_division_scan,
		thrust::plus<DivisionWrapper>{}
	);
	fprintf(stderr, "    Done\n");

	// Produce output
	fprintf(stderr, "Producing output\n");
	DivisionWrapper totals;
	cudaStatus = cudaMemcpy(&totals, dev_division_scan + frame_count - 1, sizeof(DivisionWrapper), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	unsigned int header_info_size = 4 + 2; // 4 bytes for frame count, 2 bytes for frame size
	unsigned int header_array_size = frame_count * (2 + 2); // 2 bytes per frame for out segment size, 2 bytes per frame for total removed zeros in frame
	unsigned int header_size = header_info_size + header_array_size; 

	unsigned int compressed_size_b = frame_size_b * frame_count - totals.removed_zeros;
	unsigned int gpu_output_size = header_array_size + compressed_size_b / 8 + (compressed_size_b % 8 != 0);

	output_size = header_info_size + gpu_output_size + input_size % frame_size_B;
	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&dev_output, gpu_output_size + gpu_output_size % 4);  // Add padding to ensure that the output size is a multiple of 4 (necessary for atomicCAS)
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemset(dev_output, 0, gpu_output_size + gpu_output_size % 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Cleanup;
	}

	flProduceOutput << < dim3{ frame_count, (frame_size_b / 2) / threads_per_block + 1 }, dim3{ 1, threads_per_block } >> > (dev_input, dev_divisions, dev_division_scan, frame_size_b, dev_output, header_array_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flProduceOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	flAddHeadersAndRemainders << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, dev_input, dev_divisions, dev_division_scan, frame_size_b, dev_output, header_array_size);
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

	// Add header info
	std::memcpy(output, &frame_count, 4);
	std::memcpy(output + 4, &frame_size_B, 2);

	// Synchronize
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flProduceOutput and flIncludeRemainders!\n", cudaStatus);
		goto Cleanup;
	}
	fprintf(stderr, "    Done\n");
	
	// Copy output to host
	cudaStatus = cudaMemcpy(output + header_info_size, dev_output, gpu_output_size, cudaMemcpyDeviceToHost);
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
	unsigned int threads_per_block = 1024;

	unsigned int frame_count = 0;
	unsigned int frame_size_B = 0;
	std::memcpy(&frame_count, input, 4);
	std::memcpy(&frame_size_B, input + 4, 2);

	unsigned char* dev_input = nullptr;
	unsigned int* dev_frame_lengths = nullptr;
	unsigned int* dev_frame_lengths_scan = nullptr;
	unsigned char* dev_output = nullptr;
	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Cleanup;
	}

	// Allocate memory
	fprintf(stderr, "Allocating memory and copying data\n");
	cudaStatus = cudaMalloc((void**)&dev_input, input_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_frame_lengths, frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMalloc((void**)&dev_frame_lengths_scan, frame_count * sizeof(int));
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

	// Compute the compressed frame lengths
	flComputeFrameLengths << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, frame_size_B, dev_input + 6, dev_frame_lengths);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flComputeFrameLengths launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flComputeFrameLengths!\n", cudaStatus);
		goto Cleanup;
	}

	// Perform a scan to find total length and calculate frame offsets
	thrust::inclusive_scan(
		thrust::device,
		dev_frame_lengths,
		dev_frame_lengths + frame_count,
		dev_frame_lengths_scan
	);

	unsigned int compressed_length_b = 0;
	cudaStatus = cudaMemcpy(&compressed_length_b, dev_frame_lengths_scan + frame_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	unsigned int compressed_length_B = compressed_length_b / 8 + (compressed_length_b % 8 != 0);
	unsigned int non_processed_size = input_size - 4 - 2 - frame_count * (2 + 2) - compressed_length_B;
	output_size = frame_count * frame_size_B + non_processed_size;

	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&dev_output, output_size - compressed_length_B);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemset(dev_output, 0, output_size - compressed_length_B);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Cleanup;
	}

	// Decompress frames
	// __global__ void flDecompressFrames(unsigned int frame_count, unsigned char* input, unsigned int* frame_lengths, unsigned int* comp_frame_offsets, unsigned int frame_size_B, unsigned char* output);
	flDecompressFrames << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, dev_input + 6, dev_frame_lengths, dev_frame_lengths_scan, frame_size_B, dev_output);
	


	unsigned int* debug2 = new unsigned int[frame_count];
	cudaStatus = cudaMemcpy(debug2, dev_frame_lengths, frame_count * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	// Allocate memory for the output


Cleanup:
	return cudaStatus;
}

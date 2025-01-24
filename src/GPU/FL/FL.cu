#include <vector>
#include <thrust/scan.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <chrono>

#include <GPU/FL/FL.h>
#include <GPU/FL/FLkernels.h>
#include <GPU/FL/types.hpp>


cudaError_t fixed_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter) 
{
	output_size = 0;
	output = nullptr;
	long unsigned int max_portion_size = 1 << 25;

	cudaError_t cudaStatus = cudaSuccess;

	for (long unsigned int i = 0; i < input_size; i += max_portion_size)
	{
		printf("\Processing portion #%d\n", i+1);
		unsigned int portion_size = std::min(max_portion_size, input_size - i);

		unsigned char* portion_output = nullptr;
		long unsigned int portion_output_size = 0;

		cudaStatus = fixed_length_compress_portion(input + i, portion_size, portion_output, portion_output_size, parameter);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fixed_length_compress_portion failed!");
			return cudaError_t::cudaErrorUnknown;
		}
		unsigned char* new_output = new unsigned char[output_size + 4 + portion_output_size];
		std::memcpy(new_output, output, output_size);
		std::memcpy(new_output + output_size, &portion_output_size, 4);
		std::memcpy(new_output + output_size + 4, portion_output, portion_output_size);

		delete[] output;
		delete[] portion_output;

		output = new_output;
		output_size += portion_output_size + 4;
	}

	return cudaSuccess;
}

cudaError_t fixed_length_compress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter) 
{

	unsigned int frame_size_B = parameter;
	unsigned int frame_size_b = frame_size_B * 8;
	unsigned int frame_count = input_size / frame_size_B;
	unsigned int threads_per_block = 256;

	auto start_time = std::chrono::high_resolution_clock::now();

	// Precompute helper arrays
	printf("Precomputing helper arrays\n");
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
	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

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
		return cudaError_t::cudaErrorUnknown;
	}

	// Allocate memory
	printf("Allocating memory and copying data\n");
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
	cudaStatus = cudaMalloc((void**)&dev_division_seg_sizes, divisions_count * frame_count * sizeof(int));
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
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Find the number of insignificant bits in each segment
	// Each block is one-dimensional and each row of blocks processes one frame. The x coordinate of the block is the frame number.
	printf("Finding insignificant bits for every division\n");
	flFindInsigBits << < dim3{ frame_count, seg_count / threads_per_block + 1 }, dim3{ 1, threads_per_block }, frame_size_B >> > (seg_count, dev_input, frame_size_B, dev_seg_sizes, dev_seg_offsets, dev_insig_bits_count);
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
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Find minimums within each division for each frame
	printf("Finding minimums within divisions\n");
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
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

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
	printf("Computing the number of insignificant zeros removed by each division\n");
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
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Find the division with the most zeros removed for each frame
	printf("Fiding best division\n");
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
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Compute the prefix sum of best divisions
	printf("Computing the prefix sum of best divisions\n");
	thrust::inclusive_scan(
		thrust::device,
		dev_divisions,
		dev_divisions + frame_count,
		dev_division_scan,
		thrust::plus<DivisionWrapper>{}
	);
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Produce output
	printf("Producing output\n");
	DivisionWrapper totals;
	cudaStatus = cudaMemcpy(&totals, dev_division_scan + frame_count - 1, sizeof(DivisionWrapper), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	unsigned int header_info_size = 4 + 2; // 4 bytes for frame count, 2 bytes for frame size
	unsigned int header_array_size = frame_count * (2 + 2); // 2 bytes per frame for segment size, 2 bytes per frame for total removed zeros in frame
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

	flProduceOutput << < dim3{ frame_count, (frame_size_b / 2) / threads_per_block + 1 }, dim3{ 1, threads_per_block }, frame_size_B >> > (dev_input, dev_divisions, dev_division_scan, frame_size_b, dev_output, header_array_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flProduceOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}
	flAddHeadersAndRemainders << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, dev_input, dev_divisions, dev_division_scan, frame_size_b, dev_output, header_array_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flAddHeadersAndRemainders launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flProduceOutput and flAddHeadersAndRemainders!\n", cudaStatus);
		goto Cleanup;
	}
	
	// Copy output to host
	cudaStatus = cudaMemcpy(output + header_info_size, dev_output, gpu_output_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;


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
	output_size = 0;
	output = nullptr;

	long unsigned int processed_size = 0;
	unsigned int portion_size = 0;
	std::memcpy(&portion_size, input, 4);
	input += 4;

	int i = 1;
	while (processed_size < input_size)
	{
		printf("\Processing portion #%d\n", i);
		unsigned char* portion_output = nullptr;
		long unsigned int portion_output_size = 0;

		cudaError_t cudaStatus = fixed_length_decompress_portion(input, portion_size, portion_output, portion_output_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fixed_length_decompress_portion failed!");
			return cudaError_t::cudaErrorUnknown;
		}
		unsigned char* new_output = new unsigned char[output_size + portion_output_size];

		std::memcpy(new_output, output, output_size);
		std::memcpy(new_output + output_size, portion_output, portion_output_size);

		delete[] output;
		delete[] portion_output;

		output = new_output;
		output_size += portion_output_size;
		processed_size += 4 + portion_size;
		input += portion_size;
		std::memcpy(&portion_size, input, 4);
		input += 4;
		i++;
	}

	return cudaSuccess;
}

cudaError_t fixed_length_decompress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) 
{
	auto start_time = std::chrono::high_resolution_clock::now();

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
	printf("Allocating memory and copying data\n");
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
	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Compute the compressed frame lengths
	printf("Computing compressed frame lengths\n");
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
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Perform a scan to find total length and calculate frame offsets
	printf("Finding frame offsets\n");
	thrust::inclusive_scan(
		thrust::device,
		dev_frame_lengths,
		dev_frame_lengths + frame_count,
		dev_frame_lengths_scan
	);
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	unsigned int compressed_length_b = 0;
	cudaStatus = cudaMemcpy(&compressed_length_b, dev_frame_lengths_scan + frame_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	printf("Preparing output\n");
	unsigned int compressed_length_B = compressed_length_b / 8 + (compressed_length_b % 8 != 0);
	unsigned int non_processed_size = input_size - 4 - 2 - frame_count * (2 + 2) - compressed_length_B;
	output_size = frame_count * frame_size_B + non_processed_size;

	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&dev_output, frame_count * frame_size_B);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Cleanup;
	}
	cudaStatus = cudaMemset(dev_output, 0, frame_count * frame_size_B);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Cleanup;
	}

	// Decompress frames
	flDecompressFrames << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, dev_input + 6, dev_frame_lengths, dev_frame_lengths_scan, frame_size_B, dev_output);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flDecompressFrames launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Cleanup;
	}

	// Add bytes to the end that were not processed
	for (unsigned int byte = 1; byte <= non_processed_size; ++byte)
	{
		output[output_size - byte] = input[input_size - byte];
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flDecompressFrames!\n", cudaStatus);
		goto Cleanup;
	}

	cudaStatus = cudaMemcpy(output, dev_output, frame_count * frame_size_B, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;


Cleanup:
	cudaFree(dev_input);
	cudaFree(dev_frame_lengths);
	cudaFree(dev_frame_lengths_scan);
	cudaFree(dev_output);

	return cudaStatus;
}

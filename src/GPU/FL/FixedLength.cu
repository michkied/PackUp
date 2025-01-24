#include <vector>
#include <thrust/scan.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <chrono>

#include <GPU/FL/FixedLength.h>
#include <GPU/FL/FixedLengthKernels.h>
#include <GPU/FL/types.hpp>

FixedLength::~FixedLength()
{
	cudaFree(c_dev.input);
	cudaFree(c_dev.seg_sizes);
	cudaFree(c_dev.seg_offsets);
	cudaFree(c_dev.insig_bits_count);
	cudaFree(c_dev.division_seg_sizes);
	cudaFree(c_dev.division_zeros);
	cudaFree(c_dev.output);
	cudaFree(c_dev.divisions);
	cudaFree(c_dev.division_scan);

	cudaFree(d_dev.input);
	cudaFree(d_dev.frame_lengths);
	cudaFree(d_dev.frame_lengths_scan);
	cudaFree(d_dev.output);
}

cudaError_t FixedLength::compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter)
{
	output_size = 0;
	output = nullptr;
	long unsigned int max_portion_size = 1 << 25;

	cudaError_t cudaStatus = cudaSuccess;

	// Split input into portions and compress each portion separately
	unsigned int index = 1;
	for (long unsigned int i = 0; i < input_size; i += max_portion_size)
	{
		printf("\nProcessing portion #%d\n", index);
		unsigned int portion_size = std::min(max_portion_size, input_size - i);

		unsigned char* portion_output = nullptr;
		long unsigned int portion_output_size = 0;

		cudaStatus = compress_portion(input + i, portion_size, portion_output, portion_output_size, parameter);
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

		index++;
	}

	return cudaSuccess;
}

cudaError_t FixedLength::decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size)
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
		printf("\nProcessing portion #%d\n", i);
		unsigned char* portion_output = nullptr;
		long unsigned int portion_output_size = 0;

		cudaError_t cudaStatus = decompress_portion(input, portion_size, portion_output, portion_output_size);
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

cudaError_t FixedLength::compress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter)
{
	auto start_time = std::chrono::high_resolution_clock::now();

	unsigned int frame_size_B = parameter;
	unsigned int frame_size_b = frame_size_B * 8;
	unsigned int frame_count = input_size / frame_size_B;
	unsigned int threads_per_block = 256;

	// Precompute helper arrays that contain the sizes and offsets of all segments for all divisions of the frame
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

	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaError_t::cudaErrorUnknown;
	}

	// Allocate memory
	printf("Allocating memory and copying data\n");
	cudaStatus = cudaMalloc((void**)&c_dev.input, input_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&c_dev.seg_sizes, seg_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&c_dev.seg_offsets, seg_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&c_dev.division_seg_sizes, divisions_count * frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&c_dev.insig_bits_count, seg_count * frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&c_dev.division_zeros, divisions_count * frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Copy data to device
	cudaStatus = cudaMemcpy(c_dev.input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(c_dev.seg_sizes, seg_sizes.data(), seg_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(c_dev.seg_offsets, seg_offsets.data(), seg_count * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Find the number of insignificant bits in each segment
	// Each block is one-dimensional and each row of blocks processes one frame. The x coordinate of the block is the frame number.
	printf("Finding insignificant bits for every division\n");
	flFindInsigBits << < dim3{ frame_count, seg_count / threads_per_block + 1 }, dim3{ 1, threads_per_block }, frame_size_B >> > (seg_count, c_dev.input, frame_size_B, c_dev.seg_sizes, c_dev.seg_offsets, c_dev.insig_bits_count);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flFindInsigBits launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flFindInsigBits!\n", cudaStatus);
		return cudaStatus;
	}
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Find minimums within each division for each frame
	// The output is the maximum number of insignificant bits that can be removed by each division
	printf("Finding minimums within divisions\n");
	thrust::reduce_by_key(
		thrust::device,
		CyclicIterator(c_dev.seg_sizes, seg_count),  // CyclicIterator is a custom iterator that repeats the given sequence. This avoids the need to copy dev_seg_sizes multiple times to match the size of dev_insig_bits_count
		CyclicIterator(c_dev.seg_sizes, seg_count, frame_count * seg_count),
		c_dev.insig_bits_count,
		c_dev.division_seg_sizes,
		c_dev.division_zeros,
		thrust::equal_to<unsigned int>{},
		thrust::minimum<unsigned int>{}
	);
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Allocate memory for division wrappers
	cudaStatus = cudaMalloc((void**)&c_dev.divisions, divisions_count * frame_count * sizeof(DivisionWrapper));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&c_dev.division_scan, frame_count * sizeof(DivisionWrapper));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Compute the number of insignificant zeros removed by each division
	printf("Computing the number of insignificant zeros removed by each division\n");
	flComputeNumOfZeros << < dim3{ frame_count, divisions_count / threads_per_block + 1 }, dim3{ 1, threads_per_block } >> > (divisions_count, c_dev.division_zeros, c_dev.division_seg_sizes, frame_size_b, c_dev.divisions);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flFindInsigBits launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flFindInsigBits!\n", cudaStatus);
		return cudaStatus;
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
		c_dev.divisions,
		thrust::make_discard_iterator(),
		c_dev.divisions,
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
		c_dev.divisions,
		c_dev.divisions + frame_count,
		c_dev.division_scan,
		thrust::plus<DivisionWrapper>{}
	);
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Produce output
	printf("Producing output\n");
	DivisionWrapper totals;
	cudaStatus = cudaMemcpy(&totals, c_dev.division_scan + frame_count - 1, sizeof(DivisionWrapper), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	unsigned int header_info_size = 4 + 2; // 4 bytes for frame count, 2 bytes for frame size
	unsigned int header_array_size = frame_count * (2 + 2); // 2 bytes per frame for segment size, 2 bytes per frame for total removed zeros in frame

	unsigned int compressed_size_b = frame_size_b * frame_count - totals.removed_zeros;
	unsigned int gpu_output_size = header_array_size + compressed_size_b / 8 + (compressed_size_b % 8 != 0);

	output_size = header_info_size + gpu_output_size + input_size % frame_size_B;
	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&c_dev.output, gpu_output_size + gpu_output_size % 4);  // Add padding to ensure that the output size is a multiple of 4 (necessary for atomicCAS)
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemset(c_dev.output, 0, gpu_output_size + gpu_output_size % 4);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		return cudaStatus;
	}

	flProduceOutput << < dim3{ frame_count, (frame_size_b / 2) / threads_per_block + 1 }, dim3{ 1, threads_per_block }, frame_size_B >> > (c_dev.input, c_dev.divisions, c_dev.division_scan, frame_size_b, c_dev.output, header_array_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flProduceOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	flAddHeadersAndRemainders << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, c_dev.input, c_dev.divisions, c_dev.division_scan, frame_size_b, c_dev.output, header_array_size);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flAddHeadersAndRemainders launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
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
		return cudaStatus;
	}

	// Copy output to host
	cudaStatus = cudaMemcpy(output + header_info_size, c_dev.output, gpu_output_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	return cudaStatus;
}

cudaError_t FixedLength::decompress_portion(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size)
{
	auto start_time = std::chrono::high_resolution_clock::now();

	unsigned int threads_per_block = 1024;

	unsigned int frame_count = 0;
	unsigned int frame_size_B = 0;
	std::memcpy(&frame_count, input, 4);
	std::memcpy(&frame_size_B, input + 4, 2);

	cudaError_t cudaStatus = cudaSuccess;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate memory
	printf("Allocating memory and copying data\n");
	cudaStatus = cudaMalloc((void**)&d_dev.input, input_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&d_dev.frame_lengths, frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&d_dev.frame_lengths_scan, frame_count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Copy data to device
	cudaStatus = cudaMemcpy(d_dev.input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Compute the compressed frame lengths
	printf("Computing compressed frame lengths\n");
	flComputeFrameLengths << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, frame_size_B, d_dev.input + 6, d_dev.frame_lengths);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flComputeFrameLengths launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flComputeFrameLengths!\n", cudaStatus);
		return cudaStatus;
	}
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	// Perform a scan to find total length and calculate frame offsets
	printf("Finding frame offsets\n");
	thrust::inclusive_scan(
		thrust::device,
		d_dev.frame_lengths,
		d_dev.frame_lengths + frame_count,
		d_dev.frame_lengths_scan
	);
	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	unsigned int compressed_length_b = 0;
	cudaStatus = cudaMemcpy(&compressed_length_b, d_dev.frame_lengths_scan + frame_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	printf("Preparing output\n");
	unsigned int compressed_length_B = compressed_length_b / 8 + (compressed_length_b % 8 != 0);
	unsigned int non_processed_size = input_size - 4 - 2 - frame_count * (2 + 2) - compressed_length_B;
	output_size = frame_count * frame_size_B + non_processed_size;

	output = new unsigned char[output_size];
	cudaStatus = cudaMalloc((void**)&d_dev.output, frame_count * frame_size_B);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	cudaStatus = cudaMemset(d_dev.output, 0, frame_count * frame_size_B);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		return cudaStatus;
	}

	// Decompress frames
	flDecompressFrames << < frame_count / threads_per_block + 1, threads_per_block >> > (frame_count, d_dev.input + 6, d_dev.frame_lengths, d_dev.frame_lengths_scan, frame_size_B, d_dev.output);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "flDecompressFrames launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// Add bytes to the end that were not processed
	for (unsigned int byte = 1; byte <= non_processed_size; ++byte)
	{
		output[output_size - byte] = input[input_size - byte];
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching flDecompressFrames!\n", cudaStatus);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(output, d_dev.output, frame_count * frame_size_B, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	end_time = std::chrono::high_resolution_clock::now();
	std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
	start_time = end_time;

	return cudaStatus;
}

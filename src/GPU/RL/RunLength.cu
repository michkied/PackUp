#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <chrono>

#include "GPU/RL/RunLengthkernels.h"
#include "GPU/RL/RunLength.h"
#include "thrust/functional.h"
#include <thrust/execution_policy.h>
#include "thrust/scan.h"

RunLength::~RunLength()
{
	cudaFree(c_dev.input);
	cudaFree(c_dev.A);
	cudaFree(c_dev.B);
	cudaFree(c_dev.output_counts);
	cudaFree(c_dev.output_symbols);
	cudaFree(c_dev.output_repetitions);
	cudaFree(c_dev.output_repetitions_scan);
	cudaFree(c_dev.output);

	cudaFree(d_dev.input);
	cudaFree(d_dev.offsets);
	cudaFree(d_dev.output);
}

cudaError_t RunLength::compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int partition_size = 255;
    unsigned int symbol_size = parameter; // max 255
	unsigned int threads_per_block = 256;
	unsigned int symbol_count = input_size / symbol_size;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate memory
    printf("Allocating memory and copying data\n");
    cudaStatus = cudaMalloc((void**)&c_dev.input, input_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&c_dev.A, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&c_dev.B, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    // Copy input
	cudaStatus = cudaMemcpy(c_dev.input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
	}
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

    // Prepare neighbor arrays
	printf("Preparing neighbor arrays\n");
	rlNeighborArrays << <symbol_count / threads_per_block + 1, threads_per_block >> > (c_dev.input, input_size, symbol_size, c_dev.A, c_dev.B);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlNeighborArrays launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlNeighborArrays!\n", cudaStatus);
        return cudaStatus;
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

	// Perform a scan on the B array
    printf("Performing a scan\n");
	thrust::inclusive_scan(thrust::device, c_dev.B, c_dev.B + symbol_count, c_dev.B);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

	// Perform a segmented scan on the A array with B as the key
    printf("Performing a segmented scan\n");
    thrust::inclusive_scan_by_key(thrust::device, c_dev.B, c_dev.B + symbol_count, c_dev.A, c_dev.A, thrust::equal_to<unsigned int>{}, thrust::plus<unsigned int>{});
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

	// Collect results
    printf("Collecting results\n");
    unsigned int bound;
    cudaStatus = cudaMemcpy(&bound, c_dev.B + symbol_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&c_dev.output_counts, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&c_dev.output_symbols, bound * symbol_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&c_dev.output_repetitions, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&c_dev.output_repetitions_scan, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

	rlCollectResults << <symbol_count / threads_per_block + 1, threads_per_block >> > (c_dev.input, input_size, symbol_size, c_dev.A, c_dev.B, c_dev.output_counts, c_dev.output_symbols, partition_size, c_dev.output_repetitions);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlCollectResults launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlCollectResults!\n", cudaStatus);
        return cudaStatus;
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

    // Generate output
    printf("Generating output\n");

	// Perform a scan on the repetitions array to get offsets
    thrust::exclusive_scan(thrust::device, c_dev.output_repetitions, c_dev.output_repetitions + bound, c_dev.output_repetitions_scan);

    unsigned int adjusted_bound, temp;
    cudaMemcpy(&adjusted_bound, c_dev.output_repetitions_scan + bound - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp, c_dev.output_repetitions + bound - 1, sizeof(int), cudaMemcpyDeviceToHost);
    adjusted_bound += temp;

    cudaStatus = cudaMalloc((void**)&c_dev.output, adjusted_bound + adjusted_bound * symbol_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    rlGenerateOutput << <bound / threads_per_block + 1, threads_per_block >> > (bound, symbol_size, c_dev.output_symbols, c_dev.output_counts, partition_size, c_dev.output_repetitions, c_dev.output_repetitions_scan, c_dev.output);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlGenerateOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlGenerateOutput!\n", cudaStatus);
        return cudaStatus;
    }

    // Copy result
    unsigned int header_size = 5;
    unsigned int remaining_symbols_size = input_size % symbol_size;
    unsigned int gpu_output_size = adjusted_bound + adjusted_bound * symbol_size;
    output_size = gpu_output_size + header_size + remaining_symbols_size;

    output = new unsigned char[output_size];
    cudaStatus = cudaMemcpy(output + header_size, c_dev.output, gpu_output_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    output[0] = symbol_size;
    std::memcpy(output + 1, &adjusted_bound, 4);
    for (int i = 0; i < remaining_symbols_size; i++)
    {
        output[output_size - remaining_symbols_size + i] = input[input_size - remaining_symbols_size + i];
    }

    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

    return cudaStatus;
}

cudaError_t RunLength::decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size)
{
    auto start_time = std::chrono::high_resolution_clock::now();

	unsigned int threads_per_block = 256;
    unsigned int header_size = 5;
    unsigned int symbol_size = input[0];
    unsigned int array_size;
    std::memcpy(&array_size, input + 1, 4);
    unsigned int remaining_bytes = input_size - header_size - array_size - array_size * symbol_size;

	cudaError_t cudaStatus;
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
	cudaStatus = cudaMalloc((void**)&d_dev.offsets, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

	// Copy input
	cudaStatus = cudaMemcpy(d_dev.input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

	// Since the input is in the form of a byte array, we need to convert it to an array of integers to avoid overflows
    printf("Performing array type change to allow for a scan\n");
	rlPrepareForScan << <array_size / threads_per_block + 1, threads_per_block >> > (array_size, d_dev.input + header_size, d_dev.offsets);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlPrepareForScan launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlPrepareForScan!\n", cudaStatus);
        return cudaStatus;
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

	// Compute offsets
    printf("Performing a prescan\n");
	thrust::exclusive_scan(thrust::device, d_dev.offsets, d_dev.offsets + array_size, d_dev.offsets);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

    printf("Generating output\n");
	unsigned int total_symbols = 0;
	cudaStatus = cudaMemcpy(&total_symbols, d_dev.offsets + array_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
    total_symbols += input[header_size + array_size - 1];

	output_size = symbol_size * total_symbols;
    cudaStatus = cudaMalloc((void**)&d_dev.output, output_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

	rlDecompress << <array_size / threads_per_block + 1, threads_per_block >> > (d_dev.offsets, array_size, d_dev.input + header_size, symbol_size, d_dev.output);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlDecompress launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlDecompress!\n", cudaStatus);
        return cudaStatus;
    }

    output = new unsigned char[output_size + symbol_size];
	cudaStatus = cudaMemcpy(output, d_dev.output, output_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

    while (remaining_bytes > 0)
    {
        output[output_size++] = input[input_size - remaining_bytes--];
    }

    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "    " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms\n";
    start_time = end_time;

    return cudaSuccess;
}
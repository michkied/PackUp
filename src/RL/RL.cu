#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "RL/kernels.h"
#include "RL/RL.h"
#include "thrust/functional.h"
#include <thrust/execution_policy.h>
#include "thrust/scan.h"

cudaError_t run_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size, unsigned int parameter)
{
    unsigned char* dev_input;
    unsigned char* dev_output;
    unsigned int* dev_A;
	unsigned int* dev_B;

    unsigned int* dev_output_counts;
    unsigned char* dev_output_symbols;
    unsigned int* dev_output_repetitions;
    unsigned int* dev_output_repetitions_scan;

    int partition_size = 255;
    unsigned int symbol_size = parameter; // max 255
	unsigned int threads_per_block = 256;
	unsigned int symbol_count = input_size / symbol_size;

    // God had no hand in the creation of this abhorrence
    //unsigned int array_size = 0;
    //while (array_size < symbol_count)
    //    array_size += threads_per_block * 2;

    cudaError_t cudaStatus;

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
    cudaStatus = cudaMalloc((void**)&dev_A, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }

    // Copy input
	cudaStatus = cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
        goto Cleanup;
	}

    // Prepare neighbor arrays
	fprintf(stderr, "Preparing neighbor arrays\n");
	rlNeighborArrays << <symbol_count / threads_per_block + 1, threads_per_block >> > (dev_input, input_size, symbol_size, dev_A, dev_B);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlNeighborArrays launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlNeighborArrays!\n", cudaStatus);
        goto Cleanup;
    }
    fprintf(stderr, "   Done\n");

    // Calculate scan
    fprintf(stderr, "Scan\n");
	thrust::inclusive_scan(thrust::device, dev_B, dev_B + symbol_count, dev_B);
    fprintf(stderr, "   Done\n");

    // Calculate segmented scan
    fprintf(stderr, "Scan by key\n");
    thrust::inclusive_scan_by_key(thrust::device, dev_B, dev_B + symbol_count, dev_A, dev_A, thrust::equal_to<unsigned int>{}, thrust::plus<unsigned int>{});
    fprintf(stderr, "   Done\n");

	// Collect results
    fprintf(stderr, "Collecting results\n");
    unsigned int bound;
    cudaStatus = cudaMemcpy(&bound, dev_B + symbol_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_counts, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_symbols, bound * symbol_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_repetitions, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_repetitions_scan, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }

	rlCollectResults << <symbol_count / threads_per_block + 1, threads_per_block >> > (dev_input, input_size, symbol_size, dev_A, dev_B, dev_output_counts, dev_output_symbols, partition_size, dev_output_repetitions);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlCollectResults launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlCollectResults!\n", cudaStatus);
        goto Cleanup;
    }
    fprintf(stderr, "   Done\n");

    // Generate output
    fprintf(stderr, "Generating output\n");
    thrust::exclusive_scan(thrust::device, dev_output_repetitions, dev_output_repetitions + bound, dev_output_repetitions_scan);

    unsigned int adjusted_bound, temp;
    cudaMemcpy(&adjusted_bound, dev_output_repetitions_scan + bound - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp, dev_output_repetitions + bound - 1, sizeof(int), cudaMemcpyDeviceToHost);
    adjusted_bound += temp;

    cudaStatus = cudaMalloc((void**)&dev_output, adjusted_bound + adjusted_bound * symbol_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }

    rlGenerateOutput << <bound / threads_per_block + 1, threads_per_block >> > (bound, symbol_size, dev_output_symbols, dev_output_counts, partition_size, dev_output_repetitions, dev_output_repetitions_scan, dev_output);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlGenerateOutput launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlGenerateOutput!\n", cudaStatus);
        goto Cleanup;
    }
    fprintf(stderr, "   Done\n");

    // Copy result
    unsigned int header_size = 5;
    unsigned int remaining_symbols_size = input_size % symbol_size;
    unsigned int gpu_output_size = adjusted_bound + adjusted_bound * symbol_size;
    output_size = gpu_output_size + header_size + remaining_symbols_size;

    output = new unsigned char[output_size];
    cudaStatus = cudaMemcpy(output + header_size, dev_output, gpu_output_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Cleanup;
    }
    output[0] = symbol_size;
    std::memcpy(output + 1, &adjusted_bound, 4);
    for (int i = 0; i < remaining_symbols_size; i++)
    {
        output[output_size - remaining_symbols_size + i] = input[input_size - remaining_symbols_size + i];
    }


    Cleanup:
    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_output_counts);
    cudaFree(dev_output_symbols);
    cudaFree(dev_A);
    cudaFree(dev_B);
	cudaFree(dev_output_repetitions);
	cudaFree(dev_output_repetitions_scan);

    return cudaStatus;
}

cudaError_t run_length_decompress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size)
{
	unsigned int threads_per_block = 256;
    unsigned int header_size = 5;
    unsigned int symbol_size = input[0];
    unsigned int array_size;
    std::memcpy(&array_size, input + 1, 4);
    unsigned int remaining_bytes = input_size - header_size - array_size - array_size * symbol_size;

	unsigned char* dev_input;
    unsigned int* dev_offsets;
	unsigned char* dev_output;
	cudaError_t cudaStatus;
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
	cudaStatus = cudaMalloc((void**)&dev_offsets, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }

	// Copy input
	cudaStatus = cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	rlPrepareForScan << <array_size / threads_per_block + 1, threads_per_block >> > (array_size, dev_input + header_size, dev_offsets);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlPrepareForScan launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlPrepareForScan!\n", cudaStatus);
        goto Cleanup;
    }

	thrust::exclusive_scan(thrust::device, dev_offsets, dev_offsets + array_size, dev_offsets);

	unsigned int* debug = new unsigned int[array_size];
	cudaStatus = cudaMemcpy(debug, dev_offsets, array_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

	unsigned int total_symbols = 0;
	cudaStatus = cudaMemcpy(&total_symbols, dev_offsets + array_size - 1, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}
    total_symbols += input[header_size + array_size - 1];

	output_size = symbol_size * total_symbols;
    cudaStatus = cudaMalloc((void**)&dev_output, output_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Cleanup;
    }

    // __global__ void rlDecompress(unsigned int* offsets, unsigned int array_size, unsigned char* input, unsigned int symbol_size, unsigned char* output);
	rlDecompress << <array_size / threads_per_block + 1, threads_per_block >> > (dev_offsets, array_size, dev_input + header_size, symbol_size, dev_output);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rlDecompress launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Cleanup;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlDecompress!\n", cudaStatus);
        goto Cleanup;
    }

    output = new unsigned char[output_size + symbol_size];
	cudaStatus = cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Cleanup;
	}

    while (remaining_bytes > 0)
    {
        output[output_size++] = input[input_size - remaining_bytes--];
    }

    //output_size = 0;
    //output = new unsigned char[symbol_size * array_size * 255];
    //for (unsigned int symbol_index = 0; symbol_index < array_size; ++symbol_index)
    //{
    //    for (int rep = 0; rep < input[header_size + symbol_index]; ++rep)
    //    {
    //        for (int byte = 0; byte < symbol_size; ++byte)
    //        {
    //            output[output_size] = input[header_size + array_size + symbol_index * symbol_size + byte];
    //            output_size++;
    //        }
    //    }
    //}
    //while (remaining_bytes > 0)
    //{
    //    output[output_size++] = input[input_size - remaining_bytes--];
    //}
    
Cleanup:
	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaFree(dev_offsets);

    return cudaSuccess;
}
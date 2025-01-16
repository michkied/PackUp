#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "RL/kernels.h"
#include "RL/RL.h"
#include "thrust/functional.h"
#include <thrust/execution_policy.h>
#include "thrust/scan.h"

cudaError_t run_length_compress(unsigned char* input, long unsigned int input_size, std::vector<unsigned char>& output) {
    unsigned char* dev_input;
    unsigned int* dev_output_counts;
    unsigned char* dev_output_symbols;
    unsigned int* dev_A;
	unsigned int* dev_B;
    unsigned int bound;

    unsigned int symbol_size = 1;
	unsigned int threads_per_block = 256;
	unsigned int symbol_count = input_size / symbol_size;

    unsigned int array_size = 1;
    while (array_size < symbol_count)
        array_size *= 2;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_input, input_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_counts, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_symbols, symbol_count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_A, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        return cudaStatus;
    }

	cudaStatus = cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
		cudaFree(dev_A);
		cudaFree(dev_B);
		return cudaStatus;
	}

	fprintf(stderr, "Preparing neighbor arrays\n");
	rlCompressKernel << <symbol_count / threads_per_block + 1, threads_per_block >> > (dev_input, input_size, symbol_size, dev_A, dev_B);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    fprintf(stderr, "   Done\n");

    unsigned int debug[100];
    cudaStatus = cudaMemcpy(debug, dev_B, symbol_count * sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Scan\n");
	//thrust::inclusive_scan(thrust::device, dev_B, dev_B + symbol_count, dev_B);
	/*rlPrescan << < symbol_count / threads_per_block + 1, threads_per_block, symbol_count * sizeof(int) >> > (dev_B, dev_B_scan, symbol_count);*/

    rlPrescan << < 1, array_size / 2, array_size * sizeof(int) >> > (dev_B, dev_B, array_size);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    fprintf(stderr, "   Done\n");
	cudaStatus = cudaMemcpy(debug, dev_B, symbol_count * sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Scan by key\n");
    thrust::inclusive_scan_by_key(thrust::device, dev_B, dev_B + symbol_count, dev_A, dev_A, thrust::equal_to<unsigned int>{}, thrust::plus<unsigned int>{});
    fprintf(stderr, "   Done\n");

    fprintf(stderr, "Collecting results\n");
	rlCollectResults << <symbol_count / threads_per_block + 1, threads_per_block >> > (dev_input, input_size, symbol_size, dev_A, dev_B, dev_output_counts, dev_output_symbols);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    fprintf(stderr, "   Done\n");

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&bound, dev_B + symbol_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    unsigned int* host_output_counts = new unsigned int[bound];
    unsigned char* host_output_symbols = new unsigned char[bound * symbol_size];
    cudaStatus = cudaMemcpy(host_output_counts, dev_output_counts, bound * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(host_output_symbols, dev_output_symbols, bound * symbol_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }

    cudaFree(dev_input);
    cudaFree(dev_output_counts);
    cudaFree(dev_output_symbols);
    cudaFree(dev_A);
    cudaFree(dev_B);

	//output.push_back(bound);

    // Push counts
    fprintf(stderr, "Partitioning & pushing counts\n");
    int partition_size = 255;
	int* repetitions = new int[bound];
	for (unsigned int i = 0; i < bound; i++) {
		repetitions[i] = 0;
		while (host_output_counts[i] > partition_size)
        {
			output.push_back(partition_size);
			host_output_counts[i] -= partition_size;
			repetitions[i]++;
		}
        if (host_output_counts[i] > 0)
        {
			output.push_back(host_output_counts[i]);
			repetitions[i]++;
        }
	}
    fprintf(stderr, "   Done\n");

    // Push symbols
    fprintf(stderr, "Pushing symbols\n");
	for (unsigned int i = 0; i < bound; i++)
    {
		for (int rep = 0; rep < repetitions[i]; rep++)
		{
			for (unsigned int j = 0; j < symbol_size; j++)
			{
				output.push_back(host_output_symbols[i * symbol_size + j]);
			}
		}
	}
    fprintf(stderr, "   Done\n");
    delete[] repetitions;

	// If symbol size doesn't evenly divide input size, push the remaining symbols
	int remaining_symbols = input_size % symbol_size;
	for (int i = 0; i < remaining_symbols; i++)
	{
		output.push_back(input[input_size - remaining_symbols + i]);
	}

    return cudaStatus;
}
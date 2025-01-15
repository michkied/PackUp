#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "RL/kernels.h"
#include "RL/RL.h"
#include "thrust/functional.h"
#include <thrust/execution_policy.h>
#include "thrust/scan.h"

cudaError_t run_length_compress(std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    unsigned char* dev_input;
    unsigned int* dev_output_counts;
    unsigned char* dev_output_symbols;
    unsigned int* dev_A;
	unsigned int* dev_B;
	unsigned int host_output_counts[100];
	unsigned char host_output_symbols[100];
    unsigned char bound;

    unsigned int symbol_size = 3;
	unsigned int symbol_count = input.size() / symbol_size;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_input, input.size());
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
    cudaStatus = cudaMalloc((void**)&dev_A, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, symbol_count * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        return cudaStatus;
    }

	cudaStatus = cudaMemcpy(dev_input, input.data(), input.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
		cudaFree(dev_A);
		cudaFree(dev_B);
		return cudaStatus;
	}

	rlCompressKernel << <1, input.size() / symbol_size >> > (dev_input, input.size(), symbol_size, dev_A, dev_B);

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

	thrust::inclusive_scan(thrust::device, dev_B, dev_B + input.size(), dev_B);
    thrust::inclusive_scan_by_key(thrust::device, dev_B, dev_B + input.size(), dev_A, dev_A, thrust::equal_to<unsigned char>{}, thrust::plus<unsigned char>{});

	rlCollectResults << <1, input.size() / symbol_size >> > (dev_input, input.size(), symbol_size, dev_A, dev_B, dev_output_counts, dev_output_symbols);
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

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(&bound, dev_B + symbol_count - 1, 1, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(host_output_counts, dev_output_counts, symbol_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(host_output_symbols, dev_output_symbols, symbol_count, cudaMemcpyDeviceToHost);
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

    int partition_size = 255;
	int* repetitions = new int[bound];
	for (int i = 0; i < bound; i++) {
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

	for (int i = 0; i < bound; i++)
    {
		for (int rep = 0; rep < repetitions[i]; rep++)
		{
			for (int j = 0; j < symbol_size; j++)
			{
				output.push_back(host_output_symbols[i * symbol_size + j]);
			}
		}
	}

	delete[] repetitions;

    return cudaStatus;
}
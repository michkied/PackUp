#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "RL/kernels.h"
#include "RL/RL.h"
#include "thrust/functional.h"
#include <thrust/execution_policy.h>
#include "thrust/scan.h"

cudaError_t run_length_compress(unsigned char* input, long unsigned int input_size, unsigned char*& output, long unsigned int& output_size) {
    unsigned char* dev_input;
    unsigned int* dev_output_counts;
    unsigned char* dev_output_symbols;
    unsigned int* dev_output_repetitions;
    unsigned int* dev_output_repetitions_scan;
    unsigned char* dev_output;
    unsigned int* dev_A;
	unsigned int* dev_B;
    unsigned int bound;

    unsigned int symbol_size = 3;
	unsigned int threads_per_block = 256;
	unsigned int symbol_count = input_size / symbol_size;

    //unsigned int array_size = 0;
    //while (array_size < symbol_count)
    //    array_size += threads_per_block * 2;

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlCompressKernel!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    fprintf(stderr, "   Done\n");

    //unsigned int debug[100];
    //cudaStatus = cudaMemcpy(debug, dev_B, symbol_count * sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Scan\n");
	thrust::inclusive_scan(thrust::device, dev_B, dev_B + symbol_count, dev_B);

    // TODO: check if own implementation is faster

	///*rlPrescan << < symbol_count / threads_per_block + 1, threads_per_block, symbol_count * sizeof(int) >> > (dev_B, dev_B_scan, symbol_count);*/

 //   //for (int i = 0; i < symbol_count; i += threads_per_block * 2)
 //   //{
 //   //    rlScan << < 1, threads_per_block, threads_per_block * 2 * sizeof(int) >> > (dev_B + i, threads_per_block * 2);
 //   //    cudaStatus = cudaGetLastError();
 //   //    if (cudaStatus != cudaSuccess) {
 //   //        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
 //   //        cudaFree(dev_input);
 //   //        cudaFree(dev_output_counts);
 //   //        cudaFree(dev_output_symbols);
 //   //        cudaFree(dev_A);
 //   //        cudaFree(dev_B);
 //   //        return cudaStatus;
 //   //    }
 //   //}

 //   rlScan << < array_size / 2 / threads_per_block, threads_per_block, threads_per_block * 2 * sizeof(int) >> > (dev_B, threads_per_block * 2);
 //   cudaStatus = cudaGetLastError();
 //   if (cudaStatus != cudaSuccess) {
 //       fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
 //       cudaFree(dev_input);
 //       cudaFree(dev_output_counts);
 //       cudaFree(dev_output_symbols);
 //       cudaFree(dev_A);
 //       cudaFree(dev_B);
 //       return cudaStatus;
 //   }

 //   cudaStatus = cudaDeviceSynchronize();
 //   if (cudaStatus != cudaSuccess) {
 //       fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
 //       cudaFree(dev_input);
 //       cudaFree(dev_output_counts);
 //       cudaFree(dev_output_symbols);
 //       cudaFree(dev_A);
 //       cudaFree(dev_B);
 //       return cudaStatus;
 //   }
    fprintf(stderr, "   Done\n");
 //   unsigned int debug2[100];
	//cudaStatus = cudaMemcpy(debug2, dev_B, symbol_count * sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(stderr, "Scan by key\n");
    thrust::inclusive_scan_by_key(thrust::device, dev_B, dev_B + symbol_count, dev_A, dev_A, thrust::equal_to<unsigned int>{}, thrust::plus<unsigned int>{});
    fprintf(stderr, "   Done\n");

    fprintf(stderr, "Collecting results\n");
    cudaStatus = cudaMemcpy(&bound, dev_B + symbol_count - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMalloc((void**)&dev_output_repetitions, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_output_repetitions_scan, bound * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        return cudaStatus;
    }

    int partition_size = 255;
	rlCollectResults << <symbol_count / threads_per_block + 1, threads_per_block >> > (dev_input, input_size, symbol_size, dev_A, dev_B, dev_output_counts, dev_output_symbols, partition_size, dev_output_repetitions);
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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlCollectResults!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    fprintf(stderr, "   Done\n");

    fprintf(stderr, "Generating output\n");
    thrust::exclusive_scan(thrust::device, dev_output_repetitions, dev_output_repetitions + bound, dev_output_repetitions_scan);

    unsigned int adjusted_bound, temp;
    cudaMemcpy(&adjusted_bound, dev_output_repetitions_scan + bound - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp, dev_output_repetitions + bound - 1, sizeof(int), cudaMemcpyDeviceToHost);
    adjusted_bound += temp;

    cudaStatus = cudaMalloc((void**)&dev_output, adjusted_bound + adjusted_bound * symbol_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        return cudaStatus;
    }
    rlGenerateOutput << <bound / threads_per_block + 1, threads_per_block >> > (bound, symbol_size, dev_output_symbols, dev_output_counts, partition_size, dev_output_repetitions, dev_output_repetitions_scan, dev_output);
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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rlGenerateOutput!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output_counts);
        cudaFree(dev_output_symbols);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }

    fprintf(stderr, "   Done\n");

    output_size = adjusted_bound + adjusted_bound * symbol_size;
    output = new unsigned char[output_size];
    cudaStatus = cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost);
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

    return cudaStatus;
}
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
    unsigned char* dev_output;
    unsigned char* dev_A;
	unsigned char* dev_B;
	unsigned char host_data[100];
    unsigned char bound;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
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
    cudaStatus = cudaMalloc((void**)&dev_output, input.size() * 2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_A, input.size());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, input.size());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(dev_A);
        return cudaStatus;
    }

	cudaStatus = cudaMemcpy(dev_input, input.data(), input.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_input);
        cudaFree(dev_output);
		cudaFree(dev_A);
		cudaFree(dev_B);
		return cudaStatus;
	}

	rlCompressKernel << <1, input.size()>> > (dev_input, input.size(), dev_A, dev_B);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_input);
        cudaFree(dev_output);
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
        cudaFree(dev_output);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }

	thrust::inclusive_scan(thrust::device, dev_B, dev_B + input.size(), dev_B);
    thrust::inclusive_scan_by_key(thrust::device, dev_B, dev_B + input.size(), dev_A, dev_A, thrust::equal_to<unsigned char>{}, thrust::plus<unsigned char>{});

	rlCollectResults << <1, input.size() >> > (dev_input, input.size(), dev_A, dev_B, dev_output);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_data, dev_output, input.size() * 2, cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(&bound, dev_B+input.size()-1, 1, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(dev_A);
        cudaFree(dev_B);
        return cudaStatus;
    }

    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_A);
    cudaFree(dev_B);

	for (int i = 0; i < bound * 2; i++) {
		output.push_back(host_data[i]);
	}

    return cudaStatus;
}
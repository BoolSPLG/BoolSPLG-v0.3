/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// std::system includes
#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

//================================================================================================

/* This sample queries the properties of the CUDA devices present in the system
* via CUDA Runtime API. */

int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
	int device) {
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		fprintf(
			stderr,
			"cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		DEVICE_RESET
			// Make sure we call CUDA Device Reset before exiting
			exit(EXIT_FAILURE);
	}
}

// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x30, 192 },
		{ 0x32, 192 },
		{ 0x35, 192 },
		{ 0x37, 192 },
		{ 0x50, 128 },
		{ 0x52, 128 },
		{ 0x53, 128 },
		{ 0x60, 64 },
		{ 0x61, 128 },
		{ 0x62, 128 },
		{ 0x70, 64 },
		{ 0x72, 64 },
		{ 0x75, 64 },
		{ -1, -1 } };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}
// end of GPU Architecture definitions

//================================================================================================
/*
* This is a simple test program to measure the memcopy bandwidth of the GPU.
* It can measure device to device copy bandwidth, host to device copy bandwidth
* for pageable and pinned memory, and device to host copy bandwidth for pageable
* and pinned memory.
*
* Usage:
* ./bandwidthTest [option]...
*/
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

void profileCopies(int        *h_a,
	int        *h_b,
	int        *d,
	unsigned int  n,
	char         *desc)
{
	printf("\n%s transfers\n", desc);

	unsigned int bytes = n * sizeof(int);

	// events for timing
	cudaEvent_t startEvent, stopEvent;

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));

	float time;
	checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
	printf("  Host to Device Bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
	printf("  Time (ms): %f\n", time);

	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));

	checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
	printf("  Device to Host Bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
	printf("  Time (ms): %f\n", time);

	for (unsigned int i = 0; i < n; ++i) {
		if (h_a[i] != h_b[i]) {
			printf("*** %s transfers failed ***", desc);
			break;
		}
	}

	// clean up events
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
}

//================================================================================================

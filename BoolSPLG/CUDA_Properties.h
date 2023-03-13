//BoolSPLG CUDA Properties header file
//System includes
#include <stdio.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include "cuda_help_heder.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Print CUDA Properties
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void printDevProp(cudaDeviceProp devProp)
{
	printf("Device Name:                 %s\n", devProp.name);
	printf("Compute capability:          %d.%d\n", devProp.major, devProp.minor);
	printf("Memory Clock Rate (KHz):     %d\n", devProp.memoryClockRate);
	printf("Memory Bus Width (bits):     %d\n", devProp.memoryBusWidth);
	printf("Peak Memory Bandwidth (GB/s):%f\n\n", 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth / 8) / 1.0e6);

	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %zu\n", devProp.totalConstMem);
	printf("Texture alignment:             %zu\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Check CUDA Device and call function to print CUDA Properties
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BoolSPLGCheckProperties()
{
	//check CUDA component status
	cudaError_t cudaStatus;

	//Number of CUDA devices
	int devCount;

	cudaStatus = cudaGetDeviceCount(&devCount);
	//Check status ### cudaGetDeviceCount ###
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	printf("   --- General Information for Nvidia device(s) ---\n");
	printf("CUDA Device Query...\n");
	printf("\nThere are %d CUDA device(s).\n", devCount);

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;

		cudaStatus = cudaGetDeviceProperties(&devProp, i);
		//Check status ### cudaGetDeviceCount ###
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGetDeviceProperties failed!  Do you have a CUDA-capable GPU installed?");
			return;
		}

		//call function to print CUDA properties
		printDevProp(devProp);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Check CUDA Device and call function to print CUDA Properties
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BoolSPLGCheckProperties_v1()
{
//	pArgc = &argc;
//	pArgv = argv;
	printf("\n   --- Extended General Information for Nvidia device(s) ---   \n");
	printf("\n Starting...\n\n"); //printf("%s Starting...\n\n", argv[0]);
	printf(
		"Properties of the CUDA devices present in the system. \n CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf(" There are no available device(s) that support CUDA\n");
	}
	else {
		printf(" Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		//printf("\nCUDA Device #%d: \"%s\"\n", dev, deviceProp.name);
		printf("\nCUDA Device #%d:\n", dev);
		printf("  Device Name:                                   %s\n", deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);


		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("%s", msg);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);
		printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		printf("  Peak Memory Bandwidth (GB/s):                  %f\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth / 8) / 1.0e6);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the
		// CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n",
			memBusWidth);

		printf("  Peak Memory Bandwidth (GB/s):                  %f\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth / 8) / 1.0e6);

		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				L2CacheSize);
		}

#endif

		printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %zu bytes\n",
			deviceProp.memPitch);
		printf("  Texture alignment:                             %zu bytes\n",
			deviceProp.textureAlignment);
		printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown",
			NULL };
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}

	// If there are 2 or more GPUs, query to determine whether RDMA is supported
	if (deviceCount >= 2) {
		cudaDeviceProp prop[64];
		int gpuid[64];  // we want to find the first two GPUs that can support P2P
		int gpu_p2p_count = 0;

		for (int i = 0; i < deviceCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

			// Only boards based on Fermi or later can support P2P
			if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
				// on Windows (64-bit), the Tesla Compute Cluster driver for windows
				// must be enabled to support this
				&& prop[i].tccDriver
#endif
				) {
				// This is an array of P2P capable GPUs
				gpuid[gpu_p2p_count++] = i;
			}
		}

		// Show all the combinations of support P2P GPUs
		int can_access_peer;

		if (gpu_p2p_count >= 2) {
			for (int i = 0; i < gpu_p2p_count; i++) {
				for (int j = 0; j < gpu_p2p_count; j++) {
					if (gpuid[i] == gpuid[j]) {
						continue;
					}
					checkCudaErrors(
						cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
						prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
						can_access_peer ? "Yes" : "No");
				}
			}
		}
	}

	// csv masterlog info
	// *****************************
	// exe and CUDA driver name
	printf("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	// driver version
	sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Runtime version
	sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	// Device count
	sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
	sProfileString += cTemp;
	sProfileString += "\n";
	printf("%s", sProfileString.c_str());

	printf("Result = PASS\n");

	// finish
	//exit(EXIT_SUCCESS);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Check CUDA compute capability Minimal Requires for the Library
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BoolSPLGMinimalRequires()
{
	//check CUDA component status
	cudaError_t cudaStatus;

	//cudaError_t error;
	cudaDeviceProp devProp;
	// Number of CUDA devices
	int devCount;
	cudaStatus = cudaGetDeviceCount(&devCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		// This check all possible CUDA capable device.
		// cudaDeviceProp devPropMR;

		cudaStatus = cudaGetDeviceProperties(&devProp, i);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGetDeviceProperties failed!  Do you have a CUDA-capable GPU installed?");
			return;
		}

		printf("\nMinimal requires compute capability 3.0 or later to run the Library BoolSPLG version 0.3;\n");

		//print Compute capability
		//printf("\nCompute capability:          %d.%d\n", devProp.major, devProp.minor);

		if (devProp.major < 3)
		{
			cout << "Library BoolSPLG requires a GPU with compute capability "
				<< "3.0 or later, exiting..." << endl;
			exit(EXIT_SUCCESS);
		}
	}

	//printf("\nFulfilled minimal requires to run Library BooSPLG version 0.3: \nCompute capability:          %d.%d\n\n", devProp.major, devProp.minor);
	printf("\nFulfilled minimal requires to run Library BooSPLG version 0.3:");
	//printf("\nCompute capability:%d.%d\n", devProp.major, devProp.minor);
	printf("\n  CUDA Capability Major/Minor version number:    %d.%d\n",
		devProp.major, devProp.minor);
	printf("\n Running on... \n  Device: %s\n", devProp.name);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// bandwidthTest
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void bandwidthTest(const unsigned int bytes, int nElements)
{
	cout << "\n[CUDA Bandwidth Test] - Starting...\n\n";
	//int size = 1;
	//cout << "Input memory size for transfer (MB):";
	//cin >> size;
	//unsigned int nElements = size*256*1024;
	////unsigned int nElements = size;
	//const unsigned int bytes = nElements * sizeof(int);
	cout << " Bytes for transfer are:" << bytes << "\n";
	cout << " Number of variables (INT):" << bytes / 4 << "\n";


	// host arrays
	int *h_aPageable, *h_bPageable;
	int *h_aPinned, *h_bPinned;

	// device array
	int *d_a;

	// allocate and initialize
	h_aPageable = (int*)malloc(bytes);                    // host pageable
	h_bPageable = (int*)malloc(bytes);                    // host pageable
	checkCuda(cudaMallocHost((void**)&h_aPinned, bytes)); // host pinned
	checkCuda(cudaMallocHost((void**)&h_bPinned, bytes)); // host pinned
	checkCuda(cudaMalloc((void**)&d_a, bytes));           // device

	for (int i = 0; i < nElements; ++i)
	h_aPageable[i] = i;

	memcpy(h_aPinned, h_aPageable, bytes);
	memset(h_bPageable, 0, bytes);
	memset(h_bPinned, 0, bytes);

	// output device info and transfer size
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));

	printf("\n  Running on... \n\n Device: %s\n", prop.name);
	//printf(" Transfer size (MB): %d\n", bytes / (1024 * 1024));

	// perform copies and report bandwidth
	profileCopies(h_aPageable, h_bPageable, d_a, nElements, (char*)" PAGEABLE Memory");
	profileCopies(h_aPinned, h_bPinned, d_a, nElements, (char*)" PINNED Memory");

	printf("\n");

	// cleanup
	cudaFree(d_a);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	free(h_aPageable);
	free(h_bPageable);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

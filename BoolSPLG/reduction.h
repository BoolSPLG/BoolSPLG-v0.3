// BoolSPLG GPU reduction function heder file
// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <algorithm>

// includes, project
#include "reduction_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

//#define MAX_BLOCK_DIM_SIZE 65535

#ifdef WIN32
#define strcasecmp strcmpi
#endif

extern "C"
bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// CPU function use in GPU max reduction
////////////////////////////////////////////////////////////////////////////////
int reduceCPU_max_libhelp(int *vals, int nvals)
{
	//int cmax = vals[0];

	int cmax = 0;

	for (int i = 0; i<nvals; i++)
	{
		cmax = max(abs(vals[i]), cmax);
//		cout << "max:" << cmax << "\n";
	}

	return cmax;
}
////////////////////////////////////////////////////////////////////////////////
// CPU function use in GPU min reduction
////////////////////////////////////////////////////////////////////////////////
int reduceCPU_min_libhelp(int *vals, int nvals)
{
	//int cmax = vals[0];

	int cmin = 0;

	for (int i = 0; i<nvals; i++)
	{
		cmin = min(abs(vals[i]), cmin);
	}
	return cmin;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// CPU function use in GPU min reduction
////////////////////////////////////////////////////////////////////////////////
int reduceCPU_min_deg(int* vals, int nvals)
{
	//int cmax = vals[0];

	int cmin = 32;

	for (int i = 2; i < nvals; i++)
	{
		cmin = min(abs(vals[i]), cmin);

	}
	return cmin;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

	//get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	if (whichKernel < 3)
	{
		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
		blocks = (n + threads - 1) / threads;
	}
	else
	{
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
	}

	if ((float)threads*blocks >(float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
	{
		printf("n is too large, please choose a smaller number!\n");
	}

	if (blocks > prop.maxGridSize[0])
	{
		printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
			blocks, prop.maxGridSize[0], threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	if (whichKernel == 6)
	{
		blocks = MIN(maxBlocks, blocks);
	}
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction sum of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
int benchmarkReduce(int  n,
	int  numThreads,
	int  numBlocks,
	int  maxThreads,
	int  maxBlocks,
	int  whichKernel,
	bool cpuFinalReduction,
	int  cpuFinalThreshold,
	int* h_odata,
	int* d_idata,
	int* d_odata)
{
	int gpu_result = 0;
	bool needReadBack = true;


	cudaDeviceSynchronize();

	// execute the kernel
	reduce<int>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

	// check if kernel execution generated an error
	//getLastCudaError("Kernel execution failed");
	// Clear d_idata for later use as temporary buffer.
	//cudaMemset(d_idata, 0, n*sizeof(int));

	if (cpuFinalReduction)
	{
		// sum partial sums from each block on CPU
		// copy result from device to host
		cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

		for (int i = 0; i < numBlocks; i++)
		{
			gpu_result += h_odata[i];
		}

		needReadBack = false;
	}
	else
	{
		// sum partial block sums on GPU
		int s = numBlocks;
		int kernel = whichKernel;

		int* d_temp = NULL;
		cudaMalloc((void**)&d_temp, s * sizeof(int));

		while (s > cpuFinalThreshold)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

			//cudaMemcpy(d_idata, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_temp, d_odata, s * sizeof(int), cudaMemcpyDeviceToDevice);

			//reduce(int size, int threads, int blocks,int whichKernel, T *d_idata, T *d_odata)
			reduce<int>(s, threads, blocks, kernel, d_temp, d_odata);

			if (kernel < 3)
			{
				s = (s + threads - 1) / threads;
			}
			else
			{
				s = (s + (threads * 2 - 1)) / (threads * 2);
			}
		}

		if (s > 1)
		{
			// copy result from device to host
			cudaMemcpy(h_odata, d_odata, s * sizeof(int), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
			{
				gpu_result += h_odata[i];
			}

			needReadBack = false;
		}
		cudaFree(d_temp);
	}

	cudaDeviceSynchronize();

	if (needReadBack)
	{
		// copy final sum from device to host
		cudaMemcpy(&gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
	}

	return gpu_result;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction sum test.
////////////////////////////////////////////////////////////////////////////////
int runReduction(int size, int* d_idata)
{
	//int size = 1 << 24;    // number of elements to reduce
	int maxThreads = 256;  // number of threads per block
	int whichKernel = 6;
	int maxBlocks = 64;
	bool cpuFinalReduction = false;
	int cpuFinalThreshold = 1;

	//	printf("%d elements\n", size);
	//	printf("%d threads (max)\n", maxThreads);

	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

	if (numBlocks == 1)
	{
		cpuFinalThreshold = 1;
	}

	// allocate mem for the result on host side
	int* h_odata = (int*)malloc(numBlocks * sizeof(int));

	//printf("%d blocks\n\n", numBlocks);

	// allocate device memory and data
	//T *d_idata = NULL;
	int* d_odata = NULL;

	//checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
	cudaMalloc((void**)&d_odata, numBlocks * sizeof(int));

	int gpu_result = 0;

	gpu_result = benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
		whichKernel, cpuFinalReduction,
		cpuFinalThreshold,
		h_odata, d_idata, d_odata);

	// cleanup
	//free(h_idata);
	free(h_odata);
	//checkCudaErrors(cudaFree(d_idata));
	cudaFree(d_odata);

	return gpu_result;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction max of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
int benchmarkReduceMax(int  n,
	int  numThreads,
	int  numBlocks,
	int  maxThreads,
	int  maxBlocks,
	int  whichKernel,
	bool cpuFinalReduction,
	int  cpuFinalThreshold,
	int *h_odata,
	int *d_idata,
	int *d_odata)
{
	int gpu_result = 0;
	bool needReadBack = true;


	cudaDeviceSynchronize();

	// execute the kernel
	reduce_max<int>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

	// check if kernel execution generated an error
	//getLastCudaError("Kernel execution failed");
	// Clear d_idata for later use as temporary buffer.
	//cudaMemset(d_idata, 0, n*sizeof(int));

	if (cpuFinalReduction)
	{
		// sum partial sums from each block on CPU
		// copy result from device to host
		cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);

		for (int i = 0; i<numBlocks; i++)
		{
			//gpu_result += h_odata[i];
			gpu_result = max(abs(gpu_result), abs(h_odata[i]));
		}

		needReadBack = false;
	}
	else
	{
		// sum partial block sums on GPU
		int s = numBlocks;
		int kernel = whichKernel;

		int *d_temp = NULL;
		cudaMalloc((void **)&d_temp, s*sizeof(int));

		while (s > cpuFinalThreshold)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
			//cudaMemcpy(d_idata, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_temp, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);

			//reduce(int size, int threads, int blocks,int whichKernel, T *d_idata, T *d_odata)
			reduce_max<int>(s, threads, blocks, kernel, d_temp, d_odata);

			if (kernel < 3)
			{
				s = (s + threads - 1) / threads;
			}
			else
			{
				s = (s + (threads * 2 - 1)) / (threads * 2);
			}
		}

		if (s > 1)
		{
			// copy result from device to host
			cudaMemcpy(h_odata, d_odata, s * sizeof(int), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
			{
				//gpu_result += h_odata[i];
				gpu_result = max(abs(gpu_result), abs(h_odata[i]));
			}

			needReadBack = false;
		}
		cudaFree(d_temp);
	}

	cudaDeviceSynchronize();

	if (needReadBack)
	{
		// copy final sum from device to host
		cudaMemcpy(&gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
	}

	return gpu_result;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction max test.
////////////////////////////////////////////////////////////////////////////////
int runReductionMax(int size, int *d_idata)
{
	//int size = 1 << 24;    // number of elements to reduce
	int maxThreads = 256;  // number of threads per block
	int whichKernel = 6;
	int maxBlocks = 64;
	bool cpuFinalReduction = false;
	int cpuFinalThreshold = 1;

	//	printf("%d elements\n", size);
	//	printf("%d threads (max)\n", maxThreads);

	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

	if (numBlocks == 1)
	{
		cpuFinalThreshold = 1;
	}

	// allocate mem for the result on host side
	int *h_odata = (int *)malloc(numBlocks*sizeof(int));

	//		printf("%d blocks\n\n", numBlocks);

	// allocate device memory and data
	//T *d_idata = NULL;
	int *d_odata = NULL;

	//checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
	cudaMalloc((void **)&d_odata, numBlocks*sizeof(int));

	int gpu_result = 0;

	gpu_result = benchmarkReduceMax(size, numThreads, numBlocks, maxThreads, maxBlocks,
		whichKernel, cpuFinalReduction,
		cpuFinalThreshold,
		h_odata, d_idata, d_odata);

	// cleanup
	//free(h_idata);
	free(h_odata);
	//checkCudaErrors(cudaFree(d_idata));
	cudaFree(d_odata);

	return gpu_result;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction min of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
int benchmarkReduceMin(int  n,
	int  numThreads,
	int  numBlocks,
	int  maxThreads,
	int  maxBlocks,
	int  whichKernel,
	bool cpuFinalReduction,
	int  cpuFinalThreshold,
	int *h_odata,
	int *d_idata,
	int *d_odata)
{
	int gpu_result = 0;
	bool needReadBack = true;


	cudaDeviceSynchronize();

	// execute the kernel
	reduce_min<int>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

	// check if kernel execution generated an error
	//getLastCudaError("Kernel execution failed");
	// Clear d_idata for later use as temporary buffer.
	//cudaMemset(d_idata, 0, n*sizeof(int));

	if (cpuFinalReduction)
	{
		// sum partial sums from each block on CPU
		// copy result from device to host
		cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);

		for (int i = 0; i<numBlocks; i++)
		{
			//gpu_result += h_odata[i];
			gpu_result = min(abs(gpu_result), abs(h_odata[i]));
		}

		needReadBack = false;
	}
	else
	{
		// sum partial block sums on GPU
		int s = numBlocks;
		int kernel = whichKernel;

		int *d_temp = NULL;
		cudaMalloc((void **)&d_temp, s*sizeof(int));

		while (s > cpuFinalThreshold)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

			//cudaMemcpy(d_idata, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_temp, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);

			//reduce(int size, int threads, int blocks,int whichKernel, T *d_idata, T *d_odata)
			reduce_min<int>(s, threads, blocks, kernel, d_temp, d_odata);

			if (kernel < 3)
			{
				s = (s + threads - 1) / threads;
			}
			else
			{
				s = (s + (threads * 2 - 1)) / (threads * 2);
			}
		}

		if (s > 1)
		{
			// copy result from device to host
			cudaMemcpy(h_odata, d_odata, s * sizeof(int), cudaMemcpyDeviceToHost);

			for (int i = 0; i < s; i++)
			{
				//gpu_result += h_odata[i];
				gpu_result = min(abs(gpu_result), abs(h_odata[i]));
			}

			needReadBack = false;
		}
		cudaFree(d_temp);
	}

	cudaDeviceSynchronize();

	if (needReadBack)
	{
		// copy final sum from device to host
		cudaMemcpy(&gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
	}

	return gpu_result;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction min test.
////////////////////////////////////////////////////////////////////////////////
int runReductionMin(int size, int *d_idata)
{
	//int size = 1 << 24;    // number of elements to reduce
	int maxThreads = 256;  // number of threads per block
	int whichKernel = 6;
	int maxBlocks = 64;
	bool cpuFinalReduction = false;
	int cpuFinalThreshold = 1;

	//	printf("%d elements\n", size);
	//	printf("%d threads (max)\n", maxThreads);

	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

	if (numBlocks == 1)
	{
		cpuFinalThreshold = 1;
	}

	// allocate mem for the result on host side
	int *h_odata = (int *)malloc(numBlocks*sizeof(int));

	//		printf("%d blocks\n\n", numBlocks);

	// allocate device memory and data
	//T *d_idata = NULL;
	int *d_odata = NULL;

	//checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
	cudaMalloc((void **)&d_odata, numBlocks*sizeof(int));

	int gpu_result = 0;

	gpu_result = benchmarkReduceMin(size, numThreads, numBlocks, maxThreads, maxBlocks,
		whichKernel, cpuFinalReduction,
		cpuFinalThreshold,
		h_odata, d_idata, d_odata);

	// cleanup
	//free(h_idata);
	free(h_odata);
	//checkCudaErrors(cudaFree(d_idata));
	cudaFree(d_odata);

	return gpu_result;
}
////////////////////////////////////////////////////////////////////////////////

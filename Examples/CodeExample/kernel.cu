///////////////////////////////////////////////////////////////////////////
// Copyright @2017-2022 Dusan and Iliya.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////
//C++ standard library 
#include <stdio.h>
#include <iostream>

#include <chrono>

// CUDA runtime.
#include "cuda_runtime.h"

//Main Library header file
#include <BoolSPLG/BoolSPLG_v03.cuh>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//@@ Example using of GPU function for computing Walsh spectra of Boolean function
/////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "==========================================================";
	printf("\nExample using of GPU function for computing Walsh spectra of Boolean function BoolSPLG.\n");
	cout << "==========================================================";
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@BoolSPLG Properties Library, Function to check if GPU fulfill BoolSPLG CUDA-capable requires
	BoolSPLGMinimalRequires();
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	cout << "==========================================================\n";
	printf("The example Boolean function will randomly generate with length according to the input exponent. \n");
	//@@Set size of Boolean vector
	cout << "\nInput 'exponent' for power function (base is 2). \n";
	cout << " ==> The input exponent can be between 6 - 20.\n";
	cout << "\nInput exponent:";
	int size, exponent;
	cin >> exponent;
	size = (int)(pow(2, exponent));
	printf("The size length is: 2^%d=:%d\n", exponent, size);
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Declaration, host_Vect_CPU - vector for CPU computation, host_Vect_GPU -vector for GPU computation

	//Allocate memory block. Allocates a block of size bytes of memory
	int *host_Vect_PTT = (int*)malloc(sizeof(int) * size);
	int *walshvec_cpu = (int*)malloc(sizeof(int) * size);
	int *host_Vect_rez = (int*)malloc(sizeof(int) * size);
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Function: Fill random input vector for computation
	Fill_dp_vector(size, host_Vect_PTT, host_Vect_PTT);
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start measuring time CPU
	auto startFWTCPU = chrono::steady_clock::now();
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Function: Fast Walsh Transformation function CPU (W_f(f))
	FastWalshTrans(size, host_Vect_PTT, walshvec_cpu);
	int Lin_cpu = reduceCPU_max(walshvec_cpu, size);
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Stop measuring time and calculate the elapsed time
	auto endFWTCPU = chrono::steady_clock::now();
	cout << "==========================================================";
	cout << "\nCPU Elapsed time in milliseconds (CPU - Lin(f)): ";
	cout << chrono::duration_cast<chrono::milliseconds>(endFWTCPU - startFWTCPU).count() << " ms" << endl;
	cout << "CPU Elapsed time in nanoseconds (CPU - Lin(f)): ";
	cout << chrono::duration_cast<chrono::nanoseconds>(endFWTCPU - startFWTCPU).count() << " ns" << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////
	 
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Set Device size array
	int sizeBoolean = sizeof(int) * size;
	//Declaration device vectors
	int* device_Vect, * device_Vect_rez;

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Set and Call Boolean Fast Walsh Transform
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory here

	//check CUDA component status
	cudaError_t cudaStatus;

	//input device vector
	cudaStatus = cudaMalloc((void**)&device_Vect, sizeBoolean);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	//output device vector
	cudaStatus = cudaMalloc((void**)&device_Vect_rez, sizeBoolean);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time HostToDevice
	cudaEvent_t startHToD_PTT, stopHToD_PTT;
	cudaEventCreate(&startHToD_PTT);
	cudaEventCreate(&stopHToD_PTT);

	cudaEventRecord(startHToD_PTT);
	/////////////////////////////////////////////////////////////////////////////////////////////////

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Vect, host_Vect_PTT, sizeBoolean, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time HostToDevice
	cudaEventRecord(stopHToD_PTT);
	cudaEventSynchronize(stopHToD_PTT);
	float elapsedTimeHToD_PTT = 0;
	cudaEventElapsedTime(&elapsedTimeHToD_PTT, startHToD_PTT, stopHToD_PTT);
	cout << "\n============================================\n";
	printf("\n(GPU HostToDevice) Time taken to copy PTT(f) (int): %3.6f ms \n", elapsedTimeHToD_PTT);
	/////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library function for FWT(f) calculation
	/////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute FWT GPU
	cudaEvent_t startTimeFWT, stopTimeFWT;

	cudaEventCreate(&startTimeFWT);
	cudaEventCreate(&stopTimeFWT);

	cudaEventRecord(startTimeFWT);
	/////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, Boolean Fast Walsh Transform: return Walsh Spectra and Lin(f)
	/////////////////////////////////////////////////////////////////////////////////////////////////
	int Lin_gpu = WalshSpecTranBoolGPU_ButterflyMax(device_Vect, device_Vect_rez, size, true);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute FWT GPU
	cudaEventRecord(stopTimeFWT);
	cudaEventSynchronize(stopTimeFWT);
	float elapsedTimeFWT = 0;
	cudaEventElapsedTime(&elapsedTimeFWT, startTimeFWT, stopTimeFWT);

	cout << "\n============================================\n";
	printf("\n(GPU) Time taken to Compute Lin(S): %3.6f ms \n", elapsedTimeFWT);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and 
	//returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	exit(EXIT_FAILURE);
	}

	//Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_Vect_rez, device_Vect_rez, sizeBoolean, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Compare, Check result
	cout << "\nCheck result FWT(f) -> CPU vs. GPU:";
	check_rez(size, host_Vect_rez, walshvec_cpu);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "\nLin(f)_cpu:" << Lin_cpu << "\n";
	cout << "Lin(f)_gpu:" << Lin_gpu << "\n";
	cout << "\n============================================\n";
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	printf("\n   --- End Example, Boolean function BoolSPLG. ---\n\n");
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//@Free memory
	cudaFree(device_Vect);
	cudaFree(device_Vect_rez);

	free(host_Vect_PTT);
	free(host_Vect_rez);
	free(walshvec_cpu);

	return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////


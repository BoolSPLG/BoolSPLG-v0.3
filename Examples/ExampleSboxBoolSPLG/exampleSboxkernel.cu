////////////////////////////////////////////////////////////////////////////
// Copyright @2017-2023 Dusan and Iliya.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////
// Standard library 
#include <stdio.h>
#include <iostream>

#include <chrono>

// CUDA runtime.
#include "cuda_runtime.h"

//Main Library header file
#include <BoolSPLG/BoolSPLG_v03.cuh>
//#include "BoolSPLG_v03.cuh"


//Declaration for global variables and dinamic array use in S-box functions
int sizeSbox, binary = 0;
int* SboxElemet, * SboxElemet_ANF;

bool CheckFile = true;

//Help Heder file - CPU computing S-box functions properties
#include "funct_Sbox_CPU.h"

//Functions declarations
void SetBinary();
void SetParSbox(string filename);
void readFromFileMatPerm(string filename, int* Element);

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
//S-box main Function, example using of GPU - CPU function for computing properties
//////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	cout << "==========================================================";
	printf("\nExample S-box BoolSPLG (version 0.3) Library Algorithms.\n");
	cout << "==========================================================";
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@BoolSPLG Properties Library, Function to check if GPU fulfill BoolSPLG CUDA-capable requires
	BoolSPLGMinimalRequires();
	//////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "==========================================================\n";
	printf("Compute Properties of input S-box.\n");

	//string use for name of the S-box input file
	string Sbox = "sbox"; //String for name of the file with inverse permutation and cyclic matrix

	//Call function in "helpSboxfunct.h" input S-box (permutation file)
	SetParSbox(Sbox); //Permutation file have number of element plus 1

	if (CheckFile == false)
	{
		printf("\nInital file is not set...\n");
		printf("\n   --- End Example, S-box function BoolSPLG (version 0.3) Library algorithms. ---\n\n");

		return 0;
	}
	//Set paramether to compute
	cout << "Configuration for next parameter: \n";
	cout << "Size S-box: " << sizeSbox << "\n";
	cout << "Binary: " << binary - 1 << "\n";

	SboxElemet = (int*)malloc(sizeof(int) * sizeSbox); //Allocate memory for S-box (permutation)
	SboxElemet_ANF = (int*)malloc(sizeof(int) * sizeSbox); //Allocate memory for S-box (ANF form permutation)

	//Call function in "helpSboxfunct.h" open S-box (permutation) file
	readFromFileMatPerm(Sbox, SboxElemet);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//menu for printing S-box
	//cout << "\n==========================================================\n";
	////Print input S-box or not print
	//int input;
	//cout << "\nInput 1 (to print input S-box):";
	//cout << "\nInput any character (not print S-box): \n";
	//cout << "\nInput:";
	//cin >> input;
	//cout << "\n==========================================================\n";

	//if (input == 1)
	//{
	//	cout << "\nPrint input S-box:\n"; //Print

	//	//function for print, header file "funct_Sbox_CPU.h"
	//	Print_Result(sizeSbox, SboxElemet);
	//	cout << "==========================================================\n";
	//}
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Declaration and Allocate memory blocks
	int** STT = AllocateDynamicArray<int>(binary, sizeSbox);
	int* binary_num = (int*)malloc(sizeof(int) * binary);

	SetSTT(SboxElemet, STT, binary_num, sizeSbox, binary);
	SetS_ANF(SboxElemet, SboxElemet_ANF, sizeSbox, binary);
	cout << "\n==========================================================\n";
	printf("\n(CPU) Compute Properties of the S-box.\n");

	//Computing S-box properties ##HeaderSboxProperties.h##
	MainSboxProperties(STT, SboxElemet);
	cout << "\n==========================================================\n";
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Start GPU program
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//set size array
	int sizeSboxArray = sizeof(int) * sizeSbox;

	//device vectors
	int* device_Sbox, * device_Sbox_ANF; // , *device_CF, *device_Vect_out;

	//check CUDA component status
	cudaError_t cudaStatus;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory here
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//input S-box device vector
	cudaStatus = cudaMalloc((void**)&device_Sbox, sizeSboxArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	//input S-box device vector
	cudaStatus = cudaMalloc((void**)&device_Sbox_ANF, sizeSboxArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time Host To Device
	cudaEvent_t startHToDSbox, stopHToDSbox;

	cudaEventCreate(&startHToDSbox);
	cudaEventCreate(&stopHToDSbox);

	cudaEventRecord(startHToDSbox);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// Copy S-box input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Sbox, SboxElemet, sizeSboxArray, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time Host To Device
	cudaEventRecord(stopHToDSbox);
	cudaEventSynchronize(stopHToDSbox);
	float elapsedTimeHToDSbox = 0;
	cudaEventElapsedTime(&elapsedTimeHToDSbox, startHToDSbox, stopHToDSbox);

	printf("\n(GPU HostToDevice) Time taken to copy S-box: %3.6f ms \n", elapsedTimeHToDSbox);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	printf("  Time taken to copy S-box (sec): %f\n", elapsedTimeHToDSbox * 1e-3);
	printf("  S-box elements: %d, Transfer Size (Bytes): %d \n", sizeSbox, sizeSboxArray);
	printf("  Host to Device Bandwidth (GB/s): %f\n\n", sizeSboxArray * 1e-6 / elapsedTimeHToDSbox);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy S-box input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_Sbox_ANF, SboxElemet_ANF, sizeSboxArray, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Component function
	//////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		int* device_CF;

		//CF S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		int* CPU_CF = (int*)malloc(sizeof(int) * sizeSbox * sizeSbox);
		int* host_CF = (int*)malloc(sizeof(int) * sizeSbox * sizeSbox);

		//@set GRID
		int sizethread = sizeSbox;
		int sizeblok = sizeSbox;

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Start measuring time CPU
		auto startComponentFuncCPU = chrono::steady_clock::now();
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//Compute Component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== all CF are save in one array CPU_STT ======================
			GenTTComponentFunc(i, SboxElemet, CPU_CF, sizeSbox);
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@ Stop measuring time and calculate the elapsed time
		//////////////////////////////////////////////////////////////////////////////////////////////////
		auto endComponentFuncCPU = chrono::steady_clock::now();
		cout << "==========================================================";
		cout << "\nCompute Component function.\n";
		cout << "\nCPU Elapsed time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(endComponentFuncCPU - startComponentFuncCPU).count() << " ms" << endl;
		cout << "CPU Elapsed time in nanoseconds : " << chrono::duration_cast<chrono::nanoseconds>(endComponentFuncCPU - startComponentFuncCPU).count() << " ns" << endl;
		/////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Start mesure time compute Component function GPU
		cudaEvent_t startComponentFn, stopComponentFn;

		cudaEventCreate(&startComponentFn);
		cudaEventCreate(&stopComponentFn);

		cudaEventRecord(startComponentFn);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Compute Component function GPU - BoolSPLG Library function
		ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, sizeSbox);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Stop mesure time compute Component function GPU
		cudaEventRecord(stopComponentFn);
		cudaEventSynchronize(stopComponentFn);
		float elapsedTimeHComponentFn = 0;
		cudaEventElapsedTime(&elapsedTimeHComponentFn, startComponentFn, stopComponentFn);

		printf("\n(GPU) Time taken to Compute Component function: %3.6f ms \n", elapsedTimeHComponentFn);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(host_CF, device_CF, sizeArray, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(EXIT_FAILURE);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Compere, Check result
		cout << "Compute Component functions.\n";
		cout << "\nCheck result Component functions -> CPU vs. GPU:";
		check_rez(sizeSbox * sizeSbox, CPU_CF, host_CF);
		cout << "\n==========================================================\n";
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Free memory
		cudaFree(device_CF);

		free(CPU_CF);
		free(host_CF);
		//////////////////////////////////////////////////////////////////////////////////////////////////
	}
	else
	{
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Declaration and Alocation of memory blocks
		//////////////////////////////////////////////////////////////////////////////////////////////////
		int sizeArray = sizeof(int) * sizeSbox;

		int* device_CF;

		//CF S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		int* CPU_CF = (int*)malloc(sizeof(int) * sizeSbox);
		int* host_CF = (int*)malloc(sizeof(int) * sizeSbox);

		//@set GRID
		int sizethread = BLOCK_SIZE;
		int sizeblok = sizeSbox / BLOCK_SIZE;

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Start measuring time CPU
		auto startComponentFuncCPU = chrono::steady_clock::now();
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//@@ Compute component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== One CF is save in array CPU_STT ===========================
			GenTTComponentFuncVec(i, SboxElemet, CPU_CF, sizeSbox);
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@ Stop measuring time and calculate the elapsed time
		auto endComponentFuncCPU = chrono::steady_clock::now();
		//////////////////////////////////////////////////////////////////////////////////////////////////
		cout << "==========================================================";
		cout << "\nCompute Component function.\n";
		cout << "\nCPU Elapsed time in milliseconds : " << chrono::duration_cast<chrono::milliseconds>(endComponentFuncCPU - startComponentFuncCPU).count() << " ms" << endl;
		cout << "CPU Elapsed time in nanoseconds : " << chrono::duration_cast<chrono::nanoseconds>(endComponentFuncCPU - startComponentFuncCPU).count() << " ns" << endl;
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Start mesure time compute Component function GPU
		cudaEvent_t startComponentFn, stopComponentFn;

		cudaEventCreate(&startComponentFn);
		cudaEventCreate(&stopComponentFn);

		cudaEventRecord(startComponentFn);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@ Compute component function GPU
		//////////////////////////////////////////////////////////////////////////////////////////////////
		for (int i = 0; i < sizeSbox; i++)
		{
			//@Compute Component function GPU - BoolSPLG Library function
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, i);
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Stop mesure time compute Component function GPU
		cudaEventRecord(stopComponentFn);
		cudaEventSynchronize(stopComponentFn);
		float elapsedTimeHComponentFn = 0;
		cudaEventElapsedTime(&elapsedTimeHComponentFn, startComponentFn, stopComponentFn);

		printf("\n(GPU) Time taken to Compute Component function: %3.6f ms \n", elapsedTimeHComponentFn);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(host_CF, device_CF, sizeArray, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(EXIT_FAILURE);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Compare, Check result
		//////////////////////////////////////////////////////////////////////////////////////////////////
		cout << "Compute Component functions.\n";
		cout << "\nCheck result (last) Component functions -> CPU vs. GPU:";
		check_rez(sizeSbox, CPU_CF, host_CF);
		cout << "\n==========================================================\n";
		//////////////////////////////////////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Free memory
		cudaFree(device_CF);

		free(CPU_CF);
		free(host_CF);
		//////////////////////////////////////////////////////////////////////////////////////////////////
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Linear Approximation Table (LAT) - Linearity Lin(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Compute Linearity of S-box, Lin(S).\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Declaration of device vectors
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int* device_CF, * device_WS;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_WS, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_WS, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, Walsh Transform W(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute FWT GPU
	cudaEvent_t startTimeFWT, stopTimeFWT;

	cudaEventCreate(&startTimeFWT);
	cudaEventCreate(&stopTimeFWT);

	cudaEventRecord(startTimeFWT);
	//////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Fast Walsh Transform: return Lin(S) and Walsh Spectra
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Lin_gpu = WalshSpecTranSboxGPU(device_Sbox, device_CF, device_LAT, sizeSbox);
	int Lin_gpu = WalshSpecTranSboxGPU_ButterflyMax_v03(device_Sbox, device_CF, device_WS, sizeSbox, true);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute FWT GPU
	cudaEventRecord(stopTimeFWT);
	cudaEventSynchronize(stopTimeFWT);
	float elapsedTimeFWT = 0;
	cudaEventElapsedTime(&elapsedTimeFWT, startTimeFWT, stopTimeFWT);

	printf("\n(GPU) Time taken to Compute Lin(S): %3.6f ms \n", elapsedTimeFWT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_WS);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Linear Approximation Table - Linearity) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Walsh Spectra W(S) - Linearity (Lin(S)) -> CPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int* CPU_CF_PTT = (int*)malloc(sizeof(int) * sizeSbox);
	int* CPU_WHT = (int*)malloc(sizeof(int) * sizeSbox);

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start measuring time CPU
	auto startFWTCPU = chrono::steady_clock::now();
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//(CPU) Compute Linearity, Lin(S) 
	int Lin_cpu = 0, Lin_return = 0;
	for (int i = 1; i < sizeSbox; i++)
	{
		//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
		//===== One CF is save in array CPU_STT ===========================
		GenPTTComponentFuncVec(i, SboxElemet, CPU_CF_PTT, sizeSbox);
		FastWalshTrans(sizeSbox, CPU_CF_PTT, CPU_WHT);	//Find Walsh spectra on one row

		Lin_return = reduceCPU_max(CPU_WHT, sizeSbox);

		if (Lin_cpu < Lin_return)
			Lin_cpu = Lin_return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Stop measuring time and calculate the elapsed time
	auto endFWTCPU = chrono::steady_clock::now();

	cout << "\n\nCPU Elapsed time in milliseconds Lin(S): " << chrono::duration_cast<chrono::milliseconds>(endFWTCPU - startFWTCPU).count() << " ms" << endl;
	cout << "CPU Elapsed time in nanoseconds Lin(S): " << chrono::duration_cast<chrono::nanoseconds>(endFWTCPU - startFWTCPU).count() << " ns" << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@free memory
	free(CPU_CF_PTT);
	free(CPU_WHT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Print result
	cout << "\nPrint result Linearity of S-box, Lin(S):\n";
	cout << "\n(CPU) Lin(S):" << Lin_cpu << "\n";
	cout << "(GPU) Lin(S):" << Lin_gpu << "\n";
	cout << "\n==========================================================\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Linear Approximation Table (LAT)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Compute Linear Approximation Table LAT(S).\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Declaration of device vectors
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int* device_CF_PTT, * device_LAT;
	int sizeArray;
	int* host_LAT;

	if (sizeSbox <= BLOCK_SIZE)
	{
		host_LAT = (int*)malloc(sizeof(int) * sizeSbox * sizeSbox);

		//@Declaration and Alocation of memory blocks
		sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF_PTT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_LAT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, Linear Approximation Table LAT(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute LAT GPU
	cudaEvent_t startTimeLAT, stopTimeLAT;

	cudaEventCreate(&startTimeLAT);
	cudaEventCreate(&stopTimeLAT);

	cudaEventRecord(startTimeLAT);
	//////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Fast Walsh Transform: return Lin(S) and Walsh Spectra
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	LATSboxGPU_v03(device_Sbox, device_CF, device_LAT, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute LAT(S) GPU
	cudaEventRecord(stopTimeLAT);
	cudaEventSynchronize(stopTimeLAT);
	float elapsedTimeLAT = 0;
	cudaEventElapsedTime(&elapsedTimeLAT, startTimeLAT, stopTimeLAT);

	printf("\n(GPU) Time taken to Compute LAT(S): %3.6f ms \n", elapsedTimeLAT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_LAT, device_LAT, sizeArray, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF_PTT);
	cudaFree(device_LAT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Linear Approximation Table - Linearity) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//(CPU) Compute Linear Aproximation Table LAT(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox <= 1024)
	{
		int* LAT_CPU = (int*)malloc(sizeof(int) * sizeSbox * sizeSbox);

		int* CPU_CF = (int*)malloc(sizeof(int) * sizeSbox);
		int* CPU_FWT = (int*)malloc(sizeof(int) * sizeSbox);
		/////////////////////////////////////////////////////////////////////////////////////////////////////
		//@@Start measuring time CPU
		auto startLATCPU = chrono::steady_clock::now();
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		//(CPU) Compute Linearity, Lin(S) 
		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== One CF is save in array CPU_STT ===========================
			GenPTTComponentFuncVec(i, SboxElemet, CPU_CF, sizeSbox);
			FastWalshTrans(sizeSbox, CPU_CF, CPU_FWT);	//Find Walsh spectra on one row

			for (int j = 0; j < sizeSbox; j++)
			{
				//LAT_CPU[j * sizeSbox + i] = CPU_FWT[j] / 2;
				LAT_CPU[j + sizeSbox * i] = CPU_FWT[j] / 2;
			}
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		//@Compere, Check result
		cout << "\nCheck result LAT(S) -> CPU vs. GPU:";
		check_rez(sizeSbox * sizeSbox, LAT_CPU, host_LAT);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////////////////////////////////
		//@@ Stop measuring time and calculate the elapsed time
		auto endLATCPU = chrono::steady_clock::now();

		cout << "\n\nCPU Elapsed time in milliseconds LAT(S): " << chrono::duration_cast<chrono::milliseconds>(endLATCPU - startLATCPU).count() << " ms" << endl;
		cout << "CPU Elapsed time in nanoseconds LAT(S): " << chrono::duration_cast<chrono::nanoseconds>(endLATCPU - startLATCPU).count() << " ns" << endl;
		cout << "\n==========================================================\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////////

		//@free memory
		free(CPU_CF);
		free(CPU_FWT);
		free(LAT_CPU);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	else
	{
		cout << "\nIs not implemented LAT(S) function for S-box size n>10.";
		cout << "\n==========================================================\n";
	 }
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - max Algebraic Degree, deg(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Compute Algebraic Degree (max) of S-box, deg(S).\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Declaration of ANF device vector
	int* device_ANF;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_ANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_ANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF (GPU)
	cudaEvent_t startTimeANF, stopTimeANF;

	cudaEventCreate(&startTimeANF);
	cudaEventCreate(&stopTimeANF);

	cudaEventRecord(startTimeANF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Algebraic Degree: return deg(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmax_gpu = AlgebraicDegreeSboxGPU_ButterflyMax(device_Sbox, device_CF, device_ANF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF (GPU)
	cudaEventRecord(stopTimeANF);
	cudaEventSynchronize(stopTimeANF);
	float elapsedTimeANF = 0;
	cudaEventElapsedTime(&elapsedTimeANF, startTimeANF, stopTimeANF);

	printf("\n(GPU) Time taken to Compute deg(S) (max): %3.6f ms \n", elapsedTimeANF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_ANF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree, deg(S) (max) Bitwise
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int NumOfBits = sizeof(unsigned long long int) * 8, NumInt;
	
	//@Declaration of ANF device vector
	unsigned long long int* device_NumIntVecCF, * device_NumIntVecANF;
	int * device_Vec_max_values;

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits << "\n";
		NumInt = (sizeSbox * sizeSbox) / NumOfBits;
		cout << "NumOfInt:" << NumInt << "\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_NumIntVecANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values, sizeof(int) * NumInt);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072)) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computation: \n";
		cout << "NumOfBits:" << NumOfBits << "\n";
		NumInt = sizeSbox / NumOfBits;
		cout << "NumOfInt:" << NumInt << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_NumIntVecANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values, sizeof(int) * NumInt * sizeSbox);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits << "\n";
		NumInt = sizeSbox / NumOfBits;
		cout << "NumOfInt:" << NumInt << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_NumIntVecANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values, sizeof(int) * NumInt);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF bitwise GPU
	cudaEvent_t startTimeANFbit, stopTimeANFbit;

	cudaEventCreate(&startTimeANFbit);
	cudaEventCreate(&stopTimeANFbit);

	cudaEventRecord(startTimeANFbit);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box bitwise Algebraic Degree: return deg(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmax_bitwise_gpu = BitwiseAlgebraicDegreeSboxGPU_ButterflyMax_v03(device_NumIntVecCF, device_NumIntVecANF, device_Vec_max_values, device_Sbox, device_CF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF bitwise GPU
	cudaEventRecord(stopTimeANFbit);
	cudaEventSynchronize(stopTimeANFbit);
	float elapsedTimeANFbit = 0;
	cudaEventElapsedTime(&elapsedTimeANFbit, startTimeANFbit, stopTimeANFbit);

	printf("\n(GPU) Time taken to Compute deg(S)(bitwise): %3.6f ms \n", elapsedTimeANFbit);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_NumIntVecCF);
	cudaFree(device_NumIntVecANF);
	cudaFree(device_Vec_max_values);
	cudaFree(device_CF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree (bitwise) S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Degree (deg(S)) from input Algebraic Normal Form (ANF) max -> GPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Declaration and Alocation of memory blocks
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF (GPU)
	cudaEvent_t startTimeANF_in, stopTimeANF_in;

	cudaEventCreate(&startTimeANF_in);
	cudaEventCreate(&stopTimeANF_in);

	cudaEventRecord(startTimeANF_in);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Algebraic Degree: return deg(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ANF_in_ADmax_gpu = AlgebraicDegreeSboxGPU_in_ANF_ButterflyMax(device_Sbox_ANF, device_CF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF (GPU)
	cudaEventRecord(stopTimeANF_in);
	cudaEventSynchronize(stopTimeANF_in);
	float elapsedTimeANF_in = 0;
	cudaEventElapsedTime(&elapsedTimeANF_in, startTimeANF_in, stopTimeANF_in);

	printf("\n(GPU) Time taken to Compute deg(S) - (ANF 'in'): %3.6f ms \n", elapsedTimeANF_in);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree, deg(S) (max) Bitwise (ANF 'in')
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int NumOfBits_in = sizeof(unsigned long long int) * 8, NumInt_in;

	//@Declaration of ANF device vector
	unsigned long long int* device_NumIntVecCF_in;
	int * device_Vec_max_values_in;

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits_in << "\n";
		NumInt_in = (sizeSbox * sizeSbox) / NumOfBits_in;
		cout << "NumOfInt:" << NumInt_in << "\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////


		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt_in;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_in, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_in, sizeof(int) * NumInt_in);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072)) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computation: \n";
		cout << "NumOfBits:" << NumOfBits_in << "\n";
		NumInt_in = sizeSbox / NumOfBits_in;
		cout << "NumOfInt:" << NumInt_in << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray_in = sizeof(unsigned long long int) * NumInt_in * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_in, sizeArray_in);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_in, sizeof(int) * NumInt_in * sizeSbox);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits_in << "\n";
		NumInt_in = sizeSbox / NumOfBits_in;
		cout << "NumOfInt:" << NumInt_in << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray_in = sizeof(unsigned long long int) * NumInt_in;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_in, sizeArray_in);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_in, sizeof(int) * NumInt_in);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF bitwise GPU
	cudaEvent_t startTimeANFbit_in, stopTimeANFbit_in;

	cudaEventCreate(&startTimeANFbit_in);
	cudaEventCreate(&stopTimeANFbit_in);

	cudaEventRecord(startTimeANFbit_in);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box bitwise Algebraic Degree: return deg(S) (ANF 'in')
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ANF_in_ADmax_bitwise_gpu = BitwiseAlgebraicDegreeSboxGPU_ANF_in_ButterflyMax_v03(device_NumIntVecCF_in, device_Vec_max_values_in, device_Sbox_ANF, device_CF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF bitwise GPU
	cudaEventRecord(stopTimeANFbit_in);
	cudaEventSynchronize(stopTimeANFbit_in);
	float elapsedTimeANFbit_in = 0;
	cudaEventElapsedTime(&elapsedTimeANFbit_in, startTimeANFbit_in, stopTimeANFbit_in);

	printf("\n(GPU) Time taken to Compute deg(S)(bitwise) (ANF 'in'): %3.6f ms \n", elapsedTimeANFbit_in);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_NumIntVecCF_in);
	cudaFree(device_Vec_max_values_in);
	cudaFree(device_CF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree (bitwise) S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree (deg(S)) max -> CPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int* CPU_CF_TT = (int*)malloc(sizeof(int) * sizeSbox);
	int* CPU_ANF = (int*)malloc(sizeof(int) * sizeSbox);

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start measuring time CPU
	auto startANFCPU = chrono::steady_clock::now();
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//(CPU) Compute Algebraic Degree (Max), deg(S)
	int ADmax_cpu = 0, AD_return_max = 0;
	for (int i = 1; i < sizeSbox; i++)
	{
		//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
		//===== One CF is save in array CPU_STT ===========================
		GenTTComponentFuncVec(i, SboxElemet, CPU_CF_TT, sizeSbox);

		//Function: Fast Mobiush Transformation function CPU
		FastMobiushTrans(sizeSbox, CPU_CF_TT, CPU_ANF);
		AD_return_max = AlgDegMax(sizeSbox, CPU_ANF);

		if (ADmax_cpu < AD_return_max)
			ADmax_cpu = AD_return_max;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Stop measuring time and calculate the elapsed time
	auto endANFCPU = chrono::steady_clock::now();

	cout << "\n\nCPU Elapsed time in milliseconds deg(S): " << chrono::duration_cast<chrono::milliseconds>(endANFCPU - startANFCPU).count() << " ms" << endl;
	cout << "CPU Elapsed time in nanoseconds deg(S): " << chrono::duration_cast<chrono::nanoseconds>(endANFCPU - startANFCPU).count() << " ns" << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	free(CPU_CF_TT);
	free(CPU_ANF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Print result
	cout << "\nPrint result Algebraic Degree (max), deg(S):\n";
	cout << "\n(CPU) deg(S):" << ADmax_cpu << "\n";
	cout << "(GPU) deg(S):" << ADmax_gpu << "\n";
	cout << "(GPU) deg(S) (ANF 'in'):" << ANF_in_ADmax_gpu << "\n";
	cout << "(GPU bitwise) deg(S):" << ADmax_bitwise_gpu << "\n";
	cout << "(GPU bitwise) deg(S) (ANF 'in'):" << ANF_in_ADmax_bitwise_gpu << "\n";
	cout << "\n==========================================================\n";

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - max Algebraic Degree (deg(S))
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Compute Algebraic Degree (min) of S-box, deg(S).\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Declaration of ANF device vector
	//int* device_ANF;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_ANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_ANF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF (GPU)
	cudaEvent_t startTimeANF_min, stopTimeANF_min;

	cudaEventCreate(&startTimeANF_min);
	cudaEventCreate(&stopTimeANF_min);

	cudaEventRecord(startTimeANF_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Algebraic Degree: return deg(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmin_gpu = AlgebraicDegreeSboxGPU_ButterflyMin(device_Sbox, device_CF, device_ANF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF (GPU)
	cudaEventRecord(stopTimeANF_min);
	cudaEventSynchronize(stopTimeANF_min);
	float elapsedTimeANF_min = 0;
	cudaEventElapsedTime(&elapsedTimeANF_min, startTimeANF_min, stopTimeANF_min);

	printf("\n(GPU) Time taken to Compute deg(S) (min): %3.6f ms \n", elapsedTimeANF_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_ANF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Degree (deg(S)) from input Algebraic Normal Form (ANF) min -> GPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Declaration and Alocation of memory blocks
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF (GPU)
	cudaEvent_t startTimeANF_in_min, stopTimeANF_in_min;

	cudaEventCreate(&startTimeANF_in_min);
	cudaEventCreate(&stopTimeANF_in_min);

	cudaEventRecord(startTimeANF_in_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Algebraic Degree: return deg(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ANF_in_ADmin_gpu = AlgebraicDegreeSboxGPU_in_ANF_ButterflyMin(device_Sbox_ANF, device_CF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF (GPU)
	cudaEventRecord(stopTimeANF_in_min);
	cudaEventSynchronize(stopTimeANF_in_min);
	float elapsedTimeANF_in_min = 0;
	cudaEventElapsedTime(&elapsedTimeANF_in_min, startTimeANF_in_min, stopTimeANF_in_min);

	printf("\n(GPU) Time taken to Compute deg(S) (min) - (ANF 'in'): %3.6f ms \n", elapsedTimeANF_in_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree, deg(S) (min) Bitwise
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int NumOfBits_min = sizeof(unsigned long long int) * 8, NumInt_min;
	//@Declaration of ANF device vector
	unsigned long long int* device_NumIntVecCF_min, * device_NumIntVecANF_min;
	int* device_Vec_max_values_min;

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits_min << "\n";
		NumInt_min = (sizeSbox * sizeSbox) / NumOfBits_min;
		cout << "NumOfInt:" << NumInt_min << "\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt_min;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_min, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_NumIntVecANF_min, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_min, sizeof(int) * NumInt_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072)) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computation: \n";
		cout << "NumOfBits:" << NumOfBits_min << "\n";
		NumInt_min = sizeSbox / NumOfBits_min;
		cout << "NumOfInt:" << NumInt_min << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt_min * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_min, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_NumIntVecANF_min, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_min, sizeof(int) * NumInt_min * sizeSbox);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits_min << "\n";
		NumInt_min = sizeSbox / NumOfBits_min;
		cout << "NumOfInt:" << NumInt_min << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(unsigned long long int) * NumInt_min;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_min, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_NumIntVecANF_min, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_min, sizeof(int) * NumInt_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF bitwise GPU
	cudaEvent_t startTimeANFbit_min, stopTimeANFbit_min;

	cudaEventCreate(&startTimeANFbit_min);
	cudaEventCreate(&stopTimeANFbit_min);

	cudaEventRecord(startTimeANFbit_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box bitwise Algebraic Degree: return deg(S) min
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmin_bitwise_gpu = BitwiseAlgebraicDegreeSboxGPU_ButterflyMin_v03(device_NumIntVecCF_min, device_NumIntVecANF_min, device_Vec_max_values_min, device_Sbox, device_CF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF bitwise GPU
	cudaEventRecord(stopTimeANFbit_min);
	cudaEventSynchronize(stopTimeANFbit_min);
	float elapsedTimeANFbit_min = 0;
	cudaEventElapsedTime(&elapsedTimeANFbit_min, startTimeANFbit_min, stopTimeANFbit_min);

	printf("\n(GPU) Time taken to Compute deg(S) (min) (bitwise): %3.6f ms \n", elapsedTimeANFbit_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_NumIntVecCF_min);
	cudaFree(device_NumIntVecANF_min);
	cudaFree(device_Vec_max_values_min);

	cudaFree(device_CF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree (bitwise) S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree, deg(S) (min) Bitwise (ANF 'in')
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int NumOfBits_in_min = sizeof(unsigned long long int) * 8, NumInt_in_min;

	//@Declaration of ANF device vector
	unsigned long long int* device_NumIntVecCF_in_min;
	int* device_Vec_max_values_in_min;

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits_in_min << "\n";
		NumInt_in_min = (sizeSbox * sizeSbox) / NumOfBits_in_min;
		cout << "NumOfInt:" << NumInt_in_min << "\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////


		//@Declaration and Alocation of memory blocks
		int sizeArray_in_min = sizeof(unsigned long long int) * NumInt_in_min;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_in_min, sizeArray_in_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_in_min, sizeof(int) * NumInt_in_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072)) //limitation cоme from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computation: \n";
		cout << "NumOfBits:" << NumOfBits_in_min << "\n";
		NumInt_in_min = sizeSbox / NumOfBits_in_min;
		cout << "NumOfInt:" << NumInt_in_min << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray_in_min = sizeof(unsigned long long int) * NumInt_in_min * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_in_min, sizeArray_in_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_in_min, sizeof(int) * NumInt_in_min * sizeSbox);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//print bitwise paramethers
		cout << "\nUsed paramethers for bitwise computations: \n";
		cout << "NumOfBits:" << NumOfBits_in_min << "\n";
		NumInt_in_min = sizeSbox / NumOfBits_in_min;
		cout << "NumOfInt:" << NumInt_in_min << " (for every component function)\n";
		/////////////////////////////////////////////////////////////////////////////////////////////////

		//@Declaration and Alocation of memory blocks
		int sizeArray_in_min = sizeof(unsigned long long int) * NumInt_in_min;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_NumIntVecCF_in_min, sizeArray_in_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_Vec_max_values_in_min, sizeof(int) * NumInt_in_min);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		//@Declaration and Alocation of memory blocks - v0.3
		int sizeArray_v03 = sizeof(int) * sizeSbox;

		//CF of S-box device vector - v0.3
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray_v03);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ANF(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ANF bitwise GPU
	cudaEvent_t startTimeANFbit_in_min, stopTimeANFbit_in_min;

	cudaEventCreate(&startTimeANFbit_in_min);
	cudaEventCreate(&stopTimeANFbit_in_min);

	cudaEventRecord(startTimeANFbit_in_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box bitwise Algebraic Degree: return deg(S) (ANF 'in')
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ANF_in_ADmin_bitwise_gpu = BitwiseAlgebraicDegreeSboxGPU_ANF_in_ButterflyMin_v03(device_NumIntVecCF_in_min, device_Vec_max_values_in_min, device_Sbox_ANF, device_CF, sizeSbox);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ANF bitwise GPU
	cudaEventRecord(stopTimeANFbit_in_min);
	cudaEventSynchronize(stopTimeANFbit_in_min);
	float elapsedTimeANFbit_in_min = 0;
	cudaEventElapsedTime(&elapsedTimeANFbit_in_min, startTimeANFbit_in_min, stopTimeANFbit_in_min);

	printf("\n(GPU) Time taken to Compute deg(S)(bitwise) (min) (ANF 'in'): %3.6f ms \n", elapsedTimeANFbit_in_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_NumIntVecCF_in_min);
	cudaFree(device_Vec_max_values_in_min);
	cudaFree(device_CF);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Algebraic Degree (bitwise) S-box) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////


	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Algebraic Normal Form (ANF) - Algebraic Degree (deg(S)) min -> CPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int* CPU_CF_TT_min = (int*)malloc(sizeof(int) * sizeSbox);
	int* CPU_ANF_min = (int*)malloc(sizeof(int) * sizeSbox);

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start measuring time CPU
	auto startANFCPU_min = chrono::steady_clock::now();
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	int ADmin_cpu = sizeSbox;
	//(CPU) Compute Algebraic Degree (Min), deg(S)
	for (int i = 1; i < sizeSbox; i++)
	{
		//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
		//===== One CF is save in array CPU_STT ===========================
		GenTTComponentFuncVec(i, SboxElemet, CPU_CF_TT_min, sizeSbox);

		//Function: Fast Mobiush Transformation function CPU
		FastMobiushTrans(sizeSbox, CPU_CF_TT_min, CPU_ANF_min);
		AD_return_max = AlgDegMax(sizeSbox, CPU_ANF_min);

		if (ADmin_cpu > AD_return_max)
			ADmin_cpu = AD_return_max;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Stop measuring time and calculate the elapsed time
	auto endANFCPU_min = chrono::steady_clock::now();

	cout << "\n\nCPU Elapsed time in milliseconds deg(S) (min): " << chrono::duration_cast<chrono::milliseconds>(endANFCPU_min - startANFCPU_min).count() << " ms" << endl;
	cout << "CPU Elapsed time in nanoseconds deg(S) (min): " << chrono::duration_cast<chrono::nanoseconds>(endANFCPU_min - startANFCPU_min).count() << " ns" << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	free(CPU_CF_TT_min);
	free(CPU_ANF_min);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Print result
	cout << "\nPrint result Algebraic Degree (min), deg(S):\n";
	cout << "\n(CPU) deg(S):" << ADmin_cpu << "\n";
	cout << "(GPU) deg(S):" << ADmin_gpu << "\n";
	cout << "(GPU) deg(S) (ANF 'in'):" << ANF_in_ADmin_gpu << "\n";
	cout << "(GPU bitwise) deg(S):" << ADmin_bitwise_gpu << "\n";
	cout << "(GPU bitwise) deg(S) (ANF 'in'):" << ANF_in_ADmin_bitwise_gpu << "\n";
	cout << "\n==========================================================\n";

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Autocorrelation Transform (ACT) - Autocorrelation (AC)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Compute Autocorrelation of S-box, AC(S).\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Declaration of device vectors
	int* device_ACT;

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//CF and ACT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_ACT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//CF and LAT of S-box device vector
		cudaStatus = cudaMalloc((void**)&device_CF, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}

		cudaStatus = cudaMalloc((void**)&device_ACT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ACT(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute ACT (GPU)
	cudaEvent_t startTimeACT, stopTimeACT;

	cudaEventCreate(&startTimeACT);
	cudaEventCreate(&stopTimeACT);

	cudaEventRecord(startTimeACT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Autocorrelation Transform: return AC(S) and Autocorrelation spectra
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//AC_gpu = AutocorrelationTranSboxGPU(device_Sbox, device_CF, device_ACT, sizeSbox);
	int AC_gpu = AutocorrelationTranSboxGPU_ButterflyMax_v03(device_Sbox, device_CF, device_ACT, sizeSbox, true);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute ACT (GPU)
	cudaEventRecord(stopTimeACT);
	cudaEventSynchronize(stopTimeACT);
	float elapsedTimeACT = 0;
	cudaEventElapsedTime(&elapsedTimeACT, startTimeACT, stopTimeACT);

	printf("\n(GPU) Time taken to Compute AC(S): %3.6f ms \n", elapsedTimeACT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_CF);
	cudaFree(device_ACT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (ACT(S) function) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Autocorrelation Transform (ACT) - Autocorrelation (AC) -> CPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	int* CPU_CF_PTT_ACT = (int*)malloc(sizeof(int) * sizeSbox);
	int* CPU_WHT_ACT = (int*)malloc(sizeof(int) * sizeSbox);

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start measuring time CPU
	auto startACTCPU = chrono::steady_clock::now();
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//(CPU) Compute Autocorrelation of S-box, AC(S)
	int AC_cpu = 0, AC_return = 0;
	for (int i = 1; i < sizeSbox; i++)
	{
		//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
		//===== One CF is save in array CPU_STT ===========================
		GenPTTComponentFuncVec(i, SboxElemet, CPU_CF_PTT_ACT, sizeSbox);

		//Function: Autocorelation Transformation function CPU
		FastWalshTrans(sizeSbox, CPU_CF_PTT_ACT, CPU_WHT_ACT);	//Find Walsh spectra on one row
		fun_pow2(sizeSbox, CPU_WHT_ACT);
		FastWalshTransInv(sizeSbox, CPU_WHT_ACT);
		AC_return = reduceCPU_AC(sizeSbox, CPU_WHT_ACT);

		if (AC_cpu < AC_return)
			AC_cpu = AC_return;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Stop measuring time and calculate the elapsed time
	auto endACTCPU = chrono::steady_clock::now();

	cout << "\n\nCPU Elapsed time in milliseconds AC(S): " << chrono::duration_cast<chrono::milliseconds>(endACTCPU - startACTCPU).count() << " ms" << endl;
	cout << "CPU Elapsed time in nanoseconds AC(S): " << chrono::duration_cast<chrono::nanoseconds>(endACTCPU - startACTCPU).count() << " ns" << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	free(CPU_CF_PTT_ACT);
	free(CPU_WHT_ACT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Print result
	cout << "\nPrint result Autocorrelation, AC(S):";
	cout << "\n(CPU) AC(S):" << AC_cpu << "\n";
	cout << "(GPU) AC(S):" << AC_gpu << "\n";
	cout << "\n==========================================================\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute Difference Distribution Table (DDT) - Differential uniformity (DU)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << "Compute Differential uniformity of S-box, delta(S).\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Declaration device DDT vector
	int* device_DDT;

	//if (sizeSbox <= BLOCK_SIZE)
	if (sizeSbox <= 16384)
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox * sizeSbox;

		//DDT S-box device vector
		cudaStatus = cudaMalloc((void**)&device_DDT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int sizeArray = sizeof(int) * sizeSbox;

		//DDT S-box device vector
		cudaStatus = cudaMalloc((void**)&device_DDT, sizeArray);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			exit(EXIT_FAILURE);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library DDT(S) function
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start mesure time compute DDT (GPU)
	cudaEvent_t startTimeDDT, stopTimeDDT;

	cudaEventCreate(&startTimeDDT);
	cudaEventCreate(&stopTimeDDT);

	cudaEventRecord(startTimeDDT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library, S-box Diffrent Distribution Table function: return diff(S) and DDT(S)
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Diff_gpu = DDTSboxGPU(device_Sbox, device_DDT, sizeSbox);
	int delta_gpu = DDTSboxGPU_ButterflyMax_v03_expand(device_Sbox, device_DDT, sizeSbox, true);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Stop mesure time compute DDT (GPU)
	cudaEventRecord(stopTimeDDT);
	cudaEventSynchronize(stopTimeDDT);
	float elapsedTimeDDT = 0;
	cudaEventElapsedTime(&elapsedTimeDDT, startTimeDDT, stopTimeDDT);

	printf("\n(GPU) Time taken to Compute diff(S): %3.6f ms \n", elapsedTimeDDT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_DDT);
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching (Difference Distribution Table - Differential Uniformity) Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Compute DDT(S)/diff(S) -> CPU
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Start measuring time
	auto startDDTCPU = chrono::steady_clock::now();
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//(CPU) Compute Differential uniformity of S-box, delta(S)
	int delta_cpu = 0;
	for (int row = 1; row < sizeSbox; row++)
	{
		//	DDTFnVec_kernelCPU(SboxElemet, row, sizeSbox);
		int delta = DDT_vector(sizeSbox, SboxElemet, row);

		if (delta > delta_cpu)
			delta_cpu = delta;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	// Stop measuring time and calculate the elapsed time
	auto endDDTCPU = chrono::steady_clock::now();

	cout << "\n\nCPU Elapsed time in milliseconds diff(S): " << chrono::duration_cast<chrono::milliseconds>(endDDTCPU - startDDTCPU).count() << " ms" << endl;
	cout << "CPU Elapsed time in nanoseconds diff(S): " << chrono::duration_cast<chrono::nanoseconds>(endDDTCPU - startDDTCPU).count() << " ns" << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//Print result
	cout << "\nPrint result Differential uniformity, delta(S):";
	cout << "\n(CPU) delta(f):" << delta_cpu << "\n";
	cout << "(GPU) delta(f):" << delta_gpu << "\n";
	cout << "\n==========================================================\n";
	/////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////
	//@Free memory
	cudaFree(device_Sbox);
	cudaFree(device_Sbox_ANF);

	FreeDynamicArray<int>(STT);
	free(binary_num);
	free(SboxElemet);
	free(SboxElemet_ANF);
	//////////////////////////////////////////////////////////////////////////////////////////////////
	printf("\n   --- End Example, S-box function BoolSPLG (version 0.3) Library algorithms. ---\n\n");
	//////////////////////////////////////////////////////////////////////////////////////////////////

	return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//Function set Binary
//////////////////////////////////////////////////////////////////////////////////////////////////
void SetBinary()
{
	int binar = 1;
	while (binar < sizeSbox)
	{
		binar = binar * 2;
		binary++;
	}
	binary++;
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//Function set parametar Size of S-box and Binary
//////////////////////////////////////////////////////////////////////////////////////////////////
void SetParSbox(string filename)
{
	vector <string> words; // Vector to hold our words we read in.
	string str; // Temp string to
	ifstream fin(filename); // Open it up!
	if (fin.is_open())
	{
		//cout << " " << fin.getline(str1, 10,'_') << "\n";
		//file opened successfully so we are here
		cout << "File Contain 'S-box' is Opened successfully!.\n";
		while (fin >> str) // Will read up to eof() and stop at every
		{                  // whitespace it hits. (like spaces!)
			words.push_back(str);
			//InvPerm1[counterPerFrile]=atoi(str.c_str());
		}
		fin.close(); // Close that file!

		//sizeSbox = words.size();
		sizeSbox = static_cast<int>(words.size());
	}
	else //file could not be opened
	{
		cout << "File '" << filename << "' could not be opened." << endl;
		CheckFile = false;
		return;
	}
	//set number of binary element
	SetBinary();
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//Read S-box element from file
//////////////////////////////////////////////////////////////////////////////////////////////////
void readFromFileMatPerm(string filename, int* Element)
{
	int counterPerFrile = 0;
	vector <string> words; // Vector to hold our words we read in.
	string str; // Temp string to
	ifstream fin(filename); // Open it up!
	if (fin.is_open())
	{
		//cout << " " << fin.getline(str1, 10,'_') << "\n";
		//file opened successfully so we are here
		cout << "\nFile '" << filename << "' Opened successfully!.\n";
		while (fin >> str) // Will read up to eof() and stop at every
		{                  // whitespace it hits. (like spaces!)
			words.push_back(str);
			Element[counterPerFrile] = atoi(str.c_str());

			counterPerFrile++;
		}
		fin.close(); // Close that file!
		//int chNumMatrx = counterPerFrile;
		cout << "Number of element into '" << filename << "' file: " << counterPerFrile;
	}
	else //file could not be opened
	{
		cout << "File '" << filename << "' could not be opened." << endl;
		CheckFile = false;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////
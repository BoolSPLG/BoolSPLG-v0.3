//Example Header File for Boolean functions, v0.3
#include <stdio.h>
#include <iostream>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//Boolean Fast Walsh Transforms: return Walsh spectra and Lin(f), v0.3
////////////////////////////////////////////////////////////////////////////////

template <class T>
int
runTestWalshSpec(int size, int* BoolElemet_host, T * vec_host_spectra, bool returnWS)
{
	//Declare and Allocate host memory, v0.3
	//host_Vect_Spectra03 = (T*)malloc(sizeof(T) * size);

	//Declaration device vectors
	T* device_Vect_Spectra03 = NULL;
	T* device_Vect_Max03 = NULL;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@Paramether fot converting input TT into 'int' array
	int NumOfBits32 = sizeof(unsigned int) * 8;
	int NumInt32 = size / NumOfBits32;

	unsigned int sizeVInt32 = sizeof(unsigned int) * NumInt32;

	unsigned int* NumIntVecTT32 = (unsigned int*)malloc(sizeof(unsigned int) * NumInt32);
	int* device_VectInt32;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	BinVecToDec32(NumOfBits32, BoolElemet_host, NumIntVecTT32, NumInt32);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory, v0.3 here
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//check CUDA component status
	cudaError_t cudaStatus;

	//input device vector - boolean Integer function
	cudaStatus = cudaMalloc((void**)&device_VectInt32, sizeVInt32);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//output device vector, v0.3
	cudaStatus = cudaMalloc((void**)&device_Vect_Spectra03, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//Butterfly find Max device vector, v0.3
	cudaStatus = cudaMalloc((void**)&device_Vect_Max03, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input convert TT to int vectors from host memory to GPU buffers.
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMemcpy(device_VectInt32, NumIntVecTT32, sizeVInt32, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FWT(f) two - function diffrent functions for FWT(f) calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int Lin_gpu = 0;
	Lin_gpu = WalshSpecTranBoolGPU_ButterflyMax_v03<T>(device_VectInt32, device_Vect_Spectra03, device_Vect_Max03, size, true); //use Butterfly Max
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	
	if(returnWS) //cudaMemcpyDeviceToHost walsh spectra to host memory
	{ 
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(vec_host_spectra, device_Vect_Spectra03, sizeof(T) * size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			//goto Error;
			exit(EXIT_FAILURE);
		}
	}

	//@Free memory
	//free(host_Vect_Spectra03);
	free(NumIntVecTT32);

	cudaFree(device_VectInt32);
	cudaFree(device_Vect_Spectra03);
	cudaFree(device_Vect_Max03);

	return Lin_gpu;
}

////////////////////////////////////////////////////////////////////////////////
//Boolean Fast Mobius Transforms: TT to ANF and vice versus
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
runTestMobius(int size, int* BoolElemet_host, T* host_Vect_anf_tt03, bool returnMT)
{
	//Declare and Allocate host memory, v0.3
	//T *host_Vect_anf_tt03 = (T *)malloc(sizeof(T)* size);

	//Declaration device vectors
	T* device_Vect_anf_tt03 = NULL;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Paramether fot converting input TT into 'int' array
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int NumOfBits32 = sizeof(unsigned int) * 8;
	int NumInt32 = size / NumOfBits32;

	unsigned int sizeVInt32 = sizeof(unsigned int) * NumInt32;

	unsigned int* NumIntVecTT32 = (unsigned int*)malloc(sizeof(unsigned int) * NumInt32);
	int* device_VectInt32;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	BinVecToDec32(NumOfBits32, BoolElemet_host, NumIntVecTT32, NumInt32);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory v0.3 here
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//check CUDA component status
	cudaError_t cudaStatus;

	//input device vector - boolean Integer function
	cudaStatus = cudaMalloc((void**)&device_VectInt32, sizeVInt32);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//output device vector v0.3
	cudaStatus = cudaMalloc((void**)&device_Vect_anf_tt03, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input convert TT to int vectors from host memory to GPU buffers.
	//////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMemcpy(device_VectInt32, NumIntVecTT32, sizeVInt32, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FMT(f) function
	//////////////////////////////////////////////////////////////////////////////////////////////////
	MobiusTranBoolGPU_v03<T>(device_VectInt32, device_Vect_anf_tt03, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	if (returnMT)  //cudaMemcpyDeviceToHost Mobius Transform (ANF-TT) to host memory
	{
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(host_Vect_anf_tt03, device_Vect_anf_tt03, sizeof(T) * size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(EXIT_FAILURE);
		}
	}

	//@Free memory
	//free(host_Vect_anf_tt03);
	free(NumIntVecTT32);

	cudaFree(device_VectInt32);
	cudaFree(device_Vect_anf_tt03);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Algebraic degree: deg(f)
////////////////////////////////////////////////////////////////////////////////
template <class T>
int
runTestDeg(int size, int* BoolElemet_host)
{
	//Declare and Allocate host memory, v0.3
	//T *host_Vect_anf_tt03 = (T *)malloc(sizeof(T)* size);

	//Declaration device vectors
	T* device_Vect_anf_tt03 = NULL;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Paramether fot converting input TT into 'int' array
	int NumOfBits32 = sizeof(unsigned int) * 8;
	int NumInt32 = size / NumOfBits32;

	unsigned int sizeVInt32 = sizeof(unsigned int) * NumInt32;

	unsigned int* NumIntVecTT32 = (unsigned int*)malloc(sizeof(unsigned int) * NumInt32);
	int* device_VectInt32;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	BinVecToDec32(NumOfBits32, BoolElemet_host, NumIntVecTT32, NumInt32);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory v0.3 here
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//check CUDA component status
	cudaError_t cudaStatus;

	//input device vector - boolean Integer function
	cudaStatus = cudaMalloc((void**)&device_VectInt32, sizeVInt32);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//output device vector v0.3
	cudaStatus = cudaMalloc((void**)&device_Vect_anf_tt03, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input convert TT to int vectors from host memory to GPU buffers.
	//////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMemcpy(device_VectInt32, NumIntVecTT32, sizeVInt32, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FMT(f) function - two diffrent functions for FMT(f) - Algebraic Degree calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int degMax_gpu = 0;
	degMax_gpu = AlgebraicDegreeBoolGPU_ButterflyMax_v03_T<T>(device_VectInt32, device_Vect_anf_tt03, size); //use Butterfly Max
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//@Free memory
	free(NumIntVecTT32);

	cudaFree(device_VectInt32);
	cudaFree(device_Vect_anf_tt03);

	return degMax_gpu;
}
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//Boolean Autocorrelation Transform (ACT): return Autocorrelation Spectra: and AC(f), v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
int
runTestACT(int size, int* BoolElemet_host, T* host_Vect_Spectra03, bool returnACT)
{
	//Declare and Allocate host memory, v0.3
	//T *host_Vect_Spectra03 = (T *)malloc(sizeof(T)* size);

	//Declaration device vectors
	T* device_Vect_Spectra03 = NULL;
	T* device_Vect_Max03 = NULL;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@Paramether fot converting input TT into 'int' array
	int NumOfBits32 = sizeof(unsigned int) * 8;
	int NumInt32 = size / NumOfBits32;

	unsigned int sizeVInt32 = sizeof(unsigned int) * NumInt32;

	unsigned int* NumIntVecTT32 = (unsigned int*)malloc(sizeof(unsigned int) * NumInt32);
	int* device_VectInt32;//, *MaskShare;

	//////////////////////////////////////////////////////////////////////////////////////////////////
	BinVecToDec32(NumOfBits32, BoolElemet_host, NumIntVecTT32, NumInt32);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//@@ Allocate GPU memory, v0.3 here
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//check CUDA component status
	cudaError_t cudaStatus;

	//input device vector - boolean Integer function
	cudaStatus = cudaMalloc((void**)&device_VectInt32, sizeVInt32);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//output device vector v0.3
	cudaStatus = cudaMalloc((void**)&device_Vect_Spectra03, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}

	//Butterfly find Max device vector v0.3
	cudaStatus = cudaMalloc((void**)&device_Vect_Max03, sizeof(T) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy input convert TT to int vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(device_VectInt32, NumIntVecTT32, sizeVInt32, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto Error;
		exit(EXIT_FAILURE);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library ACT(f) function - two diffrent functions for ACT(f) calculation
	//////////////////////////////////////////////////////////////////////////////////////////////////
	int AC_gpu = 0;
	AC_gpu = AutocorrelationTranBoolGPU_ButterflyMax_v03<T>(device_VectInt32, device_Vect_Spectra03, device_Vect_Max03, size, true); //use Butterfly Max
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
		exit(EXIT_FAILURE);
	}

	if (returnACT) //cudaMemcpyDeviceToHost return Autocorrelation Spectra to host memory
	{
		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(host_Vect_Spectra03, device_Vect_Spectra03, sizeof(T) * size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			exit(EXIT_FAILURE);
		}
	}

	//@Free memory
	//free(host_Vect_Spectra03);
	free(NumIntVecTT32);

	cudaFree(device_VectInt32);
	cudaFree(device_Vect_Spectra03);
	cudaFree(device_Vect_Max03);

	return AC_gpu;
}
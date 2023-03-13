//BoolSPLG Boolean device functions
// System includes
#include <stdio.h>
#include <iostream>

//#define BLOCK_SIZE 1024
//using namespace std;

//*************************************************************************************************************
//Global variables
//*************************************************************************************************************

__device__ int global_max = 0, global_min = 32;

//*************************************************************************************************************
//Functions:
//*************************************************************************************************************

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fwt_kernel_shfl_xor_SM(int *VectorValue, int *VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;

	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = (value1)*(__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function: Fast Walsh Transforms, LAT(S) S-box
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fwt_kernel_shfl_xor_SM_LAT(int* VectorValue, int* VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	//unsigned int bid = blockIdx.x;
	unsigned int laneId = blockIdx.x * blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;

	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i < 32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = (value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j < step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid & j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid & j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value/2;
	//VectorValueRez[(tid * step) + bid] = value / 2;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Fast Walsh Transforms v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void fwt_kernel_shfl_xor_SM_v03(int *VectorValue, T *VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;
	/////////////////////////////////////////////////////////////////////////////////
	//convert Integer input and Seed starting value
	unsigned int IdInt32 = laneId / 32;
	unsigned int IdInt32_Rsh = (laneId - (laneId / 32) * 32); //int IdInt32_Rsh = laneId % 32;
	unsigned int value_b = VectorValue[IdInt32];

	//value = (value >> IdInt32_Rsh) % 2;
	value_b = (value_b >> IdInt32_Rsh);
	int element = (value_b - (value_b / 2) * 2);

	//Seed starting value as inverse lane ID
	//int value = VectorValue[laneId];
	int value = -1 * (element - 1 + element); //Set polarity true table vector
	///////////////////////////////////////////////////////////////////////////////

	//__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = (value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fwt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = (value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Fast Walsh Transforms v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void fwt_kernel_shfl_xor_SM_MP_v03(T * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = (value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value);
		value = (value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			value = (value + tmpsdata[tid + j]);
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]);
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Invers Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ifmt_kernel_shfl_xor_SM(int * VectorValue, int * VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = ((value1) * (__shfl_xor(value, i) - value) + (1 - value1) * (__shfl_xor(value, i) + value)) / 2;
		value = ((value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value)) / 2;
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Invers Fast Walsh Transforms v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void ifmt_kernel_shfl_xor_SM_v03(T * VectorValue, T * VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
		value = ((value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value)) / 2;
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Invers Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ifmt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
		value = ((value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value)) / 2;
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Invers Fast Walsh Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void ifmt_kernel_shfl_xor_SM_MP_v03(T * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
		value = ((value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value)) / 2;

	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM(int *VectorValue, int *VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//printf("laneId: %d, tid: %d, value:%d \n", laneId, tid, value);

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_min(int* VectorValue, int* VectorValueRez, int step)
{
	//declaration for shared memory 
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f; 
	unsigned int bid= blockIdx.x;
	unsigned int laneId = blockIdx.x * blockDim.x + threadIdx.x;
	// Seed starting value as inverse lane ID 
	int value = VectorValue[laneId];
// "value" now contains the sum across all threads 

	for (int j = 1; j < step; j = j * 2)
	{
	__syncthreads();
	tmpsdata[tid] = value;


		if ((tid & j) == 0)
		{
		value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		else
		// if ((tid&j) != 0)
		{
		value = min(abs(value), abs(tmpsdata[tid - j]));
		}
	__syncthreads();
	}

	if(tid == 0)
	{ 
		//save value in global memory
		VectorValueRez[bid] = value;

		if (laneId == 0)
			VectorValueRez[0] = 32;

//		printf("bid: %d, tid: %d, value:%d \n", bid, tid, VectorValueRez[bid]);
	}
	
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_min_bitwise(int* VectorValue, int* VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int bid = blockIdx.x;
	unsigned int laneId = blockIdx.x * blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i < 32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j < step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid & j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid & j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//printf("laneId: %d, tid: %d, value:%d \n", laneId, tid, value);


	if (tid == 0)
	{
		//save value in global memory
		VectorValueRez[bid] = value;

		if (laneId == 0)
			VectorValueRez[0] = 32;

		//printf("bid: %d, tid: %d, value:%d \n", bid, tid, VectorValueRez[bid]);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Min Butterfly (AD)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_min_kernel_shfl_xor_SM_ad(int* VectorValue, int* VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int position_min = ((laneId * step) + step ) - 1;
	unsigned int position_max = (laneId * step) ;

	VectorValue[0]= step;

	//Seed starting value as inverse lane ID
	int value = VectorValue[position_max];

	//printf("laneId: %d, position_max:%d, value:%d \n", laneId, position_max, value);

	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i < 32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j < step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid & j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid & j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//printf("laneId: %d, tid: %d, position_max:%d, value:%d \n", laneId, tid, position_max, value);

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Max Butterfly (AD)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_kernel_shfl_xor_SM_ad(int* VectorValue, int* VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int position_min = ((laneId * step) + step ) - 1;
	unsigned int position_max = (laneId * step);

	//Seed starting value as inverse lane ID
	int value = VectorValue[position_max];

	//printf("laneId: %d, position_max:%d, value:%d \n", laneId, position_max, value);

	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i < 32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j < step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid & j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid & j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//printf("laneId: %d, tid: %d, position_max:%d, value:%d \n", laneId, tid, position_max, value);

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (S-box): Min-Max Butterfly, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_Sbox_v03(int *VectorValue, int *VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;

		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1) + max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}
	//save value in global memory
	VectorValueRez[laneId] = value;

	if ((laneId == 0) & (value > global_max))
	{
		global_max = value;
		//printf("laneId: %d threadIdx.x: %d blockIdx.x: %d Lin(max): %d\n ", laneId, threadIdx.x, blockIdx.x, d_lin_max);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (S-box): Min-Max Butterfly, AC(S) v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_AC_v03(int *VectorValue, int *VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//set first value in global array to zero
	if (laneId == 0)
	{
		VectorValue[laneId] = 0;
	//	printf("laneId: %d threadIdx.x: %d blockIdx.x: %d \n ", laneId, threadIdx.x, blockIdx.x);
	}
	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;

		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1) + max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Min-Max Butterfly, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_v03(T *VectorValue, T *VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValueRez[laneId] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;

		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1) + max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			//value = (value + tmpsdata[tid + j]);
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			//value = (-value + tmpsdata[tid - j]);
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (S-box): Max Butterfly, S-box(es), v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;

		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1) + max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			//value = (value + tmpsdata[tid + j]);
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			//value = (-value + tmpsdata[tid - j]);
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//save value in global memory
	//VectorValue[ji] = value;

	if ((laneId == 0) & (value > global_max))
	{
		global_max = value;
		//printf("laneId: %d threadIdx.x: %d blockIdx.x: %d Lin(max): %d\n ", laneId, threadIdx.x, blockIdx.x, d_lin_max);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (S-box): Min Butterfly, S-box(es), v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void Butterfly_min_kernel_shfl_xor_SM_MP_Sbox_v03(int* VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize) * fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i < fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;

		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j < fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId & j) == 0)
		{
			//value = (value + tmpsdata[tid + j]);
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid & j) != 0)
		{
			//value = (-value + tmpsdata[tid - j]);
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//save value in global memory
	//VectorValue[ji] = value;

	if ((laneId == 0) & (value < global_min))
	{
		global_min = value;
		//printf("laneId: %d threadIdx.x: %d blockIdx.x: %d value: %d\n ", laneId, threadIdx.x, blockIdx.x, value);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Min-Max Butterfly, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP_v03(T * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (laneId - (laneId / fsize)*fsize) * 1024 + (laneId / fsize); //laneId%n
	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	//__syncthreads();
	int value1 = 0;
	// Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<fsize1; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;

		//value = min(abs(value), abs(__shfl_xor(value, i)))*(value1)+max(abs(value), abs(__shfl_xor(value, i)))*(1 - value1);
		value = min(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (value1)+max(abs(value), abs(__shfl_xor_sync(0xffffffff, value, i))) * (1 - value1);

	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<fsize; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((laneId&j) == 0)
		{
			//value = (value + tmpsdata[tid + j]);
			value = max(abs(value), abs(tmpsdata[tid + j]));
		}
		if ((tid&j) != 0)
		{
			//value = (-value + tmpsdata[tid - j]);
			value = min(abs(value), abs(tmpsdata[tid - j]));
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (S-box): Invers Fast Walsh Transforms for S-box => sizeSbox <= BLOCK_SIZE
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ifmt_kernel_shfl_xor_SM_Sbox(int * VectorValue, int * VectorValueRez, int step)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int laneId = blockIdx.x*blockDim.x + threadIdx.x;

	//Seed starting value as inverse lane ID
	int value = VectorValue[laneId];
	__syncthreads();
	int value1 = 0, ZeroOne = 0;
	//Use XOR mode to perform butterfly reduction
	int z = -1;
	for (int i = 1; i<32; i *= 2)
	{
		z = z + 1;
		value1 = (laneId & i);
		value1 >>= z;
		//value = ((value1)*(__shfl_xor(value, i) - value) + (1 - value1)*(__shfl_xor(value, i) + value)) / 2;
		value = ((value1) * (__shfl_xor_sync(0xffffffff, value, i) - value) + (1 - value1) * (__shfl_xor_sync(0xffffffff, value, i) + value)) / 2;
	}
	//@ "value" now contains the sum across all threads

	for (int j = 32; j<step; j = j * 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((tid&j) == 0)
		{
			value = (value + tmpsdata[tid + j]) / 2;
		}
		if ((tid&j) != 0)
		{
			value = (-value + tmpsdata[tid - j]) / 2;
		}
		__syncthreads();
	}

	//(i - (i / n)*n) = > i%n
	//ZeroOne = -((tid - 1) - ((tid - 1) / (tid + 1))*(tid + 1)) + tid; //=>ZeroOne = -((tid - 1) % (tid + 1)) + tid;

	ZeroOne = tid && 1; //=> 0 1 1 1 1 ...

	//save value in global memory
	VectorValueRez[laneId] = value*ZeroOne;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_kernel_shfl_xor_SM(int * VectorValue, int * VectorRez, int sizefor)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	// Seed starting value as inverse lane ID
	int value = VectorValue[i];
	int f1, r = 1;

	for (int j = 1; j<32; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1)+value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor; j *= 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata[tid - j];
		}
		__syncthreads();
	}

	//save value in global memory
	VectorRez[i] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Fast Mobius Transforms, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void fmt_kernel_shfl_xor_SM_v03(int * VectorValue, T * VectorRez, int sizefor)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	/////////////////////////////////////////////////////////////////////////////////
	//convert Integer input and Seed starting value
	unsigned int IdInt32 = i / 32;
	unsigned int IdInt32_Rsh = (i - (i / 32) * 32); //int IdInt32_Rsh = laneId % 32;
	unsigned int value = VectorValue[IdInt32];
	//value = (value >> IdInt32_Rsh) % 2;
	value = (value >> IdInt32_Rsh);
	value = (value - (value / 2) * 2);

	// Seed starting value as inverse lane ID
//	int value = VectorValue[i];
	int f1, r = 1;

	for (int j = 1; j<32; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1) + value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor; j *= 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata[tid - j];
		}
		__syncthreads();
	}

	//save value in global memory
	VectorRez[i] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_kernel_shfl_xor_SM_MP(int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (i - (i / fsize)*fsize) * 1024 + (i / fsize); //laneId%n

	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	int f1, r = 1;

	for (int j = 1; j<fsize1; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1)+value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<fsize; j *= 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata[tid - j];
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Fast Mobius Transforms, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void fmt_kernel_shfl_xor_SM_MP_v03(T * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ int tmpsdata[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (i - (i / fsize)*fsize) * 1024 + (i / fsize); //laneId%n

	// Seed starting value as inverse lane ID
	int value = VectorValue[ji];
	int f1, r = 1;

	for (int j = 1; j<fsize1; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1) + value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<fsize; j *= 2)
	{
		tmpsdata[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata[tid - j];
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (Boolean): Bitwise Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_bitwise_kernel_shfl_xor_SM(unsigned long long int *vect, unsigned long long int *vect_out, int sizefor, int sizefor1)
{
	//declaration for shared memory
	extern __shared__ unsigned long long int tmpsdata1[];

	unsigned int tid = threadIdx.x;
	unsigned int ij = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	unsigned long long int value = vect[ij];

	value ^= (value & 12297829382473034410) >> 1;
	value ^= (value & 14757395258967641292) >> 2;
	value ^= (value & 17361641481138401520) >> 4;
	value ^= (value & 18374966859414961920) >> 8;
	value ^= (value & 18446462603027742720) >> 16;
	value ^= (value & 18446744069414584320) >> 32;

	int f1, r = 1;

	for (int j = 1; j<sizefor; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1)+value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor1; j *= 2)
	{
		tmpsdata1[tid] = value;
		__syncthreads();

		if ((tid&j) == j)
		{
			value = value^tmpsdata1[tid - j];

		}
	}

	//save in global memory
	vect_out[ij] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//First function (S-box): Bitwise Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_bitwise_kernel_shfl_xor_SM_Sbox(unsigned long long int *vect, unsigned long long int *vect_out, int sizefor, int sizefor1)
{
	//declaration for shared memory
	extern __shared__ unsigned long long int tmpsdata1[];

	unsigned int tid = threadIdx.x;
	unsigned int ij = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned long long int value = vect[ij];

	value ^= (value & 12297829382473034410) >> 1;
	value ^= (value & 14757395258967641292) >> 2;
	value ^= (value & 17361641481138401520) >> 4;
	value ^= (value & 18374966859414961920) >> 8;
	value ^= (value & 18446462603027742720) >> 16;
	value ^= (value & 18446744069414584320) >> 32;

	int f1, r = 1;

	for (int j = 1; j<sizefor; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1)+value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<sizefor1; j *= 2)
	{
		tmpsdata1[tid] = value;
		__syncthreads();

		if ((tid&j) == j)
		{
			value = value^tmpsdata1[tid - j];

		}
	}

	//save in global memory
	vect_out[ij] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Second function (Boolean): Bitwise Fast Mobius Transforms
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void fmt_bitwise_kernel_shfl_xor_SM_MP(unsigned long long int * VectorValue, int fsize, int fsize1)
{
	//declaration for shared memory
	extern __shared__ unsigned long long int tmpsdata1[];

	unsigned int tid = threadIdx.x;// & 0x1f;
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int ji = (i - (i / fsize)*fsize) * 1024 + (i / fsize); //laneId%n

	// Seed starting value as inverse lane ID
	unsigned long long int value = VectorValue[ji];
	int f1, r = 1;

	for (int j = 1; j<fsize1; j *= 2)
	{
		f1 = (tid >> (r - 1) & 1);
		//value = (value ^ __shfl_up(value, j))*(f1)+value*(1 - f1);
		value = (value ^ __shfl_up_sync(0xffffffff, value, j)) * (f1)+value * (1 - f1);
		r++;
		//printf("j %d, r %d, f1 %d, val: %d\n", j, r, f1, value);
	}

	for (int j = 32; j<fsize; j *= 2)
	{
		tmpsdata1[tid] = value;
		__syncthreads();

		if ((i&j) == j)
		{
			value = value^tmpsdata1[tid - j];
		}
		__syncthreads();
	}

	//save value in global memory
	VectorValue[ji] = value;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Help Function (additional): Power integer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void powInt(int *Vec, int exp)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	int base = Vec[i];

	__syncthreads();

	int result = 1;

	while (exp)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		base *= base;
	}

	Vec[i] = result;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Help Function (additional): Power integer, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void powInt_v03(T *Vec, int exp)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	int base = Vec[i];

	__syncthreads();
	int result = 1;

	while (exp)
	{
		if (exp & 1)
			result *= base;
		exp >>= 1;
		base *= base;
	}

	Vec[i] = result;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (Boolean): Algebraic degree
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AD(int *Vec)
{
	//@@ Local variable
	int ones = 0;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int value = Vec[i];

	ones = __popc(i)*value; //Count the number of bits that are set to 1 in a 32 bit integer.

	Vec[i] = ones;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (Boolean): Algebraic degree, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
__global__ void kernel_AD_v03(T *Vec)
{
	//@@ Local variable
	int ones = 0;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int value = Vec[i];

	ones = __popc(i)*value; //Count the number of bits that are set to 1 in a 32 bit integer.

	Vec[i] = ones;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (Boolean): Bitwise Algebraic degree
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bitwise_AD(unsigned long long int *NumIntVec, int *Vec_max_values, int NumOfBits)
{
	//@@ Local variable
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	unsigned int ii = i*NumOfBits;

	unsigned int ones = 0, max = 0;
	int c;
	bool bit;

	unsigned long long int k = 0;
	unsigned long long int number = NumIntVec[i]; //copy data in local variable

	for (c = NumOfBits - 1; c >= 0; c--)
	{
		k = number >> c;

		if (k & 1)
		{
			bit = 1;
			ii++;
		}
		else
		{
			bit = 0;
			ii++;
		}

		ones = __popc(ii - 1)*bit; //Count the number of bits that are set to 1 in a 32 bit integer.

		if (max<ones)
			max = ones;

	}

	//save in global memory
	Vec_max_values[i] = max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Algebraic degree S-box
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_AD_Sbox(int *Vec)
{
	//@@ Local variable
	int ones = 0;

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;

	unsigned int value = Vec[i]; //copy data in local variable

	ones = __popc(tid)*value; //Count the number of bits that are set to 1 in a 32 bit integer.

	Vec[i] = ones;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Bitwise Algebraic degree S-box
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bitwise_AD_Sbox(unsigned long long int *NumIntVec, int *Vec_max_values, int NumOfBits)
{
	//@@ Local variable
	int ones = 0, max=0;

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;

	unsigned int ii = tid*NumOfBits;

	unsigned long long int number = NumIntVec[i], k=0; //copy data in local variable

	int c;
	bool bit;

	for (c = NumOfBits - 1; c >= 0; c--)
	{
		k = number >> c;

		if (k & 1)
		{
			bit = 1;
		}
		else
		{
			bit = 0;
		}

		ones = __popc(ii)*bit; //Count the number of bits that are set to 1 in a 32 bit integer.

		ii++;

		if (max<ones)
			max = ones;
	}

	Vec_max_values[i] = max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Component Function of S-box (CF) - first function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnAll_kernel(int *Sbox_in, int *CF_out, int n)
{
	//@@ Local variable
	int logI, ones, element = 0;

	unsigned int blok = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*n + threadIdx.x;

	int value = Sbox_in[tid]; //copy data in local variable

	logI = value&blok;
	ones = __popc(logI); //Count the number of bits that are set to 1 in a 32 bit integer.
	element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Component Function - Polarity True Table of S-box (CF - PTT) - first function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnAll_kernel_PTT(int *Sbox_in, int *CF_out, int n)
{
	//@@ Local variable
	int logI, ones, element = 0;

	unsigned int blok = blockIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*n + threadIdx.x;

	int value = Sbox_in[tid]; //copy data in local variable

	logI = value&blok;
	ones = __popc(logI); //Count the number of bits that are set to 1 in a 32 bit integer.

	//element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)
	//element = 1 - (2 * element);
	element = 1 - (2 * (ones&(1))); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Component Function of S-box (CF) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnVec_kernel(int *Sbox_in, int *CF_out, int row)
{
	//@@ Local variable
	int logI, ones, element = 0;

	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int value = Sbox_in[i]; //copy data in local variable

	logI = value&row;	//logI=Vect[tid]&blok;
	ones = __popc(logI); //Count the number of bits that are set to 1 in a 32 bit integer.

	element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Component Function of S-box (CF) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnAllVecIn_kernel_v03(int *Sbox_in, int *CF_out, int row, int sizeSbox)
{
	//@@ Local variable
	int logI, ones, element = 0;

	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	unsigned int index = i+(row*sizeSbox);

	int value = Sbox_in[i]; //copy data in local variable

	logI = value&row; 	 //logI=Vect[tid]&blok;
	ones = __popc(logI); //Count the number of bits that are set to 1 in a 32 bit integer.

	element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)

	CF_out[index] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Component Function - Polarity True Table of S-box (CF-PTT) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComponentFnVec_kernel_PTT(int *Sbox_in, int *CF_out, int row)
{
	//@@ Local variable
	int logI, ones, element = 0;

	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int value = Sbox_in[i]; //copy data in local variable

	logI = value&row; //logI=Vect[tid]&blok;
	ones = __popc(logI); //Count the number of bits that are set to 1 in a 32 bit integer.

	//element = (ones&(1)); //i&(n-1): (i%n) =>(ones%2)
	//element = 1 - (2 * element);
	element = 1 - (2 * (ones&(1))); //i&(n-1): (i%n) =>(ones%2)

	CF_out[i] = element;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Binary to Decimal kernel, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void BinVecToDec_kernel_v03(unsigned long long int *device_NumIntVecCF, int *device_CF, int NumOfBits)
{
	//@@ Local variable
	int set, counterBin = 0, j;
	unsigned long long int sum = 0, decimal = 0, bin=0;

	//@indexes
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	set = i*NumOfBits;

	for (j = ((NumOfBits - 1) + set); j >= (0 + set); j--)
	{
		bin = device_CF[j]; //copy data in local variable

		decimal = bin << counterBin;
		counterBin++;
		sum = sum + decimal;
	}

	device_NumIntVecCF[i] = sum;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Binary to Decimal kernel - v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void BinVecToDec_kernel_move_v03(unsigned long long int *device_NumIntVecCF, int *device_CF, int NumOfBits, int move)
{
	//@@ Local variable
	int set, counterBin = 0, j;
	unsigned long long int sum = 0, decimal = 0, bin = 0;

	//@ indexes
	unsigned int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	unsigned int g_i = i + move;

	set = i*NumOfBits;

	for (j = ((NumOfBits - 1) + set); j >= (0 + set); j--)
	{
		bin = device_CF[j];
		//	printf("j:%j, bin:%d ", j , bin);
		decimal = bin << counterBin;
		counterBin++;
		sum = sum + decimal;
	}

	device_NumIntVecCF[g_i] = sum;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Difference Distribution Table (DDT) - first function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DDTFnAll_kernel(int *Sbox_in, int *DDT_out, int n)
{
	//declaration for shared memory
	extern __shared__ int tmpSbox[];

	//@@ Local variable
	unsigned int x2, dy;
	unsigned int x1 = threadIdx.x;// & 0x1f;
	unsigned int dx = blockIdx.x;

	tmpSbox[x1] = Sbox_in[x1];

	__syncthreads();

	x2 = x1 ^ dx;
	dy = (tmpSbox[x1] ^ tmpSbox[x2]) + blockIdx.x*n; // dy = (sbox[x1] ^ sbox[x2])+ blockIdx.x*size;

	atomicAdd(&DDT_out[dy], 1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Difference Distribution Table (DDT) - second function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DDTFnAll_kernel_expand(int* Sbox_in, int* DDT_out, int size)
{
	//@@ Local variable
	int x2, dx, dy;// , value;

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x1 = (blockIdx.x * blockDim.x + threadIdx.x) % size;

	dx = id / size;
	x2 = x1 ^ dx; //row - dx
	dy = (Sbox_in[x1] ^ Sbox_in[x2]) + dx * size;

	atomicAdd(&DDT_out[dy], 1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Function (S-box): Difference Distribution Table (DDT) - third function
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void DDTFnVec_kernel(int *Sbox_in, int *DDT_out, int row)
{
	//@@ Local variable
	int x2, dy;// , value;
	unsigned int x1 = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	x2 = x1^row; //row - dx
	dy = Sbox_in[x1] ^ Sbox_in[x2];

	atomicAdd(&DDT_out[dy], 1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

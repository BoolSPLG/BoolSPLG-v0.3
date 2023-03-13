//BoolSPLG Boolean Algorithms

//System includes Librarys
#include <stdio.h>
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (S-box) Max Butterfly function
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Butterfly_max_kernel(int sizeSbox, int *device_data)
{
	CheckSize(sizeSbox);

	int max; //max variable

	//@ Set grid
	setgrid(sizeSbox);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_data, device_data, sizethread);
	if (sizeSbox>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_data, sizeblok, sizeblok1);

	cudaMemcpy(&max, &device_data[0], sizeof(int), cudaMemcpyDeviceToHost);

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (S-box) Min Butterfly function
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Butterfly_min_kernel_ad(int sizeSbox, int* device_data)
{
	CheckSize(sizeSbox);

	int min; //max variable

	//@ Set grid
	//setgrid(sizeSbox);
	sizethread = sizeSbox;
	sizeblok = sizeSbox;

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_data, device_data, sizeSbox);
	//if (sizeSbox > 1024)
		//Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_data, sizeblok, sizeblok1);

	Butterfly_min_kernel_shfl_xor_SM_ad << <1, sizethread, sizethread * sizeof(int) >> > (device_data, device_data, sizeSbox);

	cudaMemcpy(&min, &device_data[sizeSbox-1], sizeof(int), cudaMemcpyDeviceToHost);

	return min;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (S-box) Max Butterfly function for DDT v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Butterfly_max_kernel_DDT_v03(int sizeSbox, int *device_data)
{
	CheckSize(sizeSbox);

	//@ Set grid
	setgrid(sizeSbox);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_data, device_data, sizethread);
	if (sizeSbox>1024)
	Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_data, sizeblok, sizeblok1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Fast walsh transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//call Reduction Max
		max = runReductionMax(size, device_Vect_rez);
	}
	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Fast walsh transform use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, sizeblok, sizeblok1);

		//memcpy max variable from the global memory
		cudaMemcpy(&max, &device_Vect[0], sizeof(int), cudaMemcpyDeviceToHost);
	}
	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (S-box) Fast Walsh Transform use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void WalshSpecTranS_boxGPU_ButterflyMax_v03(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, sizeblok, sizeblok1);
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Fast walsh transform use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
int WalshSpecTranBoolGPU_ButterflyMax_v03(int *device_Vect, T *device_Vect_rez, T *device_Vect_Max, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_Max, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_Max, sizeblok, sizeblok1);

		//memcpy max variable from the global memory
		cudaMemcpy(&max, &device_Vect_Max[0], sizeof(T), cudaMemcpyDeviceToHost);
	}
	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Fast Mobius Transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MobiusTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);
	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Fast Mobius transform, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
void MobiusTranBoolGPU_v03(int *device_Vect, T *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);
	//@ Set grid
	setgrid(size);

	/////////////////////////////////////////////////////
	fmt_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fmt_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Bitwise Fast Mobius transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BitwiseMobiusTranBoolGPU(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSizeBoolBitwise(size);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = size / NumOfBits;

	//@ Set grid Bitwise
	setgridBitwise(NumInt);

	///////////////////////////////////////////////////
	fmt_bitwise_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect, device_Vect_rez, sizefor, sizefor1);
	if (NumInt>1024)
		fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect_rez, sizeblok, sizeblok1);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeBoolGPU(int *device_Vect, int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	kernel_AD << <sizeblok, sizethread >> >(device_Vect_rez);

	//return Max reduction deg(f) of Boolean function
	//max = SetRunMaxReduction(size, device_Vect_rez);
	max = runReductionMax(size, device_Vect_rez);

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	kernel_AD << <sizeblok, sizethread >> >(device_Vect_rez);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//memcpy max variable from the global memory
	cudaMemcpy(&max, &device_Vect_rez[0], sizeof(int), cudaMemcpyDeviceToHost);

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree use Max Butterfly v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AlgebraicDegreeBoolGPU_ButterflyMax_v03(int *device_Vect, int *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	kernel_AD << <sizeblok, sizethread >> >(device_Vect_rez);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree use Min Butterfly v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AlgebraicDegreeBoolGPU_ButterflyMin_v03(int* device_Vect, int* device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	kernel_AD << <sizeblok, sizethread >> > (device_Vect_rez);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect_rez, device_Vect_rez, sizethread);
	if (size > 1024)
		Butterfly_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect_rez, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree use Min-Max Butterfly v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
int AlgebraicDegreeBoolGPU_ButterflyMax_v03_T(int *device_Vect, T *device_Vect_rez, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//Call BoolSPLG Library FMT(f) function
	//////////////////////////////////////////////////////////////////////////////////////////////////
	MobiusTranBoolGPU_v03<T>(device_Vect, device_Vect_rez, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//Algebriac degree find GPU algorithm 1
	kernel_AD_v03<T> << <sizeblok, sizethread >> >(device_Vect_rez);

	int max = 0; //max variable
	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//memcpy max variable from the global memory
	cudaMemcpy(&max, &device_Vect_rez[0], sizeof(T), cudaMemcpyDeviceToHost);

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bitwise (Boolean) Algebriac degree use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int *device_Vec_max_values, int *host_max_values, int size)
{
	//Check the input size of Boolean function
	CheckSizeBoolBitwise(size);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = size / NumOfBits;
	int max = 0; //max variable

	//@ Set grid Bitwise
	setgridBitwise(NumInt);

	///////////////////////////////////////////////////
	fmt_bitwise_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect, device_Vect_rez, sizefor, sizefor1);
	if (NumInt>1024)
		fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect_rez, sizeblok, sizeblok1);
	///////////////////////////////////////////////////

	kernel_bitwise_AD << < sizeblok, sizethread >> >(device_Vect_rez, device_Vec_max_values, NumOfBits);

	if (NumInt > 256)
	{
	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM <<<sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, device_Vec_max_values, sizethread);
	if (NumInt>1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP <<< sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, sizeblok, sizeblok1);

	//memcpy max variable from the global memory
	cudaMemcpy(&max, &device_Vec_max_values[0], sizeof(int), cudaMemcpyDeviceToHost);
	}

	else
	{
		//cudaMemcpy Device memory array To Host memory array
		cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int)* NumInt, cudaMemcpyDeviceToHost);

		max=reduceCPU_max_libhelp(host_max_values, NumInt);
	}
	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Bitwise (Boolean) Algebriac degree use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BitwiseAlgebraicDegreeBoolGPU_ButterflyMax_v03(unsigned long long int *device_Vect, unsigned long long int *device_Vect_rez, int *device_Vec_max_values, int *host_max_values, int size)
{
	//Check the input size of Boolean function
	CheckSizeBoolBitwise(size);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = size / NumOfBits;

	//@ Set grid Bitwise
	setgridBitwise(NumInt);

	///////////////////////////////////////////////////
	fmt_bitwise_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect, device_Vect_rez, sizefor, sizefor1);
	if (NumInt>1024)
		fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_Vect_rez, sizeblok, sizeblok1);
	///////////////////////////////////////////////////

	kernel_bitwise_AD << < sizeblok, sizethread >> >(device_Vect_rez, device_Vec_max_values, NumOfBits);

	if (NumInt > 1024)
	{
		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, device_Vec_max_values, sizethread);
		if (NumInt>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP_v03 << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, sizeblok, sizeblok1);
	}

	else
	{
		cout << "\nError:" << "\n\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree use Max Butterfly (ANF 'in') v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AlgebraicDegreeBoolGPU_ANF_in_ButterflyMax_v03(int* device_Vect, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	//MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	//kernel_AD << <sizeblok, sizethread >> > (device_Vect_rez);
	kernel_AD << <sizeblok, sizethread >> > (device_Vect);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect, device_Vect, sizethread);
	if (size > 1024)
		Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Algebriac degree use Min Butterfly (ANF 'in') v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AlgebraicDegreeBoolGPU_ANF_in_ButterflyMin_v03(int* device_Vect, int size)
{
	//Check the input size of Boolean function
	CheckSize(size);

	//MobiusTranBoolGPU(device_Vect, device_Vect_rez, size);

	//Algebriac degree find GPU algorithm 1
	//kernel_AD << <sizeblok, sizethread >> > (device_Vect_rez);
	kernel_AD << <sizeblok, sizethread >> > (device_Vect);

	//call Butterfly max min kernel
	/////////////////////////////////////////////////////
	Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect, device_Vect, sizethread);
	if (size > 1024)
		//Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect, sizeblok, sizeblok1);
		Butterfly_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vect, sizeblok, sizeblok1);

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Autocorrelation Transform
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranBoolGPU(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max =0; //max variable

	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	powInt << < sizeblok, sizethread >> >(device_Vect_rez, 2);

	ifmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		ifmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//cudaMemset device memory
		cudaMemset(device_Vect_rez, 0, 1 * sizeof(int));
		//	max = SetRunMaxReduction(size, device_Vect_rez);

		//return Max reduction return AC(f) of the Boolean function
		max = runReductionMax(size, device_Vect_rez);

	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Autocorrelation Transform use Min-Max Butterfly
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranBoolGPU_ButterflyMax(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	powInt << < sizeblok, sizethread >> >(device_Vect_rez, 2);

	ifmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		ifmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//cudaMemset device memory
		cudaMemset(device_Vect_rez, 0, 1 * sizeof(int));

		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, sizeblok, sizeblok1);

		//memcpy max variable from the global memory
		cudaMemcpy(&max, &device_Vect[0], sizeof(int), cudaMemcpyDeviceToHost);
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (S-box) Autocorrelation Transform use Min-Max Butterfly S-box, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AutocorrelationTranS_boxGPU_ButterflyMax_03(int *device_Vect, int *device_Vect_rez, int size, bool returnMaxReduction)
{
	//Check the input size
	CheckSize(size);

	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	powInt << < sizeblok, sizethread >> >(device_Vect_rez, 2);

	ifmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		ifmt_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//cudaMemset(device_Vect_rez, 0, sizeof(int));

		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM_AC_v03 << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, sizeblok, sizeblok1);

	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute (Boolean) Autocorrelation Transform use Min-Max Butterfly, v0.3
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T>
int AutocorrelationTranBoolGPU_ButterflyMax_v03(int *device_Vect, T *device_Vect_rez, T *device_Vect_Max, int size, bool returnMaxReduction)
{
	//Check the input size of Boolean function
	CheckSize(size);

	int max = 0; //max variable

	//@ Set grid
	setgrid(size);

	///////////////////////////////////////////////////
	fwt_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect, device_Vect_rez, sizethread);
	if (size>1024)
		fwt_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	powInt_v03<T><< < sizeblok, sizethread >> >(device_Vect_rez, 2);

	ifmt_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_rez, sizethread);
	if (size>1024)
		ifmt_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, sizeblok, sizeblok1);

	//return Max reduction
	if (returnMaxReduction)
	{
		//cudaMemset device memory
		cudaMemset(device_Vect_rez, 0, 1 * sizeof(T));

		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM_v03<T> << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_rez, device_Vect_Max, sizethread);
		if (size>1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP_v03<T> << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vect_Max, sizeblok, sizeblok1);

		//Memcopy max variable from device memory
		cudaMemcpy(&max, &device_Vect_Max[0], sizeof(T), cudaMemcpyDeviceToHost);
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Linear Aproximation Table (LAT) - Linearity of S-box
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranSboxGPU(int *device_Sbox, int *device_CF, int *device_WST, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute LAT of S-box
		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_WST, sizethread);

		//cudaMemset device memory
		cudaMemset(device_WST, 0, sizeSbox*sizeof(int)); //clear first row of LAT !!!

		//@Reduction Max return Lin of the S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_WST);
	}

	else
	{
		//@Declaration and Alocation of memory blocks
		int *ALL_WST = (int *)malloc(sizeof(int)* sizeSbox);
		ALL_WST[0] = 0;

		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			max = WalshSpecTranBoolGPU(device_CF, device_WST, sizeSbox, true);
			//max = runReductionMax(sizeSbox, device_LAT);
			ALL_WST[i] = max;
		}

		//cudaMemset(device_LAT, 0, sizeSbox*sizeof(int)); //clear LAT !!!
		// Copy input vectors from host memory to GPU buffers.
		cudaMemcpy(device_WST, ALL_WST, sizeof(int)*sizeSbox, cudaMemcpyHostToDevice);

		//@Reduction Max return Lin of the S-box
		max = runReductionMax(sizeSbox, device_WST);

		//@Free memory
		free(ALL_WST);
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Linear Aproximation Table (LAT) use Max Butterfly - Linearity of S-box
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_WST, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute LAT of S-box
		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_WST, sizethread);

		if (returnMax)
		{
		int max = 0; //max variable

		//cudaMemset device memory
		cudaMemset(device_WST, 0, sizeSbox*sizeof(int)); //clear first row of LAT !!!

		//use Max Butterfly return Lin of the S-box
		max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_WST);

		return max;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (returnMax)
		{
			int max = 0, MaxReturn = 0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				//@Compute Component function GPU
				ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);
				if (returnMax)
				{
					MaxReturn = WalshSpecTranBoolGPU_ButterflyMax(device_CF, device_WST, sizeSbox, true);

					if (MaxReturn>max)
						max = MaxReturn;
				}
			}
			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Walsh Spectra W(S) of S-box use Max Butterfly - Linearity of S-box, v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int WalshSpecTranSboxGPU_ButterflyMax_v03(int *device_Sbox, int *device_CF, int *device_WST, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute LAT of S-box
		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_WST, sizethread);

		if (returnMax)
		{
			int max = 0; //max variable

			//cudaMemset for device memory
			cudaMemset(device_WST, 0, sizeSbox*sizeof(int)); //clear first row of LAT !!!

			//use Max Butterfly return Lin of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_WST);

			return max;
		}
		else
		{
			return 0;
		}
	}

	else
	{
		if (returnMax)
		{
			int max = 0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				//@Compute Component function GPU
				ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

				if (returnMax)
				{
					//find max Lin for S-box component function
					WalshSpecTranS_boxGPU_ButterflyMax_v03(device_CF, device_WST, sizeSbox, true);

				}
			}
			//@return Lin(S) of the S-box
			cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

			int set_zero = 0;
			cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"


			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Linear Aproximation Table (LAT) use Max Butterfly - Linearity of S-box, v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void LATSboxGPU_v03(int* device_Sbox, int* device_CF, int* device_LAT, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> > (device_Sbox, device_CF, sizeSbox);

		//@Compute LAT of S-box
		fwt_kernel_shfl_xor_SM_LAT << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_CF, device_LAT, sizethread);
	}

	else
	{
		cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute Algebraic Normal Form (ANF) - Algebraic Degree for S-box
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int MobiusTranSboxADGPU(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ANF, sizethread);

		kernel_AD_Sbox << <sizeblok, sizethread >> >(device_ANF);

		//@Reduction Max return Lin of the S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_ANF);

		return max;
	}
	else
	{
		//@Declaration and Alocation of memory blocks
		int *ALL_ANF = (int *)malloc(sizeof(int)* sizeSbox);
		ALL_ANF[0] = 0;

		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			max = AlgebraicDegreeBoolGPU(device_CF, device_ANF, sizeSbox);
			ALL_ANF[i] = max;
		}

		//cudaMemset(device_LAT, 0, sizeSbox*sizeof(int)); //clear LAT !!!
		// Copy input vectors from host memory to GPU buffers.
		cudaMemcpy(device_ANF, ALL_ANF, sizeof(int)*sizeSbox, cudaMemcpyHostToDevice);

		//@Reduction Max return ANF of the S-box
		max = runReductionMax(sizeSbox, device_ANF);

		//@Free memory
		free(ALL_ANF);
		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Normal Form (ANF)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MobiusTranSboxGPU(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ANF, sizethread);
	}
	else
	{
		cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Normal Form (ANF) Bitwise
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void BitwiseMobiusTranSboxGPU(int *host_Sbox, int *host_Vect_CF, unsigned long long int *host_NumIntVecCF, unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSboxBitwiseMobius(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = 0;

	if (sizeSbox<16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		int sizefor, sizefor1;
		NumInt = (sizeSbox*sizeSbox) / NumOfBits;
		//Compute Component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			// CPU computing component function (CF) of S-box function - all CF are save in one array
			GenTTComponentFunc(i, host_Sbox, host_Vect_CF, sizeSbox);
		}

		//convert bool into integers
		BinVecToDec(NumOfBits, host_Vect_CF, host_NumIntVecCF, NumInt);

		//Memcopy Host to Device
		cudaMemcpy(device_NumIntVecCF, host_NumIntVecCF, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

		//Set Bitwise S-box GRID
		if (sizeSbox < 2048)
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = sizeSbox / 64;
			sizefor1 = 32;
		}
		else
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = 32;
			sizefor1 = NumInt / NumOfBits;
		}

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

	}
	////////////////////////////////////////////////////////
	else
	{
		cout << "\nIs not implemented this funcionality for S-box size n>14. \nThe output data is to big.\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Degree use Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_ANF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ANF, sizethread);

		//Compute S-box AD
		kernel_AD_Sbox << <sizeblok, sizethread >> >(device_ANF);

		//use Max Butterfly return ANF of the S-box
		max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_ANF);

		return max;
	}
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			//returnMax = AlgebraicDegreeBoolGPU_ButterflyMax(device_CF, device_ANF, sizeSbox);
			AlgebraicDegreeBoolGPU_ButterflyMax_v03(device_CF, device_ANF, sizeSbox);
		}

		//@return Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

		int set_zero = 0;
		cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"

		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Degree use Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeSboxGPU_ButterflyMin(int* device_Sbox, int* device_CF, int* device_ANF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int min = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_CF, device_ANF, sizethread);

		//Compute S-box AD
		kernel_AD_Sbox << <sizeblok, sizethread >> > (device_ANF);

		//use Max Butterfly return ANF of the S-box
		min = Butterfly_min_kernel_ad(sizeSbox, device_ANF);

		return min;
	}
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;
		sizeblok1 = sizeblok;

		if (sizeblok > 32)
			sizeblok1 = 32;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			//returnMax = AlgebraicDegreeBoolGPU_ButterflyMax(device_CF, device_ANF, sizeSbox);
			AlgebraicDegreeBoolGPU_ButterflyMin_v03(device_CF, device_ANF, sizeSbox);
		}

		//@return Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&min, global_min, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_min"

		int set_value = 32;
		cudaMemcpyToSymbol(global_min, &set_value, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_min=32"

		return min;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Degree when input is ANF with use of Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeSboxGPU_in_ANF_ButterflyMax(int* device_Sbox_ANF, int* device_CF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, sizeSbox);

		//@Compute ANF of S-box
		//fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_CF, device_ANF, sizethread);

		//Compute S-box AD 
		//kernel_AD_Sbox << <sizeblok, sizethread >> > (device_ANF);
		kernel_AD_Sbox << <sizeblok, sizethread >> > (device_CF);

		//use Max Butterfly return ANF of the S-box
		//max = Butterfly_max_kernel(sizeSbox * sizeSbox, device_ANF);
		max = Butterfly_max_kernel(sizeSbox * sizeSbox, device_CF);

		return max;
	}
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			//returnMax = AlgebraicDegreeBoolGPU_ButterflyMax(device_CF, device_ANF, sizeSbox);
			AlgebraicDegreeBoolGPU_ANF_in_ButterflyMax_v03(device_CF, sizeSbox);
		}

		//@return Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

		int set_zero = 0;
		cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"

		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Degree when input is ANF with use of Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AlgebraicDegreeSboxGPU_in_ANF_ButterflyMin(int* device_Sbox_ANF, int* device_CF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int min = 0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, sizeSbox);

		//@Compute ANF of S-box
		//fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_CF, device_ANF, sizethread);

		//Compute S-box AD 
		//kernel_AD_Sbox << <sizeblok, sizethread >> > (device_ANF);
		kernel_AD_Sbox << <sizeblok, sizethread >> > (device_CF);

		//use Max Butterfly return ANF of the S-box
		min = Butterfly_min_kernel_ad(sizeSbox, device_CF);

		return min;
	}
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		sizeblok1 = sizeblok;

		if (sizeblok > 32)
			sizeblok1 = 32;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i);

			//@Function return deg(S) of component function of the S-box
			//returnMax = AlgebraicDegreeBoolGPU_ButterflyMax(device_CF, device_ANF, sizeSbox);
			AlgebraicDegreeBoolGPU_ANF_in_ButterflyMin_v03(device_CF, sizeSbox);
		}

		//@return Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&min, global_min, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_min"

		int set_value = 32;
		cudaMemcpyToSymbol(global_min, &set_value, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_min=0"

		return min;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Algebraic Degree Table 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AlgebraicDegreeTableSboxGPU(int* device_Sbox, int* device_CF, int* device_ADT, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, sizeSbox);

		//@Compute ANF of S-box
		fmt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_CF, device_ADT, sizethread);

		//Compute S-box AD
		kernel_AD_Sbox << <sizeblok, sizethread >> > (device_ADT);

	}
	else
	{
		cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Bitwise Algebraic Degree use Max Butterfly
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMax(int *host_Sbox, int *host_Vect_CF, int *host_max_values, unsigned long long int *host_NumIntVecCF, unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int *device_Vec_max_values, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSboxBitwise(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8;
	int NumInt = 0, max = 0, returnMax = 0; //max variable

	if (sizeSbox<16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		int sizefor, sizefor1;
		NumInt = (sizeSbox*sizeSbox) / NumOfBits;
		//Compute Component function CPU
		for (int i = 0; i < sizeSbox; i++)
		{
			// CPU computing component function (CF) of S-box function - all CF are save in one array
			GenTTComponentFunc(i, host_Sbox, host_Vect_CF, sizeSbox);
		}

		//convert bool into integers
		BinVecToDec(NumOfBits, host_Vect_CF, host_NumIntVecCF, NumInt);

		//Memcpy Host to Device
		cudaMemcpy(device_NumIntVecCF, host_NumIntVecCF, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

		//Set Bitwise S-box GRID
		if (sizeSbox < 2048)
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = sizeSbox / 64;
			sizefor1 = 32;
		}
		else
		{
			sizeblok = sizeSbox;
			sizethread = sizeSbox / NumOfBits;

			sizefor = 32;
			sizefor1 = NumInt / NumOfBits;
		}

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> >(device_NumIntVecANF, device_Vec_max_values, NumOfBits);

		if (NumInt > 256)
		{
			//@ Set grid
			setgrid(NumInt);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, device_Vec_max_values, sizethread);
			if (NumInt>1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, sizeblok, sizeblok1);

			//Memcpy Host to Device
			cudaMemcpy(&max, &device_Vec_max_values[0], sizeof(int), cudaMemcpyDeviceToHost);

			return max;
		}
		else
		{
		cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int)* NumInt, cudaMemcpyDeviceToHost);

		max = reduceCPU_max_libhelp(host_max_values, NumInt);

			return max;
		}
	}
	////////////////////////////////////////////////////////
	else
	{
		NumInt = sizeSbox / NumOfBits;
		for (int i = 0; i < sizeSbox; i++)
		{
			//===== CPU computing component function (CF) of S-box function === "helpSboxfunct.h"
			//===== One CF is save in array CPU_STT ===========================
			GenTTComponentFuncVec(i, host_Sbox, host_Vect_CF, sizeSbox);

			//convert bool into integers
			BinVecToDec(NumOfBits, host_Vect_CF, host_NumIntVecCF, NumInt);

			cudaMemcpy(device_NumIntVecCF, host_NumIntVecCF, sizeof(unsigned long long int)* NumInt, cudaMemcpyHostToDevice);

			returnMax = BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(device_NumIntVecCF, device_NumIntVecANF, device_Vec_max_values, host_max_values, sizeSbox);

			if (returnMax > max)
				max = returnMax;
		}
		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Bitwise Algebraic Degree use Max Butterfly v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMax_v03(unsigned long long int *device_NumIntVecCF, unsigned long long int *device_NumIntVecANF, int *device_Vec_max_values, int *device_Sbox, int *device_CF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSboxBitwise(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8; //set number of bits
	int NumInt = 0, max = 0; //max variable

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox<16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		NumInt = (sizeSbox*sizeSbox) / NumOfBits; //compute number of Integers

		//@ For S-box size <=1024
		if (sizeSbox < 2048)
		{
			//@set GRID
			sizethread = sizeSbox;
			sizeblok = sizeSbox;

			//@Compute Component function GPU - BoolSPLG Library function
			ComponentFnAll_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);
		}
		//@ For S-box size > 1024
		else
		{
			//@set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			//cout << "Component function (device):" << "\n";
			for (int i = 0; i < sizeSbox; i++)
			{
				//@Compute Component function GPU - call ComponentFnAllVecIn_kernel_v03 and full device_CF vector
				ComponentFnAllVecIn_kernel_v03 <<<sizeblok, sizethread >> >(device_Sbox, device_CF, i, sizeSbox);
			}
		}

		//@set GRID for Binary to Decimal kernel
		setgrid_BinVecToDec(sizeSbox, NumInt);

		//@Kernel Binary to Decimal conversion
		BinVecToDec_kernel_v03 << < sizeblok, sizethread >> >(device_NumIntVecCF, device_CF, NumOfBits);

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt, NumOfBits);

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox <<<sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		kernel_bitwise_AD_Sbox <<< sizeblok, sizethread >> >(device_NumIntVecANF, device_Vec_max_values, NumOfBits);

		if (NumInt > 256)
		{
			//@ Set grid for Butterfly max kernel
			setgrid(NumInt);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM <<<sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, device_Vec_max_values, sizethread);
			if (NumInt>1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Vec_max_values, sizeblok, sizeblok1);

			//memcpy device to host
			cudaMemcpy(&max, &device_Vec_max_values[0], sizeof(int), cudaMemcpyDeviceToHost);

		}
		else
		{
			int *host_max_values = (int*)malloc(sizeof(int) * NumInt);

			//memcpy device to host
			cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int)* NumInt, cudaMemcpyDeviceToHost);

			max = reduceCPU_max_libhelp(host_max_values, NumInt);

			free(host_max_values);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192)&(sizeSbox < 131072))
	{
		int move = 0; //Declaration of variable used for setting writing output results of S-box component function
		NumInt = sizeSbox / NumOfBits; //variable - number of Integer per component function
		int sizethread_1, sizeblok_1;	//additional vatiable used for kernel paramethers

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread_1 = NumInt;
		sizeblok_1 = 1;


		for (int i = 0; i < sizeSbox; i++)
		{
			//variable used for setting writing output results
			move = i*NumInt;

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_move_v03 << < sizeblok_1, sizethread_1 >> >(device_NumIntVecCF, device_CF, NumOfBits, move);
		}

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt*sizeSbox, NumOfBits);

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> >(device_NumIntVecANF, device_Vec_max_values, NumOfBits);

		//Call Max Reduction function
		max = runReductionMax(NumInt*sizeSbox, device_Vec_max_values);

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if(sizeSbox >= 131072)
	{
		int sizethread1_1, sizeblok1_1;
		NumInt = sizeSbox / NumOfBits;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread1_1 = BLOCK_SIZE;
		sizeblok1_1 = NumInt / BLOCK_SIZE;

		//setgridBitwise(NumInt);

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		/////////////////////////////////////////////////////
		//@set grid for FMT kernel,  Algebraic Degree kernel and Butterfly max kernel
		int sizethread_11 = BLOCK_SIZE;
		int sizeblok_11 = NumInt / BLOCK_SIZE;
		int sizeblok1_11 = sizeblok_11;

		if (sizeblok_11>32)
			sizeblok1_11 = 32;

		int sizefor_11 = 32;
		int sizefor1_11 = NumInt;
		//////////////////////////////////////////////////////

		for (int i = 0; i < sizeSbox; i++)
		{

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_v03 << < sizeblok1_1, sizethread1_1 >> >(device_NumIntVecCF, device_CF, NumOfBits);

			///////////////////////////////////////////////////
			fmt_bitwise_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11*sizeof(unsigned long long int) >> >(device_NumIntVecCF, device_NumIntVecANF, sizefor_11, sizefor1_11);
			if (NumInt>1024)
				fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok_11, sizethread_11, sizethread_11*sizeof(unsigned long long int) >> >(device_NumIntVecANF, sizeblok_11, sizeblok1_11);
			///////////////////////////////////////////////////

			//kernel (Bitwise) Algebraic Degree
			kernel_bitwise_AD << < sizeblok_11, sizethread_11 >> >(device_NumIntVecANF, device_Vec_max_values, NumOfBits);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread_11);
			if (NumInt > 1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, sizeblok_11, sizeblok1_11);
			//////////////////////////////////
		}

		//@return (bitwise) Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

		int set_zero = 0;
		cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Bitwise Algebraic Degree use Min Butterfly v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMin_v03(unsigned long long int* device_NumIntVecCF, unsigned long long int* device_NumIntVecANF, int* device_Vec_max_values, int* device_Sbox, int* device_CF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSboxBitwise(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8; //set number of bits
	int NumInt = 0, min = 0; //min variable

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		NumInt = (sizeSbox * sizeSbox) / NumOfBits; //compute number of Integers

		//@ For S-box size <=1024
		if (sizeSbox < 2048)
		{
			//@set GRID
			sizethread = sizeSbox;
			sizeblok = sizeSbox;

			//@Compute Component function GPU - BoolSPLG Library function
			ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, sizeSbox);
		}
		//@ For S-box size > 1024
		else
		{
			//@set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			//cout << "Component function (device):" << "\n";
			for (int i = 0; i < sizeSbox; i++)
			{
				//@Compute Component function GPU - call ComponentFnAllVecIn_kernel_v03 and full device_CF vector
				ComponentFnAllVecIn_kernel_v03 << <sizeblok, sizethread >> > (device_Sbox, device_CF, i, sizeSbox);
			}
		}

		//@set GRID for Binary to Decimal kernel
		setgrid_BinVecToDec(sizeSbox, NumInt);

		//@Kernel Binary to Decimal conversion
		BinVecToDec_kernel_v03 << < sizeblok, sizethread >> > (device_NumIntVecCF, device_CF, NumOfBits);

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt, NumOfBits);

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);

		if (NumInt > 256)
		{
			//@ Set grid for Butterfly min kernel
			//setgrid(NumInt);
			sizeblok = sizeSbox;
			sizethread = NumInt/sizeSbox;
				
			//printf("NumInt: %d, sizethread: %d, sizeblok: %d \n", NumInt, sizethread, sizeblok);

			//call Butterfly max-min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM_min << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
			//if (NumInt > 1024)

			//@ Set grid for Butterfly max kernel
			setgrid(sizeSbox);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
			if (sizeSbox > 1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, sizeblok, sizeblok1);

			cudaMemcpy(&min, &device_Vec_max_values[sizeSbox - 1], sizeof(int), cudaMemcpyDeviceToHost);

		}
		else
		{
			int* host_max_values = (int*)malloc(sizeof(int) * NumInt);

			//memcpy device to host
			cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int) * NumInt, cudaMemcpyDeviceToHost);

			min = reduceCPU_min_deg(host_max_values, NumInt);

			free(host_max_values);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072))
	{
		int move = 0; //Declaration of variable used for setting writing output results of S-box component function
		NumInt = sizeSbox / NumOfBits; //variable - number of Integer per component function
		int sizethread_1, sizeblok_1;	//additional vatiable used for kernel paramethers

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread_1 = NumInt;
		sizeblok_1 = 1;


		for (int i = 0; i < sizeSbox; i++)
		{
			//variable used for setting writing output results
			move = i * NumInt;

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_move_v03 << < sizeblok_1, sizethread_1 >> > (device_NumIntVecCF, device_CF, NumOfBits, move);
		}

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt * sizeSbox, NumOfBits);

		//@Compute ANF of S-box
		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);

		//Call Max Reduction function
		//min = runReductionMax(NumInt * sizeSbox, device_Vec_max_values);
		
		//@ Set grid for Butterfly min kernel
		sizeblok = sizeSbox;
		sizethread = sizeSbox / NumOfBits;

		//cout << "\nNumOfInt:" << NumInt << "\n";
		//cout << "sizeblok:" << sizeblok << "\n";
		//cout << "sizethread:" << sizethread << "\n";
		//
		//call Butterfly max-min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM_min_bitwise << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);


		//@ Set grid for Butterfly max kernel
		setgrid(sizeSbox);

		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
		if (sizeSbox > 1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, sizeblok, sizeblok1);

		cudaMemcpy(&min, &device_Vec_max_values[sizeSbox - 1], sizeof(int), cudaMemcpyDeviceToHost);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		int sizethread1_1, sizeblok1_1;
		NumInt = sizeSbox / NumOfBits;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread1_1 = BLOCK_SIZE;
		sizeblok1_1 = NumInt / BLOCK_SIZE;

		//setgridBitwise(NumInt);

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		/////////////////////////////////////////////////////
		//@set grid for FMT kernel,  Algebraic Degree kernel and Butterfly max kernel
		int sizethread_11 = BLOCK_SIZE;
		int sizeblok_11 = NumInt / BLOCK_SIZE;
		int sizeblok1_11 = sizeblok_11;

		if (sizeblok_11 > 32)
			sizeblok1_11 = 32;

		int sizefor_11 = 32;
		int sizefor1_11 = NumInt;
		//////////////////////////////////////////////////////

		for (int i = 1; i < sizeSbox; i++)
		{

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_v03 << < sizeblok1_1, sizethread1_1 >> > (device_NumIntVecCF, device_CF, NumOfBits);

			///////////////////////////////////////////////////
			fmt_bitwise_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor_11, sizefor1_11);
			if (NumInt > 1024)
				fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(unsigned long long int) >> > (device_NumIntVecANF, sizeblok_11, sizeblok1_11);
			///////////////////////////////////////////////////

			//kernel (Bitwise) Algebraic Degree
			kernel_bitwise_AD << < sizeblok_11, sizethread_11 >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread_11);
			if (NumInt > 1024)
				//Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, sizeblok_11, sizeblok1_11);
				Butterfly_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, sizeblok_11, sizeblok1_11);
			//////////////////////////////////
		}

		//@return (bitwise) Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&min, global_min, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_min"

		int set_value = 32;
		cudaMemcpyToSymbol(global_min, &set_value, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_min=32"
	}

	return min;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Bitwise Algebraic Degree use Max Butterfly (ANF 'in') v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeSboxGPU_ANF_in_ButterflyMax_v03(unsigned long long int* device_NumIntVecCF, int* device_Vec_max_values, int* device_Sbox_ANF, int* device_CF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSboxBitwise(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8; //set number of bits
	int NumInt = 0, max = 0; //max variable

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		NumInt = (sizeSbox * sizeSbox) / NumOfBits; //compute number of Integers

		//@ For S-box size <=1024
		if (sizeSbox < 2048)
		{
			//@set GRID
			sizethread = sizeSbox;
			sizeblok = sizeSbox;

			//@Compute Component function GPU - BoolSPLG Library function
			ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, sizeSbox);
		}
		//@ For S-box size > 1024
		else
		{
			//@set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			//cout << "Component function (device):" << "\n";
			for (int i = 0; i < sizeSbox; i++)
			{
				//@Compute Component function GPU - call ComponentFnAllVecIn_kernel_v03 and full device_CF vector
				ComponentFnAllVecIn_kernel_v03 << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i, sizeSbox);
			}
		}

		//@set GRID for Binary to Decimal kernel 
		setgrid_BinVecToDec(sizeSbox, NumInt);

		//@Kernel Binary to Decimal conversion 
		BinVecToDec_kernel_v03 << < sizeblok, sizethread >> > (device_NumIntVecCF, device_CF, NumOfBits);

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt, NumOfBits);

		//@Compute ANF of S-box
//		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		//kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecCF, device_Vec_max_values, NumOfBits);

		if (NumInt > 256)
		{
			//@ Set grid for Butterfly max kernel
			setgrid(NumInt);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
			if (NumInt > 1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, sizeblok, sizeblok1);

			//memcpy device to host
			cudaMemcpy(&max, &device_Vec_max_values[0], sizeof(int), cudaMemcpyDeviceToHost);

		}
		else
		{
			int *host_max_values = (int*)malloc(sizeof(int) * NumInt);
			//memcpy device to host
			cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int) * NumInt, cudaMemcpyDeviceToHost);

			max = reduceCPU_max_libhelp(host_max_values, NumInt);

			free(host_max_values);

		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072))
	{
		int move = 0; //Declaration of variable used for setting writing output results of S-box component function
		NumInt = sizeSbox / NumOfBits; //variable - number of Integer per component function
		int sizethread_1, sizeblok_1;	//additional vatiable used for kernel paramethers

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread_1 = NumInt;
		sizeblok_1 = 1;


		for (int i = 0; i < sizeSbox; i++)
		{
			//variable used for setting writing output results
			move = i * NumInt;

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_move_v03 << < sizeblok_1, sizethread_1 >> > (device_NumIntVecCF, device_CF, NumOfBits, move);
		}

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt * sizeSbox, NumOfBits);

		//@Compute ANF of S-box
//		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		//kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecCF, device_Vec_max_values, NumOfBits);

		//Call Max Reduction function
		max = runReductionMax(NumInt * sizeSbox, device_Vec_max_values);

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		int sizethread1_1, sizeblok1_1;
		NumInt = sizeSbox / NumOfBits;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread1_1 = BLOCK_SIZE;
		sizeblok1_1 = NumInt / BLOCK_SIZE;

		//setgridBitwise(NumInt);

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		/////////////////////////////////////////////////////
		//@set grid for FMT kernel,  Algebraic Degree kernel and Butterfly max kernel
		int sizethread_11 = BLOCK_SIZE;
		int sizeblok_11 = NumInt / BLOCK_SIZE;
		int sizeblok1_11 = sizeblok_11;

		if (sizeblok_11 > 32)
			sizeblok1_11 = 32;

		//int sizefor_11 = 32;
		//int sizefor1_11 = NumInt;
		//////////////////////////////////////////////////////

		for (int i = 0; i < sizeSbox; i++)
		{

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_v03 << < sizeblok1_1, sizethread1_1 >> > (device_NumIntVecCF, device_CF, NumOfBits);

			///////////////////////////////////////////////////
//			fmt_bitwise_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor_11, sizefor1_11);
//			if (NumInt > 1024)
//				fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(unsigned long long int) >> > (device_NumIntVecANF, sizeblok_11, sizeblok1_11);
			///////////////////////////////////////////////////

			//kernel (Bitwise) Algebraic Degree 
			//kernel_bitwise_AD << < sizeblok_11, sizethread_11 >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);
			kernel_bitwise_AD << < sizeblok_11, sizethread_11 >> > (device_NumIntVecCF, device_Vec_max_values, NumOfBits);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread_11);
			if (NumInt > 1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, sizeblok_11, sizeblok1_11);
			//////////////////////////////////
		}

		//@return (bitwise) Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

		int set_zero = 0;
		cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Bitwise Algebraic Degree use Max Butterfly (ANF 'in') v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int BitwiseAlgebraicDegreeSboxGPU_ANF_in_ButterflyMin_v03(unsigned long long int* device_NumIntVecCF, int* device_Vec_max_values, int* device_Sbox_ANF, int* device_CF, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSboxBitwise(sizeSbox);

	int NumOfBits = sizeof(unsigned long long int) * 8; //set number of bits
	int NumInt = 0, min = 0; //min variable

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox < 16384) //limitation come from function Butterfly_max_min_kernel, where can fit 2^26/64 numbers
	{
		NumInt = (sizeSbox * sizeSbox) / NumOfBits; //compute number of Integers

		//@ For S-box size <=1024
		if (sizeSbox < 2048)
		{
			//@set GRID
			sizethread = sizeSbox;
			sizeblok = sizeSbox;

			//@Compute Component function GPU - BoolSPLG Library function
			ComponentFnAll_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, sizeSbox);
		}
		//@ For S-box size > 1024
		else
		{
			//@set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			//cout << "Component function (device):" << "\n";
			for (int i = 0; i < sizeSbox; i++)
			{
				//@Compute Component function GPU - call ComponentFnAllVecIn_kernel_v03 and full device_CF vector
				ComponentFnAllVecIn_kernel_v03 << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i, sizeSbox);
			}
		}

		//@set GRID for Binary to Decimal kernel 
		setgrid_BinVecToDec(sizeSbox, NumInt);

		//@Kernel Binary to Decimal conversion 
		BinVecToDec_kernel_v03 << < sizeblok, sizethread >> > (device_NumIntVecCF, device_CF, NumOfBits);

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt, NumOfBits);

		//@Compute ANF of S-box
//		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		//kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecCF, device_Vec_max_values, NumOfBits);

		if (NumInt > 256)
		{
			//@ Set grid for Butterfly min kernel
			//setgrid(NumInt);
			sizeblok = sizeSbox;
			sizethread = NumInt / sizeSbox;

			//printf("NumInt: %d, sizethread: %d, sizeblok: %d \n", NumInt, sizethread, sizeblok);

			//call Butterfly max-min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM_min << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
			//if (NumInt > 1024)

			//@ Set grid for Butterfly max kernel
			setgrid(sizeSbox);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
			if (sizeSbox > 1024)
				Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, sizeblok, sizeblok1);

			cudaMemcpy(&min, &device_Vec_max_values[sizeSbox - 1], sizeof(int), cudaMemcpyDeviceToHost);

		}
		else
		{
			int* host_max_values = (int*)malloc(sizeof(int) * NumInt);

			//memcpy device to host
			cudaMemcpy(host_max_values, device_Vec_max_values, sizeof(int) * NumInt, cudaMemcpyDeviceToHost);

			min = reduceCPU_min_deg(host_max_values, NumInt);

			free(host_max_values);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if ((sizeSbox > 8192) & (sizeSbox < 131072))
	{
		int move = 0; //Declaration of variable used for setting writing output results of S-box component function
		NumInt = sizeSbox / NumOfBits; //variable - number of Integer per component function
		int sizethread_1, sizeblok_1;	//additional vatiable used for kernel paramethers

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread_1 = NumInt;
		sizeblok_1 = 1;


		for (int i = 0; i < sizeSbox; i++)
		{
			//variable used for setting writing output results
			move = i * NumInt;

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_move_v03 << < sizeblok_1, sizethread_1 >> > (device_NumIntVecCF, device_CF, NumOfBits, move);
		}

		//Set Bitwise S-box GRID for FMT and Algebraic Degree
		setgrid_fmt_ad(sizeSbox, NumInt * sizeSbox, NumOfBits);

		//@Compute ANF of S-box
//		fmt_bitwise_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor, sizefor1);

		//kernel for computation of bitwise Algebraic Degree on input integer vector
		//kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);
		kernel_bitwise_AD_Sbox << < sizeblok, sizethread >> > (device_NumIntVecCF, device_Vec_max_values, NumOfBits);

		//Call Max Reduction function
		//min = runReductionMax(NumInt * sizeSbox, device_Vec_max_values);

		//@ Set grid for Butterfly min kernel
		sizeblok = sizeSbox;
		sizethread = sizeSbox / NumOfBits;

		//cout << "\nNumOfInt:" << NumInt << "\n";
		//cout << "sizeblok:" << sizeblok << "\n";
		//cout << "sizethread:" << sizethread << "\n";

		//call Butterfly max-min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM_min_bitwise << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);


		//@ Set grid for Butterfly max kernel
		setgrid(sizeSbox);

		//call Butterfly max min kernel
		/////////////////////////////////////////////////////
		Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread);
		if (sizeSbox > 1024)
			Butterfly_max_min_kernel_shfl_xor_SM_MP << < sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Vec_max_values, sizeblok, sizeblok1);

		cudaMemcpy(&min, &device_Vec_max_values[sizeSbox - 1], sizeof(int), cudaMemcpyDeviceToHost);

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (sizeSbox >= 131072)
	{
		int sizethread1_1, sizeblok1_1;
		NumInt = sizeSbox / NumOfBits;

		//@set grid for Kernel Binary to Decimal conversion GPU
		sizethread1_1 = BLOCK_SIZE;
		sizeblok1_1 = NumInt / BLOCK_SIZE;

		//setgridBitwise(NumInt);

		//@set grid for Compute Component function GPU
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		/////////////////////////////////////////////////////
		//@set grid for FMT kernel,  Algebraic Degree kernel and Butterfly max kernel
		int sizethread_11 = BLOCK_SIZE;
		int sizeblok_11 = NumInt / BLOCK_SIZE;
		int sizeblok1_11 = sizeblok_11;

		if (sizeblok_11 > 32)
			sizeblok1_11 = 32;

		//int sizefor_11 = 32;
		//int sizefor1_11 = NumInt;
		//////////////////////////////////////////////////////

		for (int i = 1; i < sizeSbox; i++)
		{

			//@Compute Component function GPU
			ComponentFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox_ANF, device_CF, i);

			//@Kernel Binary to Decimal conversion GPU
			BinVecToDec_kernel_v03 << < sizeblok1_1, sizethread1_1 >> > (device_NumIntVecCF, device_CF, NumOfBits);

			///////////////////////////////////////////////////
//			fmt_bitwise_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(unsigned long long int) >> > (device_NumIntVecCF, device_NumIntVecANF, sizefor_11, sizefor1_11);
//			if (NumInt > 1024)
//				fmt_bitwise_kernel_shfl_xor_SM_MP << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(unsigned long long int) >> > (device_NumIntVecANF, sizeblok_11, sizeblok1_11);
			///////////////////////////////////////////////////

			//kernel (Bitwise) Algebraic Degree 
			//kernel_bitwise_AD << < sizeblok_11, sizethread_11 >> > (device_NumIntVecANF, device_Vec_max_values, NumOfBits);
			kernel_bitwise_AD << < sizeblok_11, sizethread_11 >> > (device_NumIntVecCF, device_Vec_max_values, NumOfBits);

			//call Butterfly max min kernel
			/////////////////////////////////////////////////////
			Butterfly_max_min_kernel_shfl_xor_SM << <sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, device_Vec_max_values, sizethread_11);
			if (NumInt > 1024)
				//Butterfly_max_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, sizeblok_11, sizeblok1_11);
				Butterfly_min_kernel_shfl_xor_SM_MP_Sbox_v03 << < sizeblok_11, sizethread_11, sizethread_11 * sizeof(int) >> > (device_Vec_max_values, sizeblok_11, sizeblok1_11);
			//////////////////////////////////
		}

		//@return (bitwise) Algebraic Degree of the S-box
		cudaMemcpyFromSymbol(&min, global_min, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_min"

		int set_value = 32;
		cudaMemcpyToSymbol(global_min, &set_value, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_min=32"
	}

	return min;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Autocorrelation Transform (ACT) - Autocorrelation (AC)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranSboxGPU(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0, returnMax=0; //max variable
	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ACT of S-box
		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ACT, sizethread);

		powInt << < sizeblok, sizethread >> >(device_ACT, 2);

		ifmt_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_ACT, device_ACT, sizethread);

		cudaMemset(device_ACT, 0, sizeSbox*sizeof(int)); //clear first row of ACT !!!

		//@Reduction Max return ACT of the S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_ACT);
	}

	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			//@Compute Component function GPU
			ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

			//@Return AC(S) of component function of the S-box
			returnMax = AutocorrelationTranBoolGPU(device_CF, device_ACT, sizeSbox, true);

			if (returnMax>max)
				max = returnMax;
		}
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Autocorrelation Transform (ACT) use Max Butterfly - Autocorrelation (AC)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranSboxGPU_ButterflyMax(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ACT of S-box

		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ACT, sizethread);

		powInt << < sizeblok, sizethread >> >(device_ACT, 2);

		ifmt_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_ACT, device_ACT, sizethread);

		if (returnMax)
		{
			int max = 0; //max variable

			cudaMemset(device_ACT, 0, sizeSbox*sizeof(int)); //clear first row of ACT !!!

			//use Max Butterfly return ACT of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_ACT);

			return max;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (returnMax)
		{
			int max = 0, MaxReturn=0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				//@Compute Component function GPU
				ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

				//@Return AC(S) of component function of the S-box
				MaxReturn = AutocorrelationTranBoolGPU_ButterflyMax(device_CF, device_ACT, sizeSbox, true);

				if (MaxReturn>max)
					max = MaxReturn;
			}

			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Autocorrelation Transform (ACT) use Max Butterfly - Autocorrelation (AC) v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int AutocorrelationTranSboxGPU_ButterflyMax_v03(int *device_Sbox, int *device_CF, int *device_ACT, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//@set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		//@Compute Component function GPU
		ComponentFnAll_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, sizeSbox);

		//@Compute ACT of S-box

		fwt_kernel_shfl_xor_SM << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_CF, device_ACT, sizethread);

		powInt << < sizeblok, sizethread >> >(device_ACT, 2);

		ifmt_kernel_shfl_xor_SM_Sbox << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_ACT, device_ACT, sizethread);

		if (returnMax)
		{
			int max = 0; //max variable

			cudaMemset(device_ACT, 0, sizeSbox*sizeof(int)); //clear first row of ACT !!!

			//use Max Butterfly return ACT of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_ACT);

			return max;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (returnMax)
		{
			int max = 0;// MaxReturn = 0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				//@Compute Component function GPU
				ComponentFnVec_kernel_PTT << <sizeblok, sizethread >> >(device_Sbox, device_CF, i);

				//@Compute AC of S-box component function
				AutocorrelationTranS_boxGPU_ButterflyMax_03(device_CF, device_ACT, sizeSbox, true);

			}
			//@Return AC(S) of the S-box
			cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

			int set_zero = 0;
			cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"

			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Difference Distribution Table (DDT) - Differential uniformity
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DDTSboxGPU(int *device_Sbox, int *device_DDT, int sizeSbox)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0, maxReturn = 0; //max variable

	if (sizeSbox <= BLOCK_SIZE)
	{
		//set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		cudaMemset(device_DDT, 0, sizeSbox*sizeSbox*sizeof(int));
		DDTFnAll_kernel << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Sbox, device_DDT, sizeSbox);

		//return Max reduction

		cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear first row of DDT !!!

		//return Max reduction diff(S) of S-box
		max = runReductionMax(sizeSbox*sizeSbox, device_DDT);

		return max;
	}
	//////////////////////////////////////////////////////
	else
	{
		//set GRID
		sizethread = BLOCK_SIZE;
		sizeblok = sizeSbox / BLOCK_SIZE;

		for (int i = 1; i < sizeSbox; i++)
		{
			cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear DDT !!!
			DDTFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_DDT, i);

			//return Max reduction diff(S) of S-box
			maxReturn = runReductionMax(sizeSbox, device_DDT);

			if (maxReturn > max)
				max = maxReturn;
		}

		return max;
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Difference Distribution Table (DDT) use Max Butterfly - Differential uniformity
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DDTSboxGPU_ButterflyMax(int *device_Sbox, int *device_DDT, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		cudaMemset(device_DDT, 0, sizeSbox*sizeSbox*sizeof(int));
		DDTFnAll_kernel << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Sbox, device_DDT, sizeSbox);

		if (returnMax)
		{
			int max = 0; //max variable
			cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear first row of DDT !!!

			//use Max Butterfly return delta of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_DDT);

			return max;
		}
		else
		{
			return 0;
		}
	}
	//////////////////////////////////////////////////////
	else
	{
		if (returnMax)
		{
			int max = 0, maxReturn=0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear DDT !!!
				DDTFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_DDT, i);

				//use Max Butterfly return delta of the S-box component function
				maxReturn = Butterfly_max_kernel(sizeSbox, device_DDT);

				if (maxReturn > max)
					max = maxReturn;
			}

			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Difference Distribution Table (DDT) use Max Butterfly - Differential uniformity v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DDTSboxGPU_ButterflyMax_v03(int *device_Sbox, int *device_DDT, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	if (sizeSbox <= BLOCK_SIZE)
	{
		//set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		cudaMemset(device_DDT, 0, sizeSbox*sizeSbox*sizeof(int));
		DDTFnAll_kernel << <sizeblok, sizethread, sizethread*sizeof(int) >> >(device_Sbox, device_DDT, sizeSbox);

		if (returnMax)
		{
			int max = 0; //max variable
			cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear first row of DDT !!!

			//use Max Butterfly return delta of the S-box
			max = Butterfly_max_kernel(sizeSbox*sizeSbox, device_DDT);

			return max;
		}
		else
		{
			return 0;
		}
	}
	//////////////////////////////////////////////////////
	else
	{
		if (returnMax)
		{
			int max = 0; //max variable

			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				cudaMemset(device_DDT, 0, sizeSbox*sizeof(int)); //clear DDT !!!
				DDTFnVec_kernel << <sizeblok, sizethread >> >(device_Sbox, device_DDT, i);

				//use Max Butterfly to find max delta of the S-box component function
				Butterfly_max_kernel_DDT_v03(sizeSbox, device_DDT);

			}

			//@return Diff(S) of the S-box
			cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

			int set_zero = 0;
			cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"

			return max;
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Compute S-box Difference Distribution Table (DDT) use Max Butterfly - Differential uniformity v0.3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
int DDTSboxGPU_ButterflyMax_v03_expand(int* device_Sbox, int* device_DDT, int sizeSbox, bool returnMax)
{
	//Check S-box the input size
	CheckSizeSbox(sizeSbox);

	int max = 0; //max variable

	if (sizeSbox <= BLOCK_SIZE)
	{
		//set GRID
		sizethread = sizeSbox;
		sizeblok = sizeSbox;

		cudaMemset(device_DDT, 0, sizeSbox * sizeSbox * sizeof(int));
		DDTFnAll_kernel << <sizeblok, sizethread, sizethread * sizeof(int) >> > (device_Sbox, device_DDT, sizeSbox);

		if (returnMax)
		{
			//int max = 0; //max variable
			cudaMemset(device_DDT, 0, sizeSbox * sizeof(int)); //clear first row of DDT !!!

			//use Max Butterfly return delta of the S-box
			max = Butterfly_max_kernel(sizeSbox * sizeSbox, device_DDT);

		}
	}
	//////////////////////////////////////////////////////
	if ((sizeSbox > 1024) & (sizeSbox <= 16384))
	{
		//set GRID
		int sizethread = BLOCK_SIZE;
		int sizeblok = (sizeSbox * sizeSbox) / BLOCK_SIZE;

		cudaMemset(device_DDT, 0, sizeSbox * sizeSbox * sizeof(int));
		DDTFnAll_kernel_expand << <sizeblok, sizethread >> > (device_Sbox, device_DDT, sizeSbox);

		if (returnMax)
		{
			//int max = 0; //max variable
			cudaMemset(device_DDT, 0, sizeSbox * sizeof(int)); //clear first row of DDT !!!

			//use Max Butterfly return delta of the S-box
			//max = Butterfly_max_kernel(sizeSbox * sizeSbox, device_DDT);

			//return Max reduction
			max=runReductionMax(sizeSbox * sizeSbox, device_DDT);
		}
	}
	//////////////////////////////////////////////////////
	if (sizeSbox > 16384)
	{
		if (returnMax)
		{
			//set GRID
			sizethread = BLOCK_SIZE;
			sizeblok = sizeSbox / BLOCK_SIZE;

			for (int i = 1; i < sizeSbox; i++)
			{
				cudaMemset(device_DDT, 0, sizeSbox * sizeof(int)); //clear DDT !!!
				DDTFnVec_kernel << <sizeblok, sizethread >> > (device_Sbox, device_DDT, i);

				//use Max Butterfly to find max delta of the S-box component function
				Butterfly_max_kernel_DDT_v03(sizeSbox, device_DDT);
			}

			//@return Diff(S) of the S-box
			cudaMemcpyFromSymbol(&max, global_max, sizeof(int), 0, cudaMemcpyDeviceToHost); //read global variable "global_max"

			int set_zero = 0;
			cudaMemcpyToSymbol(global_max, &set_zero, sizeof(int), 0, cudaMemcpyHostToDevice); //set global variable "global_max=0"
		}
		else
		{
			cout << "\nIs not implemented this funcionality for S-box size n>10. \nThe output data is to big.\n";
			return 0;
		}
	}

	return max;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

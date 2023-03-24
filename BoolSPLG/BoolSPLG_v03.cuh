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

//@@ Main Library header file

//System includes Librarys
#include <stdio.h>
#include <iostream>
#include <string>

//CUDA runtime
#include "cuda_runtime.h"

#define BLOCK_SIZE 1024

//@ Global variable for grid
int sizeblok, sizeblok1, sizethread, sizefor, sizefor1;
int numThreadsRDM, numBlocksRDM, whichKernelRDM; //data for reduction Max operation

//BoolSPLG CUDA Properties header file
#include "CUDA_Properties.h"

//BoolSPLG base CUDA functions
#include "boolsplg_base_funct.cuh"

//BoolSPLG Boolean device functions
#include "boolsplg_dev_funct.cuh"

//BoolSPLG GPU reduction function heder file
#include "reduction.h"

//BoolSPLG Boolean Algorithms
#include "boolsplg_algorithms.cuh"

//Heder file 2D DynamicArray
#include "2D_DynamicArray.h"

//BoolSPLG Host Boolean functions
#include "boolsplg_host_boolean_funct.h"
#include "boolsplg_host_boolean_funct_v03.h"

//BoolSPLG Host S-box functions
#include "boolsplg_host_sbox_funct.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////
//Declaration of function
/////////////////////////////////////////////////////////////////////////

////Declaration for CUDA Properties function
void printDevProp(cudaDeviceProp devProp); //v0.1
void BoolSPLGCheckProperties(); //v0.1
void BoolSPLGMinimalRequires(); //v0.1
void BoolSPLGCheckProperties_v1(); //v0.2
void bandwidthTest(const unsigned int bytes, int nElements); //v0.2

//Declaration for CPU help function
int reduceCPU_max_libhelp(int* vals, int nvals); //v0.1
int reduceCPU_min_libhelp(int* vals, int nvals); //v0.1

//Declaration for base CUDA functions
//void cudaMallocBoolSPLG(int **d_vec, int sizeBoolean);
//void cudaMemcpyBoolSPLG_HtoD(int *d_vec, int *h_vec, int sizeBoolean);

//Function: Set GRID
inline void setgrid(int size); //v0.1
inline void setgridBitwise(int size); //v0.2
inline void setgrid_BinVecToDec(int size, int NumInt); //v0.3
inline void setgrid_fmt_ad(int sizeSbox, int NumInt, int NumOfBits); //v0.3

//Function: Return Most significant bit start from 0
unsigned int msb32(unsigned int x); //v0.1

//Function: Check array size
inline void CheckSize(int size); //v0.1
inline void CheckSizeSbox(int size); //v0.2
inline void CheckSizeBoolBitwise(int size); //v0.2
inline void CheckSizeSboxBitwise(int size); //v0.2
inline void CheckSizeSboxBitwiseMobius(int size); //v0.2

//Function: CPU set TT in 64 bit int variables and vise versa
void BinVecToDec(int size, int* Bin_Vec, unsigned long long int* NumIntVec, int NumInt); //v0.2
void DecVecToBin(int NumOfBits, int* Bin_Vec, unsigned long long int* NumIntVec, int NumInt); //v0.2

//Function:CPU set 32 binary input in integer data type variable(s)
void BinVecToDec32(int size, int* Bin_Vec, unsigned int* NumIntVec, int NumInt); //v0.3

//GPU Fast Walsh Transform
extern __global__ void fwt_kernel_shfl_xor_SM(int* VectorValue, int* VectorValueRez, int step); //v0.1
extern __global__ void fwt_kernel_shfl_xor_SM_MP(int* VectorValue, int fsize, int fsize1); //v0.1

extern __global__ void fwt_kernel_shfl_xor_SM_LAT(int* VectorValue, int* VectorValueRez, int step); //v0.3

template <class T>
__global__ void fwt_kernel_shfl_xor_SM_v03(int* VectorValue, T* VectorValueRez, int step); //v0.3
template <class T>
__global__ void fwt_kernel_shfl_xor_SM_MP_v03(T* VectorValue, int fsize, int fsize1); //v0.3

//GPU Fast Mobius Transform
extern __global__ void fmt_kernel_shfl_xor_SM(int* VectorValue, int* VectorRez, int sizefor); //v0.1
extern __global__ void fmt_kernel_shfl_xor_SM_MP(int* VectorValue, int fsize, int fsize1); //v0.1

//GPU Bitwise Fast Mobius Transform
extern __global__ void fmt_bitwise_kernel_shfl_xor_SM(unsigned long long int* vect, unsigned long long int* vect_out, int sizefor, int sizefor1); //v0.2
extern __global__ void fmt_bitwise_kernel_shfl_xor_SM_MP(unsigned long long int* VectorValue, int fsize, int fsize1); //v0.2

template <class T>
__global__ void fmt_kernel_shfl_xor_SM_v03(int* VectorValue, T* VectorRez, int sizefor); //v0.3
template <class T>
__global__ void fmt_kernel_shfl_xor_SM_MP_v03(T* VectorValue, int fsize, int fsize1); //v0.3

//GPU compute Algebraic Degree
extern __global__ void kernel_AD(int* Vec); //v0.1
extern __global__ void kernel_bitwise_AD(unsigned long long int* NumIntVec, int* Vec_max_values, int NumOfBits); //v0.2

//GPU Inverse Fast Walsh Transform
extern __global__ void ifmt_kernel_shfl_xor_SM(int* VectorValue, int* VectorValueRez, int step); //v0.1
extern __global__ void ifmt_kernel_shfl_xor_SM_MP(int* VectorValue, int fsize, int fsize1); //v0.1

template <class T>
__global__ void ifmt_kernel_shfl_xor_SM_v03(T* VectorValue, T* VectorValueRez, int step); //v0.3
template <class T>
__global__ void ifmt_kernel_shfl_xor_SM_MP_v03(T* VectorValue, int fsize, int fsize1); //v0.3

extern __global__ void ifmt_kernel_shfl_xor_SM_Sbox(int* VectorValue, int* VectorValueRez, int step); //v0.1 Invers Fast Walsh Transforms for S-box


//GPU Min-Max Butterfly
extern __global__ void Butterfly_max_min_kernel_shfl_xor_SM(int* VectorValue, int* VectorValueRez, int step); //v0.2
extern __global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP(int* VectorValue, int fsize, int fsize1); //v0.2

template <class T>
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_v03(T* VectorValue, T* VectorValueRez, int step); //v0.3
template <class T>
__global__ void Butterfly_max_min_kernel_shfl_xor_SM_MP_v03(T* VectorValue, int fsize, int fsize1); //v0.3


//GPU Bitwise Fast Mobius Transform
extern __global__ void fmt_bitwise_kernel_shfl_xor_SM_Sbox(unsigned long long int* vect, unsigned long long int* vect_out, int sizefor, int sizefor1); //v0.2

//GPU compute Algebraic Degree
extern __global__ void kernel_AD_Sbox(int* Vec); //v0.1
extern __global__ void kernel_bitwise_AD_Sbox(unsigned long long int* NumIntVecANF, int* max_values, int NumOfBits); //v0.2

//GPU Difference Distribution Table
extern __global__ void DDTFnAll_kernel(int* Sbox_in, int* DDT_out, int n); //v0.1
extern __global__ void DDTFnAll_kernel_expand(int* Sbox_in, int* DDT_out, int size); //v0.3
extern __global__ void DDTFnVec_kernel(int* Sbox_in, int* DDT_out, int row); //v0.1

//GPU S-box Component functions
extern __global__ void ComponentFnAll_kernel(int* Sbox_in, int* CF_out, int n); //v0.1
extern __global__ void ComponentFnVec_kernel(int* Sbox_in, int* CF_out, int row); //v0.1

extern __global__ void ComponentFnAll_kernel_PTT(int* Sbox_in, int* CF_out, int n);//v0.1
extern __global__ void ComponentFnVec_kernel_PTT(int* Sbox_in, int* CF_out, int row);//v0.1

//Declaration for Boolean procedures
int WalshSpecTranBoolGPU(int* device_Vect, int* device_Vect_rez, int size, bool returnMaxReduction); //v0.1 BoolFWT_compute
void MobiusTranBoolGPU(int* device_Vect, int* device_Vect_rez, int size);		//v0.1 BoolFMT_compute
int AlgebraicDegreeBoolGPU(int* device_Vect, int* device_Vect_rez, int size);	//v0.1 BoolAD_compute
int AutocorrelationTranBoolGPU(int* device_Vect, int* device_Vect_rez, int size, bool returnMaxReduction);	//v0.1 BoolAC_compute

int WalshSpecTranBoolGPU_ButterflyMax(int* device_Vect, int* device_Vect_rez, int size, bool returnMaxReduction); //v0.2 BoolFWT_compute
int AlgebraicDegreeBoolGPU_ButterflyMax(int* device_Vect, int* device_Vect_rez, int size);							 //v0.2 BoolAD_compute
int AutocorrelationTranBoolGPU_ButterflyMax(int* device_Vect, int* device_Vect_rez, int size, bool returnMaxReduction);  //v0.2 BoolAC_compute

template <class T>
int WalshSpecTranBoolGPU_ButterflyMax_v03(int* device_Vect, T* device_Vect_rez, T* device_Vect_Max, int size, bool returnMaxReduction); //v0.3 BoolFWT_compute
template <class T>
void MobiusTranBoolGPU_v03(int* device_Vect, T* device_Vect_rez, int size); //v0.3 BoolFMT_compute
template <class T>
int AlgebraicDegreeBoolGPU_ButterflyMax_v03(int* device_Vect, T* device_Vect_rez, int size); //v0.3 BoolAD_compute
template <class T>
int AutocorrelationTranBoolGPU_ButterflyMax_v03(int* device_Vect, T* device_Vect_rez, T* device_Vect_Max, int size, bool returnMaxReduction); //v0.3 BoolAC_compute

template <class T>
int runTestWalshSpec(int size, int* BoolElemet_host, T* vec_host_spectra, bool returnWS);
template <class T>
void runTestMobius(int size, int* BoolElemet_host, T* host_Vect_anf_tt03, bool returnMT);
template <class T>
int runTestDeg(int size, int* BoolElemet_host);
template <class T>
int runTestACT(int size, int* BoolElemet_host, T* host_Vect_Spectra03, bool returnACT);

//Declaration for Bitwise Boolean procedures
void BitwiseMobiusTranBoolGPU(unsigned long long int* device_Vect, unsigned long long int* device_Vect_rez, int size); //v0.2 Bitwise BoolFMT_compute
int BitwiseAlgebraicDegreeBoolGPU_ButterflyMax(unsigned long long int* device_Vect, unsigned long long int* device_Vect_rez, int* device_Vec_max_values, int* host_Vec_max_values, int size); //v0.2

//Declaration for S-box procedures
int WalshSpecTranSboxGPU(int* device_Sbox, int* device_CF, int* device_WST, int sizeSbox); //v0.1
int MobiusTranSboxADGPU(int* device_Sbox, int* device_CF, int* device_ANF, int sizeSbox); //v0.1
int AutocorrelationTranSboxGPU(int* device_Sbox, int* device_CF, int* device_ACT, int sizeSbox); //v0.1

int WalshSpecTranSboxGPU_ButterflyMax(int* device_Sbox, int* device_CF, int* device_WST, int sizeSbox, bool returnMax); //v0.2
int WalshSpecTranSboxGPU_ButterflyMax_v03(int* device_Sbox, int* device_CF, int* device_WST, int sizeSbox, bool returnMax); //v0.3

void LATSboxGPU_v03(int* device_Sbox, int* device_CF, int* device_LAT, int sizeSbox); //v0.3
void MobiusTranSboxGPU(int* device_Sbox, int* device_CF, int* device_ANF, int sizeSbox); //v0.2
int AutocorrelationTranSboxGPU_ButterflyMax(int* device_Sbox, int* device_CF, int* device_ACT, int sizeSbox, bool returnMax); //v0.2

// DDT(S) Difference Distribution Table, return δ Differential uniformity of S-box  
int DDTSboxGPU(int* device_Sbox, int* device_DDT, int sizeSbox); //v0.1
int DDTSboxGPU_ButterflyMax(int* device_Sbox, int* device_DDT, int sizeSbox, bool returnMax); //v0.2
int DDTSboxGPU_ButterflyMax_v03(int* device_Sbox, int* device_DDT, int sizeSbox, bool returnMax); //v0.3
int DDTSboxGPU_ButterflyMax_v03_expand(int* device_Sbox, int* device_DDT, int sizeSbox, bool returnMax); //v0.3

// return Algebraic Degree of Sbox, deg(S)
int AlgebraicDegreeSboxGPU_ButterflyMax(int* device_Sbox, int* device_CF, int* device_ANF, int sizeSbox); //v0.2
int AlgebraicDegreeSboxGPU_ButterflyMin(int* device_Sbox, int* device_CF, int* device_ANF, int sizeSbox); //v0.3

int AlgebraicDegreeSboxGPU_in_ANF_ButterflyMax(int* device_Sbox_ANF, int* device_CF, int sizeSbox); //v0.3
int AlgebraicDegreeSboxGPU_in_ANF_ButterflyMin(int* device_Sbox_ANF, int* device_CF, int sizeSbox); //v0.3

//ADT(S) Table with the Algebraic Degrees,
void AlgebraicDegreeTableSboxGPU(int* device_Sbox, int* device_CF, int* device_ADT, int sizeSbox); //v0.3

//Declaration for Bitwise S-box procedures
void BitwiseMobiusTranSboxGPU(int* host_Sbox, int* host_Vect_CF, unsigned long long int* host_NumIntVecCF, unsigned long long int* device_NumIntVecCF, unsigned long long int* device_NumIntVecANF, int sizeSbox); //v0.2
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMax(int* host_Sbox, int* host_Vect_CF, int* host_max_values, unsigned long long int* host_NumIntVecCF, unsigned long long int* device_NumIntVecCF, unsigned long long int* device_NumIntVecANF, int* device_Vec_max_values, int sizeSbox); //v0.2
int BitwiseAlgebraicDegreeSboxGPU_ButterflyMax_v03(unsigned long long int* device_NumIntVecCF, unsigned long long int* device_NumIntVecANF, int* device_Vec_max_values, int* device_Sbox, int* device_CF, int sizeSbox);//v0.3

int BitwiseAlgebraicDegreeSboxGPU_ANF_in_ButterflyMax_v03(unsigned long long int* device_NumIntVecCF, int* device_Vec_max_values, int* device_Sbox_ANF, int* device_CF, int sizeSbox); //v0.3
int BitwiseAlgebraicDegreeSboxGPU_ANF_in_ButterflyMin_v03(unsigned long long int* device_NumIntVecCF, int* device_Vec_max_values, int* device_Sbox_ANF, int* device_CF, int sizeSbox); //v0.3

//Declaration for Sum, Max and Min Reduction function
int runReduction(int size, int* d_idata); //v0.3
int runReductionMax(int size, int* d_idata); //v0.1
int runReductionMin(int size, int* d_idata); //v0.2

//Instantiate the reduction function
template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T* d_idata, T* d_odata);

template <class T>
void reduce_max(int size, int threads, int blocks, int whichKernel, T* d_idata, T* d_odata);

template <class T>
void reduce_min(int size, int threads, int blocks, int whichKernel, T* d_idata, T* d_odata);

//Declaration for Butterfly max function
int Butterfly_max_kernel(int sizeSbox, int* device_data); //v0.2

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration, Host Boolean functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void FastWalshTrans(int size, int* BoolSbox, int* walshvec);
void FastWalshTransInv(int size, int* walshvec);
void FastMobiushTrans(int size, int* TT, int* ANF);

int FindMaxDeg(int size, int* ANF_CPU);
void CPU_FMT_bitwise(unsigned long long int* NumIntVec, unsigned long long int* NumIntVecANF, int NumInt);

int DecVecToBin_maxDeg(int NumOfBits, unsigned long long int* NumIntVec, int NumInt);

int reduceCPU_AC(int nvals, int* AC);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration of Host S-box functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void SetSTT(int* sbox, int** STT, int* binary_num, int sizeSbox, int binary);
void SetS_ANF(int* sbox, int* SboxElemet_ANF, int sizeSbox, int binary);

int DDT_vector(int sizeSbox, int* sbox, int dx);

int AlgDegMax(int sizeSbox, int* ANF_CPU);
int AlgDegMin(int sizeSbox, int* ANF_CPU);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Declaration, Host additional support functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int ipow(int base, int exp);
void fun_pow2(int size, int* vec);

int reduceCPU_max(int* vals, int nvals);

void HistogramSpectrum(int size, int* HistohramVector, int* InputVector, string NameSpectrum);

//Allocate 2D Dynamic Array
template <typename T>
T** AllocateDynamicArray(int nRows, int nCols);

//Delete 2D Dynamic Array 
template <typename T>
void FreeDynamicArray(T** dArray);

void Fill_dp_vector(int n, int* vector, int* vecPTT);

void binary1(int number, int* binary_num, int binary);

void check_rez(int size, int* vector1, int* vector2);
void check_rez_int(int size, unsigned long long int* vector1, unsigned long long int* vector2);
template <class T>
void check_rez03(int size, T* vector1, int* vector2);

void Print_Result(int size, int* result);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////